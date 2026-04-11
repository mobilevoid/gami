#!/usr/bin/env python3
"""Enrich entities with better descriptions and extract claims from entity contexts.

Runs after the dream extraction to:
1. Upgrade short/generic entity descriptions using their segment contexts
2. Extract specific claims (not just "has_password" but full operational facts)
3. Remove remaining garbage entities that slipped through
"""
import os, sys, time, json, logging, re, hashlib, requests
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.config import settings
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                   handlers=[logging.FileHandler("/tmp/enrich_entities.log"), logging.StreamHandler()])
log = logging.getLogger("enrich")
engine = create_engine(settings.DATABASE_URL_SYNC)

VLLM_URL = settings.VLLM_URL


def call_vllm(prompt, max_tokens=1500):
    try:
        from api.llm.vllm_monitor import emit as _emit
    except Exception:
        _emit = None
    t0 = time.time()
    try:
        r = requests.post(f"{VLLM_URL}/chat/completions", json={
            "model": "qwen35-27b-unredacted",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.1,
        }, timeout=120)
        if r.status_code == 200:
            content = r.json()["choices"][0]["message"]["content"]
            tokens = r.json().get("usage", {}).get("completion_tokens", 0)
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            lines = content.split('\n')
            clean_lines = []
            in_thinking = False
            for line in lines:
                stripped = line.strip()
                if re.match(r'^(Thinking Process|Analysis|Step \d|\d+\.\s+\*\*|\*\*\w)', stripped):
                    in_thinking = True
                    continue
                if in_thinking and (stripped.startswith('*') or stripped.startswith('-') or re.match(r'^\d+\.', stripped)):
                    continue
                if stripped and not stripped.startswith('*') and len(stripped) > 20:
                    in_thinking = False
                    clean_lines.append(stripped)

            result = ' '.join(clean_lines).strip()
            result = re.sub(r'\*\*([^*]+)\*\*', r'\1', result)
            result = re.sub(r'\*([^*]+)\*', r'\1', result)
            if _emit:
                _emit("enrich", prompt, result or content, (time.time()-t0)*1000, "ok", tokens)
            return result if result else content
    except Exception as e:
        log.warning(f"vLLM call failed: {e}")
        if _emit:
            _emit("enrich", prompt, "", (time.time()-t0)*1000, "error", 0, str(e))
    return None


def find_json(response, bracket='{'):
    """Find JSON object or array in response text (handles thinking preamble)."""
    close = '}' if bracket == '{' else ']'
    for i in range(len(response) - 1, -1, -1):
        if response[i] == close:
            for j in range(i, -1, -1):
                if response[j] == bracket:
                    try:
                        result = json.loads(response[j:i+1])
                        return result
                    except:
                        continue
            break
    return None


def gen_id(prefix, name):
    h = hashlib.md5(name.encode()).hexdigest()[:8]
    clean = re.sub(r'[^a-zA-Z0-9_]', '_', name)[:30]
    return f"{prefix}_{clean}_{h}"


def enrich_top_entities(limit=50):
    """Upgrade descriptions for top entities by gathering their segment contexts."""
    log.info(f"Enriching top {limit} entities...")

    with engine.connect() as conn:
        # Get top entities by mention count that have short descriptions
        entities = conn.execute(text("""
            SELECT e.entity_id, e.canonical_name, e.entity_type,
                   e.description, e.mention_count, e.owner_tenant_id
            FROM entities e
            WHERE e.status = 'active'
            AND e.owner_tenant_id = 'claude-opus'
            AND (e.description IS NULL OR length(e.description) < 60)
            ORDER BY e.mention_count DESC
            LIMIT :lim
        """), {"lim": limit}).fetchall()

        log.info(f"  Found {len(entities)} entities with short descriptions")
        upgraded = 0

        for ent in entities:
            # Get segments mentioning this entity
            segs = conn.execute(text("""
                SELECT s.text FROM segments s
                JOIN provenance p ON p.segment_id = s.segment_id
                WHERE p.target_id = :eid AND p.target_type = 'entity'
                ORDER BY s.created_at DESC LIMIT 3
            """), {"eid": ent.entity_id}).fetchall()

            if not segs:
                continue

            context = "\n---\n".join(s[0][:500] for s in segs)

            prompt = f"""Based on these text excerpts about "{ent.canonical_name}" ({ent.entity_type}), write a comprehensive one-paragraph description.

Current description: {ent.description or 'None'}

Context excerpts:
{context[:2000]}

Write a specific, factual description (2-3 sentences). Include IPs, ports, versions, roles if mentioned. Output ONLY the description text, nothing else."""

            response = call_vllm(prompt, max_tokens=300)
            if not response or len(response) < 20:
                continue

            # Clean: remove thinking preamble, take the actual description
            # Find the first sentence-like content
            lines = [l.strip() for l in response.split('\n') if l.strip() and not l.strip().startswith('*')]
            if lines:
                desc = ' '.join(lines)
                # Remove any remaining thinking markers
                desc = re.sub(r'^(Thinking Process:|Analysis:|Description:)\s*', '', desc, flags=re.IGNORECASE)
                if len(desc) > len(ent.description or '') and len(desc) > 30:
                    conn.execute(text("""
                        UPDATE entities SET description = :desc, updated_at = NOW()
                        WHERE entity_id = :eid
                    """), {"desc": desc[:500], "eid": ent.entity_id})
                    upgraded += 1
                    if upgraded <= 5:
                        log.info(f"  Upgraded: {ent.canonical_name} → {desc[:80]}")

        conn.commit()
        log.info(f"  Upgraded {upgraded}/{len(entities)} entity descriptions")
        return upgraded


def extract_rich_claims(limit=100):
    """Extract specific operational claims from high-value segments."""
    log.info(f"Extracting rich claims from top segments...")

    with engine.connect() as conn:
        # Get segments that have entities but few claims
        segs = conn.execute(text("""
            SELECT s.segment_id, s.text, s.source_id, count(p.provenance_id) as ent_count
            FROM segments s
            JOIN provenance p ON p.segment_id = s.segment_id AND p.target_type = 'entity'
            WHERE s.owner_tenant_id = 'claude-opus'
            AND length(s.text) BETWEEN 200 AND 2000
            AND s.segment_type NOT IN ('tool_call', 'tool_result')
            GROUP BY s.segment_id, s.text, s.source_id
            HAVING count(p.provenance_id) >= 2
            ORDER BY count(p.provenance_id) DESC
            LIMIT :lim
        """), {"lim": limit}).fetchall()

        log.info(f"  Found {len(segs)} entity-rich segments")
        claims_created = 0

        for seg in segs:
            prompt = f"""Extract specific factual claims from this text. Each claim should be a complete, self-contained statement of fact.

Focus on: configurations, relationships between systems, procedures, credentials, IP addresses, ports, versions.
NOT generic statements. Each claim should be specific enough to be useful for answering a question.

Text:
{seg.text[:2000]}

Return a JSON array of strings, each a specific factual claim:
["claim 1", "claim 2", ...]"""

            response = call_vllm(prompt, max_tokens=1500)
            if not response:
                continue

            claims = find_json(response, '[')
            if not claims or not isinstance(claims, list):
                continue

            for claim_text in claims:
                if not isinstance(claim_text, str) or len(claim_text) < 20:
                    continue

                cid = gen_id("CLM_rich", claim_text[:40])
                conn.execute(text("""
                    INSERT INTO claims (claim_id, owner_tenant_id, predicate, summary_text,
                        confidence, modality, status)
                    VALUES (:cid, 'claude-opus', 'operational_fact', :txt, 0.8, 'extracted', 'active')
                    ON CONFLICT (claim_id) DO NOTHING
                """), {"cid": cid, "txt": claim_text[:500]})

                # Provenance
                pid = gen_id("PROV", f"{cid}_{seg.segment_id}")
                conn.execute(text("""
                    INSERT INTO provenance (provenance_id, target_type, target_id,
                        source_id, segment_id, extraction_method, extractor_version, confidence)
                    VALUES (:pid, 'claim', :cid, :src, :seg, 'enrich_claims', 'v1', 0.8)
                    ON CONFLICT (provenance_id) DO NOTHING
                """), {"pid": pid, "cid": cid, "src": seg.source_id, "seg": seg.segment_id})

                claims_created += 1

            conn.commit()

        log.info(f"  Created {claims_created} rich claims from {len(segs)} segments")
        return claims_created


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--entities", type=int, default=50, help="Entities to enrich")
    parser.add_argument("--claims", type=int, default=100, help="Segments to extract claims from")
    args = parser.parse_args()

    log.info("Starting entity enrichment + claim extraction...")
    t0 = time.time()
    upgraded = enrich_top_entities(args.entities)
    claims = extract_rich_claims(args.claims)
    log.info(f"Done in {(time.time()-t0)/60:.1f}min: {upgraded} entities upgraded, {claims} claims created")
