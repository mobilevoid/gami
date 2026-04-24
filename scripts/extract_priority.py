import os
#!/usr/bin/env python3
"""Extract entities and claims from priority GAMI segments.

Targets memory files, infrastructure docs, and high-value conversation segments.
Uses vLLM (qwen35-27b-unredacted) for extraction with rate limiting.
"""

import json
import hashlib
import logging
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import quote_plus

import requests
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB_URL = os.getenv("DATABASE_URL", "postgresql://gami:gami@localhost:5432/gami")
VLLM_URL = "http://localhost:8000/v1"
EXTRACTION_MODEL = "qwen35-27b-unredacted"
TENANT_ID = "shared"
EXTRACTOR_VERSION = "1.0.0"
MAX_RATE = 2.0  # requests per second
MIN_INTERVAL = 1.0 / MAX_RATE  # 0.5s between requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("extract_priority")

# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

engine = create_engine(DB_URL, pool_size=3, pool_pre_ping=True)
Session = sessionmaker(engine, expire_on_commit=False)

# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

import re

def strip_thinking(text_str: str) -> str:
    if not text_str:
        return text_str
    text_str = re.sub(r"<think>.*?</think>", "", text_str, flags=re.DOTALL).strip()
    text_str = re.sub(r"<think>.*", "", text_str, flags=re.DOTALL).strip()
    return text_str


def parse_json_from_llm(text_str: str):
    if not text_str or not text_str.strip():
        return None
    raw = strip_thinking(text_str.strip())
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", raw, re.DOTALL)
    if fence_match:
        raw = fence_match.group(1).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    for start_char, end_char in [("[", "]"), ("{", "}")]:
        idx = raw.find(start_char)
        if idx == -1:
            continue
        candidate = raw[idx:]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        depth = 0
        in_string = False
        escape_next = False
        for i, c in enumerate(candidate):
            if escape_next:
                escape_next = False
                continue
            if c == "\\":
                escape_next = True
                continue
            if c == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if c in ("[", "{"):
                depth += 1
            elif c in ("]", "}"):
                depth -= 1
        if depth > 0:
            fixed = candidate
            for _ in range(depth):
                fixed = fixed.rstrip().rstrip(",")
                if start_char == "[":
                    fixed += "]"
                else:
                    fixed += "}"
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass
    return None


_last_request_time = 0.0

def call_vllm(prompt: str, system_prompt: str, max_tokens: int = 1024) -> str:
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - elapsed)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    payload = {
        "model": EXTRACTION_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    resp = requests.post(f"{VLLM_URL}/chat/completions", json=payload, timeout=300)
    _last_request_time = time.time()
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    return strip_thinking(content)


def _make_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


# ---------------------------------------------------------------------------
# Entity extraction prompt
# ---------------------------------------------------------------------------

ENTITY_PROMPT = """Extract all named entities from the following text. Return a JSON array.
Each entity must have these fields:
- "name": the canonical name of the entity
- "type": one of "person", "place", "organization", "technology", "service", "infrastructure", "concept", "credential"
- "aliases": array of alternate names or abbreviations (empty array if none)
- "description": one-line description

Rules:
- Include IP addresses, hostnames, container IDs (e.g. web-01), server names, software names
- Include credentials as type "credential" (passwords, API keys, tokens)
- Include people by name or role
- Do NOT include generic words like "system" or "server" unless they are specific named entities
- Deduplicate: if the same entity appears multiple times, include it only once

Text:
{text}

Return ONLY a valid JSON array, no other text."""

CLAIM_PROMPT = """Extract factual claims from the following text. Return a JSON array.
Each claim must have these fields:
- "subject": the entity or thing the claim is about
- "predicate": the relationship or property (e.g. "has password", "runs on", "is located at", "has IP")
- "object": the value or target of the claim
- "confidence": float 0.0-1.0 (how certain is this claim based on the text)
- "modality": one of "asserted" (stated as fact), "tentative" (might be true), "negated" (stated as not true)

Rules:
- Extract specific facts: IP addresses, passwords, versions, configurations, relationships
- Include infrastructure facts: "Server A has IP 10.0.0.5", "GitLab runs on prod-server"
- Include credentials: "Service X password is ExamplePass123"
- Include relationships: "Proxy routes traffic to backend-server"
- Be precise — copy exact values from the text

Text:
{text}

Return ONLY a valid JSON array, no other text."""


# ---------------------------------------------------------------------------
# Extraction functions
# ---------------------------------------------------------------------------

def extract_entities_from_text(db, text_content: str, source_id: str, segment_id: str) -> list[dict]:
    input_text = text_content[:3000]
    prompt = ENTITY_PROMPT.format(text=input_text)
    system_prompt = "You are a precise entity extraction system. Return ONLY valid JSON arrays. No explanations."

    try:
        raw = call_vllm(prompt, system_prompt)
        entities_raw = parse_json_from_llm(raw)
    except Exception as exc:
        log.error("Entity extraction failed for %s: %s", segment_id, exc)
        return []

    if not entities_raw or not isinstance(entities_raw, list):
        log.warning("No valid entity JSON for %s", segment_id)
        return []

    results = []
    now = datetime.now(timezone.utc)
    valid_types = {"person", "place", "organization", "technology", "service",
                   "infrastructure", "concept", "credential"}

    for ent in entities_raw:
        if not isinstance(ent, dict):
            continue
        name = (ent.get("name") or "").strip()
        etype = (ent.get("type") or "concept").strip().lower()
        if not name or len(name) < 2:
            continue
        if etype not in valid_types:
            etype = "concept"

        aliases = ent.get("aliases", [])
        if not isinstance(aliases, list):
            aliases = []
        description = (ent.get("description") or "")[:500]

        # Check existing
        existing = db.execute(
            text(
                "SELECT entity_id, mention_count FROM entities "
                "WHERE owner_tenant_id = :tid AND LOWER(canonical_name) = :name "
                "AND entity_type = :etype LIMIT 1"
            ),
            {"tid": TENANT_ID, "name": name.lower(), "etype": etype},
        ).fetchone()

        if existing:
            entity_id = existing[0]
            db.execute(
                text(
                    "UPDATE entities SET mention_count = :cnt, "
                    "last_seen_at = :now, updated_at = :now WHERE entity_id = :eid"
                ),
                {"cnt": existing[1] + 1, "now": now, "eid": entity_id},
            )
        else:
            entity_id = _make_id("ENT")
            db.execute(
                text(
                    "INSERT INTO entities "
                    "(entity_id, owner_tenant_id, entity_type, canonical_name, "
                    "aliases_json, description, importance_score, mention_count, "
                    "source_count, first_seen_at, last_seen_at, created_at, updated_at) "
                    "VALUES (:eid, :tid, :etype, :name, CAST(:aliases AS jsonb), :desc, "
                    "0.5, 1, 1, :now, :now, :now, :now)"
                ),
                {
                    "eid": entity_id, "tid": TENANT_ID, "etype": etype,
                    "name": name, "aliases": json.dumps(aliases),
                    "desc": description, "now": now,
                },
            )

        # Provenance
        prov_id = _make_id("PROV")
        db.execute(
            text(
                "INSERT INTO provenance "
                "(provenance_id, target_type, target_id, source_id, segment_id, "
                "extraction_method, extractor_version, confidence, created_at) "
                "VALUES (:pid, 'entity', :tid, :sid, :segid, "
                "'llm_extraction', :ver, 0.8, :now)"
            ),
            {
                "pid": prov_id, "tid": entity_id, "sid": source_id,
                "segid": segment_id, "ver": EXTRACTOR_VERSION, "now": now,
            },
        )

        results.append({
            "entity_id": entity_id, "name": name, "type": etype,
            "aliases": aliases, "description": description,
            "is_new": existing is None,
        })

    db.commit()
    return results


def extract_claims_from_text(db, text_content: str, source_id: str,
                              segment_id: str, entity_map: dict) -> list[dict]:
    input_text = text_content[:3000]
    prompt = CLAIM_PROMPT.format(text=input_text)
    system_prompt = "You are a precise claim extraction system. Return ONLY valid JSON arrays. No explanations."

    try:
        raw = call_vllm(prompt, system_prompt)
        claims_raw = parse_json_from_llm(raw)
    except Exception as exc:
        log.error("Claim extraction failed for %s: %s", segment_id, exc)
        return []

    if not claims_raw or not isinstance(claims_raw, list):
        log.warning("No valid claim JSON for %s", segment_id)
        return []

    results = []
    now = datetime.now(timezone.utc)

    for claim in claims_raw:
        if not isinstance(claim, dict):
            continue
        subject = (claim.get("subject") or "").strip()
        predicate = (claim.get("predicate") or "").strip()
        obj = (claim.get("object") or "").strip()
        confidence = float(claim.get("confidence", 0.7))
        modality = (claim.get("modality") or "asserted").lower()

        if not subject or not predicate:
            continue
        if modality not in ("asserted", "tentative", "negated", "hypothetical"):
            modality = "asserted"

        subj_entity_id = entity_map.get(subject.lower())
        obj_entity_id = entity_map.get(obj.lower()) if obj else None

        claim_id = _make_id("CLM")
        summary_text = f"{subject} {predicate} {obj}" if obj else f"{subject} {predicate}"

        db.execute(
            text(
                "INSERT INTO claims "
                "(claim_id, owner_tenant_id, subject_entity_id, predicate, "
                "object_entity_id, object_literal_json, modality, confidence, "
                "summary_text, salience_score, novelty_score, support_count, "
                "created_at, updated_at) "
                "VALUES (:cid, :tid, :subj_eid, :pred, :obj_eid, "
                "CAST(:obj_lit AS jsonb), :mod, :conf, :summary, "
                "0.5, 0.5, 1, :now, :now)"
            ),
            {
                "cid": claim_id, "tid": TENANT_ID, "subj_eid": subj_entity_id,
                "pred": predicate, "obj_eid": obj_entity_id,
                "obj_lit": json.dumps({"value": obj}) if obj else None,
                "mod": modality, "conf": confidence,
                "summary": summary_text[:500], "now": now,
            },
        )

        # Provenance
        prov_id = _make_id("PROV")
        db.execute(
            text(
                "INSERT INTO provenance "
                "(provenance_id, target_type, target_id, source_id, segment_id, "
                "extraction_method, extractor_version, confidence, created_at) "
                "VALUES (:pid, 'claim', :tid, :sid, :segid, "
                "'llm_extraction', :ver, :conf, :now)"
            ),
            {
                "pid": prov_id, "tid": claim_id, "sid": source_id,
                "segid": segment_id, "ver": EXTRACTOR_VERSION,
                "conf": confidence, "now": now,
            },
        )

        results.append({
            "claim_id": claim_id, "subject": subject, "predicate": predicate,
            "object": obj, "confidence": confidence, "modality": modality,
        })

    db.commit()
    return results


# ---------------------------------------------------------------------------
# Segment selection
# ---------------------------------------------------------------------------

def select_priority_segments(db) -> list[dict]:
    """Select top ~30 segments for extraction, ordered by priority."""
    segments = []

    # Priority 1: Memory .md files (MEMORY.md variants) — richest infra data
    rows = db.execute(text(
        "SELECT s.segment_id, s.source_id, s.text, src.title "
        "FROM segments s JOIN sources src ON s.source_id = src.source_id "
        "WHERE src.title = 'MEMORY.md' AND length(s.text) > 300 "
        "ORDER BY length(s.text) DESC LIMIT 8"
    )).fetchall()
    for r in rows:
        segments.append({"segment_id": r[0], "source_id": r[1], "text": r[2], "title": r[3], "priority": 1})

    # Priority 2: network-infrastructure.md
    rows = db.execute(text(
        "SELECT s.segment_id, s.source_id, s.text, src.title "
        "FROM segments s JOIN sources src ON s.source_id = src.source_id "
        "WHERE src.title = 'network-infrastructure.md' AND length(s.text) > 300 "
        "ORDER BY length(s.text) DESC LIMIT 5"
    )).fetchall()
    for r in rows:
        segments.append({"segment_id": r[0], "source_id": r[1], "text": r[2], "title": r[3], "priority": 2})

    # Priority 3: CLAUDE.md and AI-STACK.md
    rows = db.execute(text(
        "SELECT s.segment_id, s.source_id, s.text, src.title "
        "FROM segments s JOIN sources src ON s.source_id = src.source_id "
        "WHERE (src.title LIKE '%CLAUDE.md%' OR src.title = 'AI-STACK.md') "
        "AND length(s.text) > 300 "
        "ORDER BY length(s.text) DESC LIMIT 5"
    )).fetchall()
    for r in rows:
        segments.append({"segment_id": r[0], "source_id": r[1], "text": r[2], "title": r[3], "priority": 3})

    # Priority 4: wan-failover.md
    rows = db.execute(text(
        "SELECT s.segment_id, s.source_id, s.text, src.title "
        "FROM segments s JOIN sources src ON s.source_id = src.source_id "
        "WHERE src.title = 'wan-failover.md' AND length(s.text) > 300 "
        "ORDER BY length(s.text) DESC LIMIT 3"
    )).fetchall()
    for r in rows:
        segments.append({"segment_id": r[0], "source_id": r[1], "text": r[2], "title": r[3], "priority": 4})

    # Priority 5: Clawd/OpenClaw memory DBs
    rows = db.execute(text(
        "SELECT s.segment_id, s.source_id, s.text, src.title "
        "FROM segments s JOIN sources src ON s.source_id = src.source_id "
        "WHERE (src.title LIKE '%Clawdbot%' OR src.title LIKE '%OpenClaw%') "
        "AND length(s.text) > 300 "
        "ORDER BY length(s.text) DESC LIMIT 5"
    )).fetchall()
    for r in rows:
        segments.append({"segment_id": r[0], "source_id": r[1], "text": r[2], "title": r[3], "priority": 5})

    # Priority 6: Other high-value markdown docs
    rows = db.execute(text(
        "SELECT s.segment_id, s.source_id, s.text, src.title "
        "FROM segments s JOIN sources src ON s.source_id = src.source_id "
        "WHERE src.title IN ("
        "  'acme-certificates.md', 'dcs-platform-credentials.md', "
        "  'servers.md', 'HARDWARE.md', 'local-ai-models.md'"
        ") AND length(s.text) > 300 "
        "ORDER BY length(s.text) DESC LIMIT 5"
    )).fetchall()
    for r in rows:
        segments.append({"segment_id": r[0], "source_id": r[1], "text": r[2], "title": r[3], "priority": 6})

    # Deduplicate by segment_id
    seen = set()
    unique = []
    for s in segments:
        if s["segment_id"] not in seen:
            seen.add(s["segment_id"])
            unique.append(s)

    return unique[:35]


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------

def main():
    db = Session()
    try:
        # Select segments
        segments = select_priority_segments(db)
        log.info("Selected %d priority segments for extraction", len(segments))

        # Group by priority for reporting
        by_priority = {}
        for s in segments:
            by_priority.setdefault(s["priority"], []).append(s)
        for p in sorted(by_priority):
            titles = set(s["title"] for s in by_priority[p])
            log.info("  Priority %d: %d segments from %s", p, len(by_priority[p]), titles)

        # Track all extracted entities globally (for entity_map building)
        global_entity_map = {}  # lowercase name -> entity_id
        total_entities = 0
        total_claims = 0
        new_entities = 0
        new_claims = 0
        entity_samples = []
        claim_samples = []
        errors = 0

        # Phase 1: Entity extraction on all segments
        log.info("=" * 60)
        log.info("PHASE 1: Entity extraction on %d segments", len(segments))
        log.info("=" * 60)

        for i, seg in enumerate(segments):
            log.info("[%d/%d] Extracting entities from %s / %s (%d chars)",
                     i + 1, len(segments), seg["title"], seg["segment_id"],
                     len(seg["text"]))
            try:
                entities = extract_entities_from_text(
                    db, seg["text"], seg["source_id"], seg["segment_id"]
                )
                total_entities += len(entities)
                for e in entities:
                    if e["is_new"]:
                        new_entities += 1
                    global_entity_map[e["name"].lower()] = e["entity_id"]
                    for alias in e.get("aliases", []):
                        if alias:
                            global_entity_map[alias.lower()] = e["entity_id"]
                    if len(entity_samples) < 30:
                        entity_samples.append(e)
                log.info("  -> %d entities (%d new)", len(entities),
                         sum(1 for e in entities if e["is_new"]))
            except Exception as exc:
                errors += 1
                log.error("  -> ERROR: %s", exc)
                continue

        # Phase 2: Claim extraction on top 50 segments
        claim_segments = segments[:15]
        log.info("=" * 60)
        log.info("PHASE 2: Claim extraction on %d segments", len(claim_segments))
        log.info("=" * 60)

        for i, seg in enumerate(claim_segments):
            log.info("[%d/%d] Extracting claims from %s / %s (%d chars)",
                     i + 1, len(claim_segments), seg["title"], seg["segment_id"],
                     len(seg["text"]))
            try:
                claims = extract_claims_from_text(
                    db, seg["text"], seg["source_id"], seg["segment_id"],
                    global_entity_map
                )
                total_claims += len(claims)
                new_claims += len(claims)
                for c in claims:
                    if len(claim_samples) < 30:
                        claim_samples.append(c)
                log.info("  -> %d claims", len(claims))
            except Exception as exc:
                errors += 1
                log.error("  -> ERROR: %s", exc)
                continue

        # Final report
        log.info("=" * 60)
        log.info("EXTRACTION COMPLETE")
        log.info("=" * 60)

        # Count totals in DB
        db_entities = db.execute(text("SELECT COUNT(*) FROM entities")).fetchone()[0]
        db_claims = db.execute(text("SELECT COUNT(*) FROM claims")).fetchone()[0]
        db_provenance = db.execute(text("SELECT COUNT(*) FROM provenance")).fetchone()[0]

        log.info("Database totals:")
        log.info("  Entities: %d", db_entities)
        log.info("  Claims:   %d", db_claims)
        log.info("  Provenance records: %d", db_provenance)
        log.info("")
        log.info("This run:")
        log.info("  Segments processed: %d (entities) + %d (claims)", len(segments), len(claim_segments))
        log.info("  Entity extractions: %d total, %d new", total_entities, new_entities)
        log.info("  Claim extractions: %d total", total_claims)
        log.info("  Errors: %d", errors)

        log.info("")
        log.info("SAMPLE ENTITIES:")
        for e in entity_samples:
            log.info("  [%s] %s - %s%s",
                     e["type"], e["name"], e["description"][:60],
                     " (NEW)" if e["is_new"] else "")

        log.info("")
        log.info("SAMPLE CLAIMS:")
        for c in claim_samples:
            log.info("  %s | %s | %s (conf=%.2f, %s)",
                     c["subject"], c["predicate"], c["object"],
                     c["confidence"], c["modality"])

        # Show entity type distribution
        type_dist = db.execute(text(
            "SELECT entity_type, COUNT(*) FROM entities GROUP BY entity_type ORDER BY COUNT(*) DESC"
        )).fetchall()
        log.info("")
        log.info("ENTITY TYPE DISTRIBUTION:")
        for t in type_dist:
            log.info("  %s: %d", t[0], t[1])

    finally:
        db.close()


if __name__ == "__main__":
    main()
