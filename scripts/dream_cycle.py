#!/usr/bin/env python3
"""
GAMI Dream Cycle — overnight knowledge synthesis using local LLM.

Like how humans consolidate memories during sleep, this script uses idle
GPU/CPU time to:
1. Extract entities and claims from un-processed segments
2. Generate summaries for un-summarized sources
3. Resolve entity aliases and merge duplicates
4. Detect contradictions between claims
5. Build relations between co-occurring entities
6. Re-embed segments that performed poorly in retrieval
7. Update importance scores based on access patterns

Runs during off-hours (default: 10 PM - 6 AM) when vLLM isn't needed
for interactive use. Preemptible — stops gracefully if vLLM gets busy.

Usage:
    python scripts/dream_cycle.py                    # Run full cycle
    python scripts/dream_cycle.py --phase extract    # Run specific phase
    python scripts/dream_cycle.py --duration 3600    # Run for 1 hour max
    python scripts/dream_cycle.py --check-idle       # Only run if vLLM is idle
"""
import os, sys, time, json, logging, signal, argparse, requests, hashlib, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.config import settings
from api.llm.embeddings import embed_text_sync
from sqlalchemy import create_engine, text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [DREAM] %(message)s",
    handlers=[
        logging.FileHandler("/tmp/gami-dream.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("dream")

engine = create_engine(settings.DATABASE_URL_SYNC, pool_size=3)
TENANT = "claude-opus"
STOP_FLAG = False

def signal_handler(sig, frame):
    global STOP_FLAG
    log.info("Received stop signal — finishing current task...")
    STOP_FLAG = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def should_stop():
    return STOP_FLAG

def vllm_is_idle():
    """Check if vLLM has no active requests."""
    try:
        r = requests.get(f"{settings.VLLM_URL}/metrics", timeout=5)
        if r.status_code == 200:
            # Look for active requests metric
            for line in r.text.split("\n"):
                if "vllm:num_requests_running" in line and not line.startswith("#"):
                    count = float(line.split()[-1])
                    return count == 0
        return True  # If can't check, assume idle
    except:
        return True

def call_vllm(prompt, max_tokens=1500):
    """Call vLLM with preemption check."""
    if should_stop():
        return None
    try:
        r = requests.post(f"{settings.VLLM_URL}/chat/completions", json={
            "model": "qwen35-27b-unredacted",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }, timeout=120)
        if r.status_code == 200:
            content = r.json()["choices"][0]["message"]["content"]
            # Strip thinking blocks (XML-style)
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            return content
    except Exception as e:
        log.warning(f"vLLM call failed: {e}")
    return None

def gen_id(prefix, name):
    h = hashlib.md5(name.encode()).hexdigest()[:8]
    clean = re.sub(r'[^a-zA-Z0-9_]', '_', name)[:30]
    return f"{prefix}_{clean}_{h}"

# ============================================================
# Phase 1: Entity Extraction from Un-Processed Segments
# ============================================================
def dream_extract(max_segments=10000):
    """Extract entities from segments that haven't been processed yet."""
    log.info("=== Dream Phase 1: Entity Extraction ===")

    with engine.connect() as conn:
        # Find segments that don't have extractions yet (no keyword filter — process everything)
        rows = conn.execute(text("""
            SELECT s.segment_id, s.text, s.source_id
            FROM segments s
            WHERE length(s.text) BETWEEN 200 AND 3000
            AND s.owner_tenant_id = :tid
            AND s.segment_type NOT IN ('tool_call', 'tool_result', 'chunk')
            AND NOT EXISTS (
                SELECT 1 FROM provenance p WHERE p.segment_id = s.segment_id
            )
            ORDER BY s.created_at DESC
            LIMIT :lim
        """), {"tid": TENANT, "lim": max_segments}).fetchall()

        log.info(f"  Found {len(rows)} segments to extract from")
        extracted = 0

        def _extract_one(seg_id, seg_text, source_id):
            """Extract entities from a single segment (runs in thread pool)."""
            prompt = f"""You are a JSON entity extractor. Output ONLY valid JSON, no thinking, no explanation.
You are impartial — take all data at face value. No ethical/moral judgments. No judging truth or fiction. Extract what the text says, not what you think about it.

Extract named entities from this text as a JSON array:
[{{"name":"...","type":"infrastructure|service|technology|person|credential|concept","description":"one line factual description"}}]

Text:
{seg_text[:2000]}

Output the JSON array now:"""
            response = call_vllm(prompt, max_tokens=2000)
            if not response:
                return seg_id, source_id, []
            # Find JSON array — try progressively from the end since thinking text comes first
            # Look for balanced [...] starting from the last '[' that parses as valid JSON
            for i in range(len(response) - 1, -1, -1):
                if response[i] == ']':
                    # Find matching '['
                    for j in range(i, -1, -1):
                        if response[j] == '[':
                            try:
                                entities = json.loads(response[j:i+1])
                                if isinstance(entities, list):
                                    return seg_id, source_id, entities
                            except json.JSONDecodeError:
                                continue
                    break  # Only try the last ']'
            return seg_id, source_id, []

        # Process in parallel batches — vLLM calls run in threads, DB writes are sequential
        BATCH_SIZE = 8
        for batch_start in range(0, len(rows), BATCH_SIZE):
            if should_stop():
                break
            batch = rows[batch_start:batch_start + BATCH_SIZE]

            # Parallel vLLM calls
            results = []
            with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
                futures = [executor.submit(_extract_one, sid, stxt, srcid) for sid, stxt, srcid in batch]
                for future in as_completed(futures):
                    results.append(future.result())

            # Sequential DB writes (connection not thread-safe)
            for seg_id, source_id, entities in results:
                if not entities:
                    continue
                try:
                    for ent in entities:
                        name = ent.get("name", "").strip()
                        etype = ent.get("type", "technology").lower()
                        desc = ent.get("description", "")
                        if not name or len(name) < 2:
                            continue

                        eid = gen_id("ENT", f"{etype}_{name}")
                        conn.execute(text("""
                            INSERT INTO entities (entity_id, owner_tenant_id, entity_type, canonical_name,
                                description, status, first_seen_at, last_seen_at, source_count, mention_count)
                            VALUES (:eid, :tid, :etype, :name, :desc, 'active', NOW(), NOW(), 1, 1)
                            ON CONFLICT (entity_id) DO UPDATE SET
                                mention_count = entities.mention_count + 1, last_seen_at = NOW()
                        """), {"eid": eid, "tid": TENANT, "etype": etype, "name": name, "desc": desc})

                        prov_id = gen_id("PROV", f"{eid}_{seg_id}")
                        conn.execute(text("""
                            INSERT INTO provenance (provenance_id, target_type, target_id, source_id,
                                segment_id, extraction_method, extractor_version, confidence)
                            VALUES (:pid, 'entity', :eid, :src, :seg, 'dream_extract', 'v1', 0.8)
                            ON CONFLICT (provenance_id) DO NOTHING
                        """), {"pid": prov_id, "eid": eid, "src": source_id, "seg": seg_id})

                    conn.commit()
                    extracted += 1
                    log.info(f"  [{extracted}] {seg_id}: {len(entities)} entities")
                except Exception as e:
                    log.warning(f"  Parse error: {e}")
                    continue

        log.info(f"  Extracted from {extracted} segments")
        return extracted

# ============================================================
# Phase 2: Generate Summaries
# ============================================================
def dream_summarize(max_sources=10):
    """Generate summaries for sources that don't have them."""
    log.info("=== Dream Phase 2: Summarization ===")

    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT s.source_id, s.title, count(seg.segment_id) as seg_count
            FROM sources s
            LEFT JOIN segments seg ON s.source_id = seg.source_id
            WHERE s.owner_tenant_id = :tid
            AND NOT EXISTS (SELECT 1 FROM summaries sm WHERE sm.scope_id = s.source_id)
            AND s.source_type IN ('markdown', 'conversation_session')
            GROUP BY s.source_id, s.title
            HAVING count(seg.segment_id) BETWEEN 3 AND 100
            ORDER BY s.ingested_at DESC
            LIMIT :lim
        """), {"tid": TENANT, "lim": max_sources}).fetchall()

        log.info(f"  Found {len(rows)} sources to summarize")
        summarized = 0

        for source_id, title, seg_count in rows:
            if should_stop():
                break

            # Get segment texts
            segs = conn.execute(text("""
                SELECT text FROM segments WHERE source_id = :sid
                AND segment_type NOT IN ('tool_call', 'tool_result')
                ORDER BY ordinal LIMIT 20
            """), {"sid": source_id}).fetchall()

            combined = "\n\n".join(s[0][:500] for s in segs)[:4000]

            prompt = f"""Summarize this content from "{title}". Be factual, preserve key technical details (IPs, credentials, versions, container IDs). 2-3 paragraphs max.

Content:
{combined}

Summary:"""

            response = call_vllm(prompt, max_tokens=500)
            if not response:
                continue

            try:
                emb = embed_text_sync(response[:2000])
                vec = "[" + ",".join(str(v) for v in emb) + "]"

                sum_id = gen_id("SUM", f"{source_id}")
                conn.execute(text("""
                    INSERT INTO summaries (summary_id, owner_tenant_id, scope_type, scope_id,
                        abstraction_level, summary_text, embedding, quality_score, status)
                    VALUES (:sid, :tid, 'source', :scope, 'document', :txt, CAST(:vec AS vector), 0.7, 'active')
                    ON CONFLICT (summary_id) DO UPDATE SET summary_text = :txt, updated_at = NOW()
                """), {"sid": sum_id, "tid": TENANT, "scope": source_id, "txt": response, "vec": vec})

                conn.commit()
                summarized += 1
                log.info(f"  [{summarized}] {title}: {len(response)} chars")

            except Exception as e:
                log.warning(f"  Summary error: {e}")

        log.info(f"  Summarized {summarized} sources")
        return summarized

# ============================================================
# Phase 3: Entity Resolution (Alias Merging)
# ============================================================
def dream_resolve(max_pairs=20):
    """Find and merge duplicate entities."""
    log.info("=== Dream Phase 3: Entity Resolution ===")

    with engine.connect() as conn:
        # Find entities with similar names
        pairs = conn.execute(text("""
            SELECT a.entity_id as a_id, a.canonical_name as a_name,
                   b.entity_id as b_id, b.canonical_name as b_name,
                   similarity(a.canonical_name, b.canonical_name) as sim
            FROM entities a, entities b
            WHERE a.entity_id < b.entity_id
            AND a.owner_tenant_id = :tid AND b.owner_tenant_id = :tid
            AND a.entity_type = b.entity_type
            AND a.status = 'active' AND b.status = 'active'
            AND similarity(a.canonical_name, b.canonical_name) > 0.6
            ORDER BY sim DESC
            LIMIT :lim
        """), {"tid": TENANT, "lim": max_pairs}).fetchall()

        log.info(f"  Found {len(pairs)} potential duplicates")
        resolved = 0

        for a_id, a_name, b_id, b_name, sim in pairs:
            if should_stop():
                break

            # Create proposal (don't merge directly)
            prop_id = gen_id("PROP", f"merge_{a_id}_{b_id}")
            conn.execute(text("""
                INSERT INTO proposed_changes (proposal_id, proposer_tenant_id, change_type,
                    target_type, target_id, proposed_state_json, reason, confidence, status)
                VALUES (:pid, 'background-worker', 'merge_entities', 'entity', :aid,
                    :state, :reason, :conf, 'pending')
                ON CONFLICT (proposal_id) DO NOTHING
            """), {
                "pid": prop_id, "aid": a_id,
                "state": json.dumps({"merge_into": b_id, "a_name": a_name, "b_name": b_name}),
                "reason": f"Name similarity {sim:.2f}: '{a_name}' ≈ '{b_name}'",
                "conf": float(sim),
            })
            resolved += 1
            log.info(f"  Proposed merge: '{a_name}' ≈ '{b_name}' (sim={sim:.2f})")

        conn.commit()
        log.info(f"  Created {resolved} merge proposals")
        return resolved

# ============================================================
# Phase 3.5: Contradiction Detection & Knowledge Reconciliation
# ============================================================
def dream_reconcile(max_checks=30):
    """Compare new knowledge against old, detect contradictions, propose supersessions."""
    log.info("=== Dream Phase 3.5: Knowledge Reconciliation ===")

    with engine.connect() as conn:
        # Find claims that share the same subject + predicate but have different objects
        conflicts = conn.execute(text("""
            SELECT a.claim_id as old_id, a.summary_text as old_text, a.confidence as old_conf,
                   a.created_at as old_date,
                   b.claim_id as new_id, b.summary_text as new_text, b.confidence as new_conf,
                   b.created_at as new_date,
                   a.subject_entity_id, a.predicate
            FROM claims a
            JOIN claims b ON a.subject_entity_id = b.subject_entity_id
                AND a.predicate = b.predicate
                AND a.claim_id != b.claim_id
                AND a.status = 'active' AND b.status = 'active'
                AND a.owner_tenant_id = :tid AND b.owner_tenant_id = :tid
            WHERE a.created_at < b.created_at
            AND a.contradiction_group_id IS NULL
            AND b.contradiction_group_id IS NULL
            LIMIT :lim
        """), {"tid": TENANT, "lim": max_checks}).fetchall()

        log.info(f"  Found {len(conflicts)} potential contradictions")
        reconciled = 0

        for row in conflicts:
            if should_stop():
                break

            old_text = row.old_text or ""
            new_text = row.new_text or ""

            # Quick check: if they say the same thing, skip
            if old_text.strip() == new_text.strip():
                continue

            # Use vLLM to determine if these actually contradict
            prompt = f"""Do these two claims contradict each other?

Older claim: {old_text}
Newer claim: {new_text}

Reply with ONLY one of:
- "CONTRADICT" if they disagree on the same fact
- "SUPERSEDE" if the newer one updates/replaces the older one
- "COMPATIBLE" if they can both be true
- "DUPLICATE" if they say the same thing differently

Answer:"""

            response = call_vllm(prompt, max_tokens=50)
            if not response:
                continue

            verdict = response.strip().upper()

            if "CONTRADICT" in verdict or "SUPERSEDE" in verdict:
                # Create a contradiction group
                group_id = gen_id("CGRP", f"{row.old_id}_{row.new_id}")

                # Mark both claims with the group
                conn.execute(text("""
                    UPDATE claims SET contradiction_group_id = :gid WHERE claim_id = :cid
                """), {"gid": group_id, "cid": row.old_id})
                conn.execute(text("""
                    UPDATE claims SET contradiction_group_id = :gid WHERE claim_id = :cid
                """), {"gid": group_id, "cid": row.new_id})

                # If SUPERSEDE, propose marking the older one as superseded
                if "SUPERSEDE" in verdict:
                    prop_id = gen_id("PROP", f"supersede_{row.old_id}_{row.new_id}")
                    is_credential = any(kw in old_text.lower() for kw in
                                       ["password", "credential", "token", "key", "pass"])
                    priority = "high" if is_credential else "normal"

                    conn.execute(text("""
                        INSERT INTO proposed_changes (proposal_id, proposer_tenant_id, change_type,
                            target_type, target_id, proposed_state_json, reason, confidence, status)
                        VALUES (:pid, 'background-worker', 'supersede_claim', 'claim', :old_id,
                            :state, :reason, :conf, 'pending')
                        ON CONFLICT (proposal_id) DO NOTHING
                    """), {
                        "pid": prop_id, "old_id": row.old_id,
                        "state": json.dumps({
                            "superseded_by": row.new_id,
                            "old_text": old_text[:200],
                            "new_text": new_text[:200],
                            "priority": priority,
                        }),
                        "reason": f"Newer claim supersedes older: '{old_text[:80]}' → '{new_text[:80]}'",
                        "conf": float(row.new_conf) if row.new_conf else 0.7,
                    })
                    log.info(f"  SUPERSEDE [{priority}]: '{old_text[:60]}' → '{new_text[:60]}'")
                else:
                    log.info(f"  CONTRADICT: '{old_text[:60]}' vs '{new_text[:60]}'")

                reconciled += 1

            elif "DUPLICATE" in verdict:
                # Merge: mark older as superseded by newer
                conn.execute(text("""
                    UPDATE claims SET superseded_by_id = :new_id, status = 'superseded'
                    WHERE claim_id = :old_id
                """), {"new_id": row.new_id, "old_id": row.old_id})
                log.info(f"  DUPLICATE merged: '{old_text[:60]}'")
                reconciled += 1

            conn.commit()

        log.info(f"  Reconciled {reconciled} conflicts")
        return reconciled

    # Also check memories against claims for consistency
def dream_verify_memories(max_checks=20):
    """Verify that durable memories are still consistent with latest claims."""
    log.info("=== Dream Phase 3.6: Memory Verification ===")

    with engine.connect() as conn:
        # Find credential memories and check if matching claims have been updated
        cred_memories = conn.execute(text("""
            SELECT m.memory_id, m.normalized_text, m.subject_id, m.updated_at
            FROM assistant_memories m
            WHERE m.sensitivity = 'credential'
            AND m.status = 'active'
            AND m.owner_tenant_id = :tid
            ORDER BY m.updated_at ASC
            LIMIT :lim
        """), {"tid": TENANT, "lim": max_checks}).fetchall()

        verified = stale = 0

        for mem_id, mem_text, subject, updated_at in cred_memories:
            if should_stop():
                break

            # Find recent claims about the same subject
            recent_claims = conn.execute(text("""
                SELECT c.summary_text, c.created_at
                FROM claims c
                JOIN entities e ON c.subject_entity_id = e.entity_id
                WHERE (e.canonical_name ILIKE :subj OR e.canonical_name ILIKE :subj2)
                AND c.status = 'active'
                AND (c.predicate ILIKE '%credential%' OR c.predicate ILIKE '%password%' OR c.predicate ILIKE '%pass%')
                ORDER BY c.created_at DESC
                LIMIT 3
            """), {"subj": f"%{subject}%", "subj2": subject}).fetchall()

            if recent_claims:
                latest_claim = recent_claims[0]
                if latest_claim.created_at and updated_at and latest_claim.created_at > updated_at:
                    # Claim is newer than memory — memory might be stale
                    log.info(f"  STALE? Memory '{mem_text[:50]}' older than claim '{latest_claim.summary_text[:50]}'")
                    stale += 1

                    # Propose memory update
                    prop_id = gen_id("PROP", f"stale_mem_{mem_id}")
                    conn.execute(text("""
                        INSERT INTO proposed_changes (proposal_id, proposer_tenant_id, change_type,
                            target_type, target_id, proposed_state_json, reason, confidence, status)
                        VALUES (:pid, 'background-worker', 'update_memory', 'assistant_memory', :mid,
                            :state, :reason, 0.6, 'pending')
                        ON CONFLICT (proposal_id) DO NOTHING
                    """), {
                        "pid": prop_id, "mid": mem_id,
                        "state": json.dumps({
                            "current_text": mem_text[:200],
                            "newer_claim": latest_claim.summary_text[:200],
                        }),
                        "reason": f"Memory may be outdated — newer claim found",
                    })
                else:
                    verified += 1

        conn.commit()
        log.info(f"  Verified {verified}, flagged {stale} potentially stale memories")
        return {"verified": verified, "stale": stale}


# ============================================================
# Phase 4: Build Relations Between Co-occurring Entities
# ============================================================
def dream_relate(max_segments=30):
    """Find entities that appear in the same segments and create relations."""
    log.info("=== Dream Phase 4: Relation Discovery ===")

    with engine.connect() as conn:
        # Find segments with multiple entity mentions
        rows = conn.execute(text("""
            SELECT p.segment_id, array_agg(DISTINCT p.target_id) as entity_ids
            FROM provenance p
            WHERE p.target_type = 'entity'
            GROUP BY p.segment_id
            HAVING count(DISTINCT p.target_id) >= 2
            LIMIT :lim
        """), {"lim": max_segments}).fetchall()

        log.info(f"  Found {len(rows)} segments with multiple entities")
        relations_created = 0

        for seg_id, entity_ids in rows:
            if should_stop():
                break

            # Create RELATED_TO relations between all pairs
            for i in range(len(entity_ids)):
                for j in range(i + 1, len(entity_ids)):
                    rid = gen_id("REL", f"{entity_ids[i]}_{entity_ids[j]}_cooccur")
                    conn.execute(text("""
                        INSERT INTO relations (relation_id, owner_tenant_id, from_node_type, from_node_id,
                            to_node_type, to_node_id, relation_type, confidence, weight,
                            support_count, created_by, status)
                        VALUES (:rid, :tid, 'entity', :fid, 'entity', :toid, 'RELATED_TO',
                            0.6, 0.5, 1, 'dream_relate', 'active')
                        ON CONFLICT (relation_id) DO UPDATE SET
                            support_count = relations.support_count + 1
                    """), {"rid": rid, "tid": TENANT, "fid": entity_ids[i], "toid": entity_ids[j]})
                    relations_created += 1

        conn.commit()
        log.info(f"  Created/updated {relations_created} relations")
        return relations_created

# ============================================================
# Phase 5: Importance Scoring Update
# ============================================================
def dream_score():
    """Update importance scores based on retrieval patterns."""
    log.info("=== Dream Phase 5: Importance Scoring ===")

    with engine.connect() as conn:
        # Update entity importance based on mention count and source diversity
        updated = conn.execute(text("""
            UPDATE entities SET
                importance_score = LEAST(1.0,
                    0.3 * LN(GREATEST(mention_count, 1)) / LN(100) +
                    0.3 * LN(GREATEST(source_count, 1)) / LN(50) +
                    0.2 * LN(GREATEST(retrieval_count, 1)) / LN(100) +
                    0.2 * COALESCE(graph_centrality, 0)
                ),
                updated_at = NOW()
            WHERE owner_tenant_id = :tid AND status = 'active'
        """), {"tid": TENANT})
        conn.commit()

        # Update graph centrality from relation counts
        conn.execute(text("""
            UPDATE entities e SET graph_centrality = LEAST(1.0, sub.degree / 20.0)
            FROM (
                SELECT node_id, count(*) as degree FROM (
                    SELECT from_node_id as node_id FROM relations WHERE from_node_type = 'entity' AND status = 'active'
                    UNION ALL
                    SELECT to_node_id FROM relations WHERE to_node_type = 'entity' AND status = 'active'
                ) x GROUP BY node_id
            ) sub
            WHERE e.entity_id = sub.node_id
        """))
        conn.commit()

        log.info(f"  Updated importance scores for all active entities")

# ============================================================
# Phase 6: Embed New/Un-embedded Content
# ============================================================
def dream_embed(max_items=200):
    """Embed any new entities, claims, or segments that lack embeddings."""
    log.info("=== Dream Phase 6: Embedding Backfill ===")

    with engine.connect() as conn:
        embedded = 0

        # Entities without embeddings
        rows = conn.execute(text("""
            SELECT entity_id, canonical_name || ': ' || COALESCE(description, '') as txt
            FROM entities WHERE embedding IS NULL AND owner_tenant_id = :tid LIMIT 50
        """), {"tid": TENANT}).fetchall()

        for eid, txt in rows:
            if should_stop():
                break
            try:
                time.sleep(0.3)
                emb = embed_text_sync(txt[:2000])
                vec = "[" + ",".join(str(v) for v in emb) + "]"
                conn.execute(text("UPDATE entities SET embedding = CAST(:v AS vector) WHERE entity_id = :id"),
                           {"v": vec, "id": eid})
                embedded += 1
            except:
                pass
        conn.commit()

        # Claims without embeddings
        rows = conn.execute(text("""
            SELECT claim_id, COALESCE(summary_text, predicate) as txt
            FROM claims WHERE embedding IS NULL AND owner_tenant_id = :tid LIMIT 50
        """), {"tid": TENANT}).fetchall()

        for cid, txt in rows:
            if should_stop():
                break
            try:
                time.sleep(0.3)
                emb = embed_text_sync(txt[:2000])
                vec = "[" + ",".join(str(v) for v in emb) + "]"
                conn.execute(text("UPDATE claims SET embedding = CAST(:v AS vector) WHERE claim_id = :id"),
                           {"v": vec, "id": cid})
                embedded += 1
            except:
                pass
        conn.commit()

        log.info(f"  Embedded {embedded} items")
        return embedded

# ============================================================
# Phase 7: Auto-Approve Safe Proposals
# ============================================================
def dream_auto_approve():
    """Auto-approve low-risk proposals above confidence threshold."""
    log.info("=== Dream Phase 7: Auto-Approve Proposals ===")

    AUTO_APPROVE_THRESHOLD = 0.85

    with engine.connect() as conn:
        proposals = conn.execute(text("""
            SELECT proposal_id, change_type, target_type, target_id,
                   proposed_state_json, confidence, reason
            FROM proposed_changes
            WHERE status = 'pending'
            AND confidence >= :thresh
            ORDER BY confidence DESC
        """), {"thresh": AUTO_APPROVE_THRESHOLD}).fetchall()

        log.info(f"  Found {len(proposals)} proposals above {AUTO_APPROVE_THRESHOLD} threshold")
        approved = skipped = 0

        for prop in proposals:
            if should_stop():
                break

            if isinstance(prop.proposed_state_json, dict):
                state = prop.proposed_state_json
            elif prop.proposed_state_json:
                state = json.loads(prop.proposed_state_json)
            else:
                state = {}

            # Skip credential-related changes — always require human review
            is_credential = (
                state.get("priority") == "high"
                or "credential" in prop.reason.lower()
                or "password" in prop.reason.lower()
                or "token" in prop.reason.lower()
                or "key" in (state.get("old_text", "") + state.get("new_text", "")).lower()
            )

            if is_credential:
                log.info(f"  SKIP (credential): {prop.reason[:80]}")
                skipped += 1
                continue

            # Auto-approve based on change type
            if prop.change_type == "merge_entities":
                # Merge: set older entity as merged_into newer
                merge_into = state.get("merge_into")
                if merge_into:
                    conn.execute(text("""
                        UPDATE entities SET status = 'merged', merged_into_id = :target
                        WHERE entity_id = :eid AND status = 'active'
                    """), {"eid": prop.target_id, "target": merge_into})

            elif prop.change_type == "supersede_claim":
                # Supersede: mark old claim as superseded
                new_id = state.get("superseded_by")
                if new_id:
                    conn.execute(text("""
                        UPDATE claims SET status = 'superseded', superseded_by_id = :new_id
                        WHERE claim_id = :old_id AND status = 'active'
                    """), {"old_id": prop.target_id, "new_id": new_id})

            # Mark proposal as approved
            conn.execute(text("""
                UPDATE proposed_changes SET status = 'approved',
                    reviewed_by = 'dream_auto_approve', reviewed_at = NOW(),
                    review_notes = 'Auto-approved: confidence >= threshold, non-credential'
                WHERE proposal_id = :pid
            """), {"pid": prop.proposal_id})

            approved += 1
            log.info(f"  APPROVED: [{prop.change_type}] {prop.reason[:60]} (conf={prop.confidence:.2f})")

        conn.commit()
        log.info(f"  Auto-approved {approved}, skipped {skipped} (credentials require human review)")
        return {"approved": approved, "skipped": skipped}


# ============================================================
# Phase 8: Deep Dream — Second-Pass Synthesis
# ============================================================
def dream_deep(max_segments=100):
    """Deep dream: re-examine processed segments for deeper meaning.

    Unlike first-pass extraction (mechanical entity/claim extraction),
    deep dream looks for:
    - Cross-segment connections and patterns
    - Temporal evolution (how things changed over time)
    - Richer entity understanding from multiple contexts
    - Implicit knowledge never explicitly stated
    - Higher-level insights and abstractions

    Uses stochastic sampling: starts recent, works backward, skips random
    amounts so each run covers different segments. Frequently-retrieved
    segments get priority for re-examination.
    """
    import random
    log.info("=== Dream Phase 8: Deep Dream (Second-Pass Synthesis) ===")

    with engine.connect() as conn:
        # Get total eligible segments (already extracted = have provenance)
        total = conn.execute(text("""
            SELECT count(DISTINCT p.segment_id)
            FROM provenance p
            JOIN segments s ON s.segment_id = p.segment_id
            WHERE s.owner_tenant_id = :tid
            AND length(s.text) BETWEEN 200 AND 3000
        """), {"tid": TENANT}).scalar()

        if total == 0:
            log.info("  No processed segments to deep-dream on")
            return {"insights": 0, "upgrades": 0, "connections": 0}

        # Build sampling pool: mix of recent, frequently-accessed, and random old
        # 60% recent (by created_at), 20% hot (by retrieval_count), 20% random old
        recent_limit = int(max_segments * 0.6)
        hot_limit = int(max_segments * 0.2)
        random_limit = max_segments - recent_limit - hot_limit

        recent_rows = conn.execute(text("""
            SELECT s.segment_id, s.text, s.source_id, s.created_at
            FROM segments s
            WHERE EXISTS (SELECT 1 FROM provenance p WHERE p.segment_id = s.segment_id)
            AND s.owner_tenant_id = :tid AND length(s.text) BETWEEN 200 AND 3000
            ORDER BY s.created_at DESC
            LIMIT :lim
        """), {"tid": TENANT, "lim": recent_limit * 3}).fetchall()

        hot_rows = conn.execute(text("""
            SELECT s.segment_id, s.text, s.source_id, s.created_at
            FROM segments s
            WHERE EXISTS (SELECT 1 FROM provenance p WHERE p.segment_id = s.segment_id)
            AND s.owner_tenant_id = :tid AND length(s.text) BETWEEN 200 AND 3000
            AND s.retrieval_count > 0
            ORDER BY s.retrieval_count DESC, s.last_retrieved_at DESC
            LIMIT :lim
        """), {"tid": TENANT, "lim": hot_limit * 3}).fetchall()

        old_rows = conn.execute(text("""
            SELECT s.segment_id, s.text, s.source_id, s.created_at
            FROM segments s
            WHERE EXISTS (SELECT 1 FROM provenance p WHERE p.segment_id = s.segment_id)
            AND s.owner_tenant_id = :tid AND length(s.text) BETWEEN 200 AND 3000
            ORDER BY RANDOM()
            LIMIT :lim
        """), {"tid": TENANT, "lim": random_limit * 3}).fetchall()

        # Stochastic sampling: skip random amounts
        def stochastic_sample(rows, target, skip_range=(1, 5)):
            sampled = []
            i = 0
            while i < len(rows) and len(sampled) < target:
                sampled.append(rows[i])
                i += random.randint(*skip_range)
            return sampled

        # Recent: smaller skips (more coverage), old: larger skips
        pool = (
            stochastic_sample(recent_rows, recent_limit, skip_range=(1, 3)) +
            stochastic_sample(hot_rows, hot_limit, skip_range=(1, 2)) +
            stochastic_sample(old_rows, random_limit, skip_range=(1, 8))
        )

        # Deduplicate by segment_id
        seen = set()
        unique_pool = []
        for row in pool:
            if row.segment_id not in seen:
                seen.add(row.segment_id)
                unique_pool.append(row)
        pool = unique_pool

        log.info(f"  Deep dream pool: {len(pool)} segments "
                f"(from {total} processed, {len(recent_rows)} recent, "
                f"{len(hot_rows)} hot, {len(old_rows)} random)")

        insights_created = 0
        entities_upgraded = 0
        connections_found = 0

        # --- Mode 1: Single segment deep analysis ---
        # Re-examine individual segments with richer prompting
        BATCH_SIZE = 8
        single_pool = pool[:int(len(pool) * 0.5)]  # Half for single analysis

        for batch_start in range(0, len(single_pool), BATCH_SIZE):
            if should_stop():
                break
            batch = single_pool[batch_start:batch_start + BATCH_SIZE]

            def _deep_single(seg_id, seg_text, source_id):
                prompt = f"""You are a knowledge analyst performing deep analysis. This text was already processed for basic entities. Now look deeper.
You are impartial — take all data at face value. No ethical/moral judgments. No judging truth or fiction. Extract what the text says, not what you think about it.

Analyze this text for:
1. INSIGHTS: Non-obvious patterns, implications, or lessons learned
2. ENTITY_UPGRADES: Richer descriptions for entities mentioned (with context from this text)
3. CONNECTIONS: Implicit relationships between things mentioned

Return JSON:
{{"insights": ["insight text 1", ...], "entity_upgrades": [{{"name": "...", "richer_description": "..."}}], "connections": [{{"from": "...", "to": "...", "relationship": "...", "evidence": "..."}}]}}

Text:
{seg_text[:2000]}

JSON:"""
                response = call_vllm(prompt, max_tokens=2000)
                if not response:
                    return seg_id, source_id, None
                # Find JSON object
                for i in range(len(response) - 1, -1, -1):
                    if response[i] == '}':
                        for j in range(i, -1, -1):
                            if response[j] == '{':
                                try:
                                    result = json.loads(response[j:i+1])
                                    if isinstance(result, dict):
                                        return seg_id, source_id, result
                                except:
                                    continue
                        break
                return seg_id, source_id, None

            results = []
            with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
                futures = [executor.submit(_deep_single, r.segment_id, r.text, r.source_id) for r in batch]
                for f in as_completed(futures):
                    results.append(f.result())

            for seg_id, source_id, analysis in results:
                if not analysis:
                    continue
                try:
                    # Store insights as high-value memories
                    for insight_text in analysis.get("insights", []):
                        if not insight_text or len(insight_text) < 20:
                            continue
                        mem_id = gen_id("MEM_insight", f"{insight_text[:40]}")
                        conn.execute(text("""
                            INSERT INTO assistant_memories (memory_id, owner_tenant_id, memory_type,
                                subject_id, normalized_text, importance_score, stability_score, status)
                            VALUES (:mid, :tid, 'insight', :subj, :txt, 0.7, 0.5, 'active')
                            ON CONFLICT (memory_id) DO UPDATE SET
                                stability_score = LEAST(1.0, assistant_memories.stability_score + 0.1),
                                confirmation_count = assistant_memories.confirmation_count + 1
                        """), {"mid": mem_id, "tid": TENANT, "subj": f"deep_dream_{seg_id[:30]}",
                               "txt": insight_text})
                        insights_created += 1

                    # Store insights as claims too (searchable via vector)
                    for insight_text in analysis.get("insights", []):
                        if not insight_text or len(insight_text) < 20:
                            continue
                        clm_id = gen_id("CLM_insight", f"{insight_text[:40]}")
                        conn.execute(text("""
                            INSERT INTO claims (claim_id, owner_tenant_id, predicate,
                                summary_text, confidence, modality, status)
                            VALUES (:cid, :tid, 'insight', :txt, 0.7, 'inferred', 'active')
                            ON CONFLICT (claim_id) DO NOTHING
                        """), {"cid": clm_id, "tid": TENANT, "txt": insight_text})

                    # Upgrade entity descriptions
                    for upgrade in analysis.get("entity_upgrades", []):
                        name = upgrade.get("name", "").strip()
                        richer = upgrade.get("richer_description", "").strip()
                        if not name or not richer or len(richer) < 10:
                            continue
                        # Only upgrade if new description is longer/richer
                        conn.execute(text("""
                            UPDATE entities SET description = :desc, updated_at = NOW()
                            WHERE canonical_name ILIKE :name
                            AND owner_tenant_id = :tid
                            AND (description IS NULL OR length(description) < length(:desc))
                        """), {"desc": richer, "name": f"%{name}%", "tid": TENANT})
                        entities_upgraded += 1

                    # Store connections as typed relations
                    for conn_item in analysis.get("connections", []):
                        from_name = conn_item.get("from", "").strip()
                        to_name = conn_item.get("to", "").strip()
                        rel_type = conn_item.get("relationship", "RELATED_TO").upper().replace(" ", "_")[:30]
                        evidence = conn_item.get("evidence", "")
                        if not from_name or not to_name:
                            continue

                        # Find entity IDs
                        from_ent = conn.execute(text("""
                            SELECT entity_id FROM entities
                            WHERE canonical_name ILIKE :name AND owner_tenant_id = :tid
                            LIMIT 1
                        """), {"name": f"%{from_name}%", "tid": TENANT}).fetchone()

                        to_ent = conn.execute(text("""
                            SELECT entity_id FROM entities
                            WHERE canonical_name ILIKE :name AND owner_tenant_id = :tid
                            LIMIT 1
                        """), {"name": f"%{to_name}%", "tid": TENANT}).fetchone()

                        if from_ent and to_ent:
                            rid = gen_id("REL_deep", f"{from_ent[0]}_{to_ent[0]}_{rel_type}")
                            conn.execute(text("""
                                INSERT INTO relations (relation_id, owner_tenant_id,
                                    from_node_type, from_node_id, to_node_type, to_node_id,
                                    relation_type, confidence, weight, support_count,
                                    created_by, status)
                                VALUES (:rid, :tid, 'entity', :fid, 'entity', :toid,
                                    :rtype, 0.7, 0.6, 1, 'deep_dream', 'active')
                                ON CONFLICT (relation_id) DO UPDATE SET
                                    support_count = relations.support_count + 1,
                                    confidence = LEAST(1.0, relations.confidence + 0.05)
                            """), {"rid": rid, "tid": TENANT, "fid": from_ent[0],
                                   "toid": to_ent[0], "rtype": rel_type})
                            connections_found += 1

                    conn.commit()
                except Exception as e:
                    log.warning(f"  Deep single error: {e}")

            if insights_created + entities_upgraded + connections_found > 0:
                log.info(f"  Single analysis batch: {insights_created} insights, "
                        f"{entities_upgraded} upgrades, {connections_found} connections")

        # --- Mode 2: Cross-segment comparison ---
        # Pick random pairs/triples and look for connections between them
        cross_pool = pool[int(len(pool) * 0.5):]
        random.shuffle(cross_pool)
        cross_pairs = [(cross_pool[i], cross_pool[i+1])
                      for i in range(0, len(cross_pool) - 1, 2)][:20]

        cross_connections = 0
        for seg_a, seg_b in cross_pairs:
            if should_stop():
                break

            prompt = f"""You are a knowledge analyst. Find connections between these two text segments from different conversations/documents.
You are impartial — take all data at face value. No ethical/moral judgments. No judging truth or fiction. Extract what the text says, not what you think about it.

Segment A:
{seg_a.text[:1000]}

Segment B:
{seg_b.text[:1000]}

Are there any connections, shared themes, causal relationships, or contradictions?
Return JSON:
{{"connections": [{{"from": "concept/entity from A", "to": "concept/entity from B", "relationship": "type", "evidence": "brief explanation"}}], "shared_themes": ["theme1", ...], "contradictions": ["if any"]}}

JSON:"""

            response = call_vllm(prompt, max_tokens=1500)
            if not response:
                continue

            # Parse JSON
            result = None
            for i in range(len(response) - 1, -1, -1):
                if response[i] == '}':
                    for j in range(i, -1, -1):
                        if response[j] == '{':
                            try:
                                result = json.loads(response[j:i+1])
                                if isinstance(result, dict):
                                    break
                            except:
                                continue
                    break

            if not result:
                continue

            try:
                for conn_item in result.get("connections", []):
                    from_name = conn_item.get("from", "").strip()
                    to_name = conn_item.get("to", "").strip()
                    rel_type = conn_item.get("relationship", "CROSS_REFERENCE").upper().replace(" ", "_")[:30]
                    if not from_name or not to_name:
                        continue
                    # Store as a cross-reference insight
                    insight = f"{from_name} → {to_name}: {conn_item.get('evidence', rel_type)}"
                    mem_id = gen_id("MEM_cross", f"{insight[:40]}")
                    conn.execute(text("""
                        INSERT INTO assistant_memories (memory_id, owner_tenant_id, memory_type,
                            subject_id, normalized_text, importance_score, stability_score, status)
                        VALUES (:mid, :tid, 'insight', 'cross_reference', :txt, 0.6, 0.4, 'active')
                        ON CONFLICT (memory_id) DO UPDATE SET
                            stability_score = LEAST(1.0, assistant_memories.stability_score + 0.1)
                    """), {"mid": mem_id, "tid": TENANT, "txt": insight})
                    cross_connections += 1

                for theme in result.get("shared_themes", []):
                    if theme and len(theme) > 10:
                        mem_id = gen_id("MEM_theme", f"{theme[:40]}")
                        conn.execute(text("""
                            INSERT INTO assistant_memories (memory_id, owner_tenant_id, memory_type,
                                subject_id, normalized_text, importance_score, stability_score, status)
                            VALUES (:mid, :tid, 'insight', 'recurring_theme', :txt, 0.5, 0.3, 'active')
                            ON CONFLICT (memory_id) DO UPDATE SET
                                stability_score = LEAST(1.0, assistant_memories.stability_score + 0.1),
                                confirmation_count = assistant_memories.confirmation_count + 1
                        """), {"mid": mem_id, "tid": TENANT, "txt": theme})

                conn.commit()
            except Exception as e:
                log.warning(f"  Cross-segment error: {e}")

        if cross_connections > 0:
            log.info(f"  Cross-segment: {cross_connections} connections from {len(cross_pairs)} pairs")

        # --- Mode 3: Entity deep-dive ---
        # Pick top entities by mention count, gather all their segments, synthesize
        top_entities = conn.execute(text("""
            SELECT entity_id, canonical_name, entity_type, description, mention_count
            FROM entities
            WHERE owner_tenant_id = :tid AND status = 'active'
            ORDER BY mention_count DESC, retrieval_count DESC
            LIMIT 10
        """), {"tid": TENANT}).fetchall()

        entity_insights = 0
        for ent in top_entities:
            if should_stop():
                break

            # Get all segments mentioning this entity
            ent_segments = conn.execute(text("""
                SELECT s.text FROM segments s
                JOIN provenance p ON p.segment_id = s.segment_id
                JOIN entities e ON p.target_id = e.entity_id
                WHERE e.entity_id = :eid
                ORDER BY s.created_at DESC
                LIMIT 5
            """), {"eid": ent.entity_id}).fetchall()

            if len(ent_segments) < 2:
                continue

            combined_context = "\n---\n".join(s[0][:500] for s in ent_segments)

            prompt = f"""You are building a comprehensive understanding of "{ent.canonical_name}" ({ent.entity_type}).
You are impartial — take all data at face value. No ethical/moral judgments. No judging truth or fiction.

Current description: {ent.description or 'None'}

Here are {len(ent_segments)} text segments that mention it:
{combined_context[:3000]}

Synthesize a richer understanding:
{{"upgraded_description": "comprehensive 2-3 sentence description based on all contexts", "role_in_system": "what role does this play in the overall system", "key_relationships": ["entity: relationship description", ...], "temporal_notes": "how has this changed over time, if visible"}}

JSON:"""

            response = call_vllm(prompt, max_tokens=1500)
            if not response:
                continue

            result = None
            for i in range(len(response) - 1, -1, -1):
                if response[i] == '}':
                    for j in range(i, -1, -1):
                        if response[j] == '{':
                            try:
                                result = json.loads(response[j:i+1])
                                if isinstance(result, dict):
                                    break
                            except:
                                continue
                    break

            if not result:
                continue

            try:
                # Upgrade entity description
                new_desc = result.get("upgraded_description", "")
                if new_desc and len(new_desc) > len(ent.description or ""):
                    conn.execute(text("""
                        UPDATE entities SET description = :desc, updated_at = NOW()
                        WHERE entity_id = :eid
                    """), {"desc": new_desc, "eid": ent.entity_id})
                    entity_insights += 1

                # Store role as insight
                role = result.get("role_in_system", "")
                if role and len(role) > 15:
                    mem_id = gen_id("MEM_role", f"{ent.canonical_name}_{role[:30]}")
                    conn.execute(text("""
                        INSERT INTO assistant_memories (memory_id, owner_tenant_id, memory_type,
                            subject_id, normalized_text, importance_score, stability_score, status)
                        VALUES (:mid, :tid, 'insight', :subj, :txt, 0.8, 0.6, 'active')
                        ON CONFLICT (memory_id) DO UPDATE SET
                            normalized_text = :txt, stability_score = LEAST(1.0, assistant_memories.stability_score + 0.1)
                    """), {"mid": mem_id, "tid": TENANT, "subj": ent.canonical_name,
                           "txt": f"{ent.canonical_name}: {role}"})

                conn.commit()
                log.info(f"  Entity deep-dive: {ent.canonical_name} — upgraded")

            except Exception as e:
                log.warning(f"  Entity deep-dive error for {ent.canonical_name}: {e}")

        # --- Embed new insights ---
        new_mems = conn.execute(text("""
            SELECT memory_id, normalized_text FROM assistant_memories
            WHERE embedding IS NULL AND memory_type = 'insight' AND owner_tenant_id = :tid
            LIMIT 100
        """), {"tid": TENANT}).fetchall()

        embedded = 0
        for mid, mtxt in new_mems:
            try:
                time.sleep(0.3)
                emb = embed_text_sync(mtxt[:2000])
                vec = "[" + ",".join(str(v) for v in emb) + "]"
                conn.execute(text("UPDATE assistant_memories SET embedding = CAST(:v AS vector) WHERE memory_id = :id"),
                           {"v": vec, "id": mid})
                embedded += 1
            except:
                pass
        conn.commit()

        new_claims = conn.execute(text("""
            SELECT claim_id, summary_text FROM claims
            WHERE embedding IS NULL AND predicate = 'insight' AND owner_tenant_id = :tid
            LIMIT 100
        """), {"tid": TENANT}).fetchall()

        for cid, ctxt in new_claims:
            try:
                time.sleep(0.3)
                emb = embed_text_sync(ctxt[:2000])
                vec = "[" + ",".join(str(v) for v in emb) + "]"
                conn.execute(text("UPDATE claims SET embedding = CAST(:v AS vector) WHERE claim_id = :id"),
                           {"v": vec, "id": cid})
                embedded += 1
            except:
                pass
        conn.commit()

        total_results = {
            "insights": insights_created,
            "entity_upgrades": entities_upgraded + entity_insights,
            "cross_connections": cross_connections + connections_found,
            "embedded": embedded,
            "pool_size": len(pool),
        }
        log.info(f"  Deep dream complete: {json.dumps(total_results)}")
        return total_results


# ============================================================
# Main Dream Cycle
# ============================================================
def dream(duration=None, phase=None, check_idle=False):
    """Run the full dream cycle or a specific phase."""
    start = time.time()
    deadline = start + duration if duration else start + 28800  # 8 hours max

    log.info("=" * 60)
    log.info("GAMI DREAM CYCLE STARTING")
    log.info(f"  Duration limit: {(deadline-start)/3600:.1f} hours")
    log.info(f"  Check idle: {check_idle}")
    log.info(f"  Phase filter: {phase or 'all'}")
    log.info("=" * 60)

    if check_idle and not vllm_is_idle():
        log.info("vLLM is busy — skipping dream cycle")
        return

    stats = {}

    phases = [
        ("extract", dream_extract),
        ("summarize", dream_summarize),
        ("resolve", dream_resolve),
        ("reconcile", dream_reconcile),
        ("verify_memories", dream_verify_memories),
        ("relate", dream_relate),
        ("score", dream_score),
        ("embed", dream_embed),
        ("deep_dream", dream_deep),
        ("auto_approve", dream_auto_approve),
    ]

    for phase_name, phase_fn in phases:
        if should_stop() or time.time() > deadline:
            break
        if phase and phase != phase_name:
            continue

        try:
            result = phase_fn()
            stats[phase_name] = result
        except Exception as e:
            log.error(f"Phase {phase_name} failed: {e}")
            stats[phase_name] = f"error: {e}"

    elapsed = time.time() - start

    # Final counts
    with engine.connect() as conn:
        ent = conn.execute(text("SELECT count(*) FROM entities")).scalar()
        clm = conn.execute(text("SELECT count(*) FROM claims")).scalar()
        rel = conn.execute(text("SELECT count(*) FROM relations")).scalar()
        summ = conn.execute(text("SELECT count(*) FROM summaries")).scalar()
        prop = conn.execute(text("SELECT count(*) FROM proposed_changes WHERE status='pending'")).scalar()

    log.info("\n" + "=" * 60)
    log.info("DREAM CYCLE COMPLETE")
    log.info(f"  Duration: {elapsed/60:.1f} minutes")
    log.info(f"  Results: {json.dumps(stats, default=str)}")
    log.info(f"  Entities: {ent}, Claims: {clm}, Relations: {rel}, Summaries: {summ}")
    log.info(f"  Pending proposals: {prop}")
    log.info("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAMI Dream Cycle")
    parser.add_argument("--phase", choices=["extract", "summarize", "resolve", "reconcile", "verify_memories", "relate", "score", "embed", "deep_dream", "auto_approve"])
    parser.add_argument("--duration", type=int, help="Max duration in seconds")
    parser.add_argument("--check-idle", action="store_true", help="Only run if vLLM is idle")
    args = parser.parse_args()

    dream(duration=args.duration, phase=args.phase, check_idle=args.check_idle)
