"""Structured extraction service — entities, claims, relations, events.

Provides both async (FastAPI) and sync (Celery) interfaces.
Uses vLLM for extraction via Jinja2 prompts.
"""
import hashlib
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

from jinja2 import Environment, FileSystemLoader
from sqlalchemy import text
from sqlalchemy.orm import Session

from api.config import settings
from api.llm.client import (
    call_vllm_sync,
    parse_json_from_llm,
)
from api.llm.embeddings import embed_text_sync

logger = logging.getLogger("gami.services.extraction")

# Jinja2 environment for prompts
PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "prompts")
jinja_env = Environment(loader=FileSystemLoader(PROMPTS_DIR))

EXTRACTOR_VERSION = "1.0.0"


def _make_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def _entity_fingerprint(name: str, entity_type: str) -> str:
    """Deterministic fingerprint for dedup."""
    return hashlib.sha256(f"{name.lower().strip()}:{entity_type.lower()}".encode()).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------

def extract_entities(
    db: Session,
    text_content: str,
    source_id: str,
    segment_id: Optional[str],
    tenant_id: str,
) -> list[dict]:
    """Extract entities from text, deduplicate, store, return entity dicts."""
    # Truncate very long text for the LLM
    input_text = text_content[:3000] if len(text_content) > 3000 else text_content

    template = jinja_env.get_template("entity_extraction.j2")
    prompt = template.render(text=input_text)

    system_prompt = (
        "You are a precise entity extraction system. "
        "Return ONLY valid JSON arrays. No explanations."
    )

    try:
        raw = call_vllm_sync(prompt, system_prompt=system_prompt, max_tokens=2048, temperature=0.1)
        entities_raw = parse_json_from_llm(raw)
    except Exception as exc:
        logger.error("Entity extraction LLM call failed: %s", exc)
        return []

    if not entities_raw or not isinstance(entities_raw, list):
        logger.warning("Entity extraction returned no valid JSON array")
        return []

    results = []
    now = datetime.now(timezone.utc)

    for ent in entities_raw:
        if not isinstance(ent, dict):
            continue
        name = (ent.get("name") or "").strip()
        etype = (ent.get("type") or "concept").strip().lower()
        if not name or len(name) < 2:
            continue

        # Validate type
        valid_types = {
            "person", "place", "organization", "technology", "service",
            "infrastructure", "concept", "credential",
        }
        if etype not in valid_types:
            etype = "concept"

        aliases = ent.get("aliases", [])
        if not isinstance(aliases, list):
            aliases = []
        description = (ent.get("description") or "")[:500]

        # Check for existing entity (same name+type in same tenant)
        existing = db.execute(
            text(
                "SELECT entity_id, mention_count FROM entities "
                "WHERE owner_tenant_id = :tid "
                "AND LOWER(canonical_name) = :name "
                "AND entity_type = :etype "
                "LIMIT 1"
            ),
            {"tid": tenant_id, "name": name.lower(), "etype": etype},
        ).fetchone()

        if existing:
            entity_id = existing[0]
            new_count = existing[1] + 1
            db.execute(
                text(
                    "UPDATE entities SET mention_count = :cnt, "
                    "last_seen_at = :now, updated_at = :now "
                    "WHERE entity_id = :eid"
                ),
                {"cnt": new_count, "now": now, "eid": entity_id},
            )
        else:
            entity_id = _make_id("ENT")
            import json as _json
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
                    "eid": entity_id,
                    "tid": tenant_id,
                    "etype": etype,
                    "name": name,
                    "aliases": _json.dumps(aliases),
                    "desc": description,
                    "now": now,
                },
            )

        # Create provenance
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
                "pid": prov_id,
                "tid": entity_id,
                "sid": source_id,
                "segid": segment_id,
                "ver": EXTRACTOR_VERSION,
                "now": now,
            },
        )

        results.append({
            "entity_id": entity_id,
            "name": name,
            "type": etype,
            "aliases": aliases,
            "description": description,
            "is_new": existing is None,
        })

    db.commit()
    logger.info("Extracted %d entities from segment %s", len(results), segment_id)
    return results


# ---------------------------------------------------------------------------
# Claim extraction
# ---------------------------------------------------------------------------

def extract_claims(
    db: Session,
    text_content: str,
    source_id: str,
    segment_id: Optional[str],
    tenant_id: str,
    entity_map: Optional[dict] = None,
) -> list[dict]:
    """Extract factual claims from text, store them, return claim dicts."""
    input_text = text_content[:3000] if len(text_content) > 3000 else text_content

    template = jinja_env.get_template("claim_extraction.j2")
    prompt = template.render(text=input_text)

    system_prompt = (
        "You are a precise claim extraction system. "
        "Return ONLY valid JSON arrays. No explanations."
    )

    try:
        raw = call_vllm_sync(prompt, system_prompt=system_prompt, max_tokens=2048, temperature=0.1)
        claims_raw = parse_json_from_llm(raw)
    except Exception as exc:
        logger.error("Claim extraction LLM call failed: %s", exc)
        return []

    if not claims_raw or not isinstance(claims_raw, list):
        logger.warning("Claim extraction returned no valid JSON array")
        return []

    results = []
    now = datetime.now(timezone.utc)
    entity_map = entity_map or {}

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

        # Try to link subject/object to known entities
        subj_entity_id = entity_map.get(subject.lower())
        obj_entity_id = entity_map.get(obj.lower()) if obj else None

        claim_id = _make_id("CLM")
        import json as _json
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
                "cid": claim_id,
                "tid": tenant_id,
                "subj_eid": subj_entity_id,
                "pred": predicate,
                "obj_eid": obj_entity_id,
                "obj_lit": _json.dumps({"value": obj}) if obj else None,
                "mod": modality,
                "conf": confidence,
                "summary": summary_text[:500],
                "now": now,
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
                "pid": prov_id,
                "tid": claim_id,
                "sid": source_id,
                "segid": segment_id,
                "ver": EXTRACTOR_VERSION,
                "conf": confidence,
                "now": now,
            },
        )

        results.append({
            "claim_id": claim_id,
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "confidence": confidence,
            "modality": modality,
        })

    db.commit()
    logger.info("Extracted %d claims from segment %s", len(results), segment_id)
    return results


# ---------------------------------------------------------------------------
# Relation extraction
# ---------------------------------------------------------------------------

def extract_relations(
    db: Session,
    entities: list[dict],
    text_content: str,
    source_id: str,
    segment_id: Optional[str],
    tenant_id: str,
) -> list[dict]:
    """Extract relations between known entities, store them."""
    if len(entities) < 2:
        return []

    input_text = text_content[:3000] if len(text_content) > 3000 else text_content

    template = jinja_env.get_template("relation_extraction.j2")
    prompt = template.render(text=input_text, entities=entities)

    system_prompt = (
        "You are a precise relation extraction system. "
        "Return ONLY valid JSON arrays. No explanations."
    )

    try:
        raw = call_vllm_sync(prompt, system_prompt=system_prompt, max_tokens=2048, temperature=0.1)
        rels_raw = parse_json_from_llm(raw)
    except Exception as exc:
        logger.error("Relation extraction LLM call failed: %s", exc)
        return []

    if not rels_raw or not isinstance(rels_raw, list):
        return []

    # Build entity name → id map
    ent_map = {}
    for e in entities:
        ent_map[e["name"].lower()] = e["entity_id"]
        for alias in e.get("aliases", []):
            if alias:
                ent_map[alias.lower()] = e["entity_id"]

    valid_rel_types = {
        "USES", "PART_OF", "LOCATED_IN", "RELATED_TO", "DERIVED_FROM",
        "INSTANCE_OF", "CAUSED_BY", "PRECEDES", "FOLLOWS", "SUPPORTS",
        "CONTRADICTS", "AUTHORED_BY", "DEFINED_BY", "MENTIONS",
        "ALIAS_OF", "ABOUT", "ELABORATES",
    }

    results = []
    now = datetime.now(timezone.utc)

    for rel in rels_raw:
        if not isinstance(rel, dict):
            continue
        from_name = (rel.get("from_entity") or "").strip().lower()
        to_name = (rel.get("to_entity") or "").strip().lower()
        rel_type = (rel.get("relation_type") or "RELATED_TO").upper()
        confidence = float(rel.get("confidence", 0.7))

        from_id = ent_map.get(from_name)
        to_id = ent_map.get(to_name)
        if not from_id or not to_id or from_id == to_id:
            continue
        if rel_type not in valid_rel_types:
            rel_type = "RELATED_TO"

        # Check for existing relation
        existing = db.execute(
            text(
                "SELECT relation_id, support_count FROM relations "
                "WHERE owner_tenant_id = :tid "
                "AND from_node_id = :fid AND to_node_id = :toid "
                "AND relation_type = :rtype "
                "LIMIT 1"
            ),
            {"tid": tenant_id, "fid": from_id, "toid": to_id, "rtype": rel_type},
        ).fetchone()

        if existing:
            rel_id = existing[0]
            db.execute(
                text(
                    "UPDATE relations SET support_count = :cnt, "
                    "last_confirmed_at = :now "
                    "WHERE relation_id = :rid"
                ),
                {"cnt": existing[1] + 1, "now": now, "rid": rel_id},
            )
        else:
            rel_id = _make_id("REL")
            db.execute(
                text(
                    "INSERT INTO relations "
                    "(relation_id, owner_tenant_id, from_node_type, from_node_id, "
                    "to_node_type, to_node_id, relation_type, confidence, weight, "
                    "support_count, first_created_at, last_confirmed_at, "
                    "created_by, extractor_version) "
                    "VALUES (:rid, :tid, 'entity', :fid, 'entity', :toid, "
                    ":rtype, :conf, 1.0, 1, :now, :now, 'extraction_pipeline', :ver)"
                ),
                {
                    "rid": rel_id,
                    "tid": tenant_id,
                    "fid": from_id,
                    "toid": to_id,
                    "rtype": rel_type,
                    "conf": confidence,
                    "now": now,
                    "ver": EXTRACTOR_VERSION,
                },
            )

        results.append({
            "relation_id": rel_id,
            "from_entity": from_name,
            "to_entity": to_name,
            "relation_type": rel_type,
            "confidence": confidence,
            "is_new": existing is None,
        })

    db.commit()
    logger.info("Extracted %d relations from segment %s", len(results), segment_id)
    return results


# ---------------------------------------------------------------------------
# Event extraction
# ---------------------------------------------------------------------------

def extract_events(
    db: Session,
    text_content: str,
    source_id: str,
    segment_id: Optional[str],
    tenant_id: str,
    entity_map: Optional[dict] = None,
) -> list[dict]:
    """Extract events from text, store them, return event dicts."""
    input_text = text_content[:3000] if len(text_content) > 3000 else text_content

    template = jinja_env.get_template("event_extraction.j2")
    prompt = template.render(text=input_text)

    system_prompt = (
        "You are a precise event extraction system. "
        "Return ONLY valid JSON arrays. No explanations."
    )

    try:
        raw = call_vllm_sync(prompt, system_prompt=system_prompt, max_tokens=2048, temperature=0.1)
        events_raw = parse_json_from_llm(raw)
    except Exception as exc:
        logger.error("Event extraction LLM call failed: %s", exc)
        return []

    if not events_raw or not isinstance(events_raw, list):
        return []

    results = []
    now = datetime.now(timezone.utc)
    entity_map = entity_map or {}

    for event in events_raw:
        if not isinstance(event, dict):
            continue
        event_type = (event.get("event_type") or "other").strip().lower()
        summary = (event.get("summary") or "").strip()
        if not summary:
            continue

        actors = event.get("actors", [])
        objects = event.get("objects", [])
        location = (event.get("location") or "").strip()
        confidence = float(event.get("confidence", 0.7))

        if not isinstance(actors, list):
            actors = []
        if not isinstance(objects, list):
            objects = []

        # Resolve location entity
        location_entity_id = entity_map.get(location.lower()) if location else None

        import json as _json
        event_id = _make_id("EVT")
        db.execute(
            text(
                "INSERT INTO events "
                "(event_id, owner_tenant_id, event_type, summary, "
                "location_entity_id, actor_entity_ids, object_entity_ids, "
                "confidence, created_at) "
                "VALUES (:eid, :tid, :etype, :summary, :loc, "
                "CAST(:actors AS jsonb), CAST(:objects AS jsonb), "
                ":conf, :now)"
            ),
            {
                "eid": event_id,
                "tid": tenant_id,
                "etype": event_type,
                "summary": summary[:1000],
                "loc": location_entity_id,
                "actors": _json.dumps(actors),
                "objects": _json.dumps(objects),
                "conf": confidence,
                "now": now,
            },
        )

        # Provenance
        prov_id = _make_id("PROV")
        db.execute(
            text(
                "INSERT INTO provenance "
                "(provenance_id, target_type, target_id, source_id, segment_id, "
                "extraction_method, extractor_version, confidence, created_at) "
                "VALUES (:pid, 'event', :tid, :sid, :segid, "
                "'llm_extraction', :ver, :conf, :now)"
            ),
            {
                "pid": prov_id,
                "tid": event_id,
                "sid": source_id,
                "segid": segment_id,
                "ver": EXTRACTOR_VERSION,
                "conf": confidence,
                "now": now,
            },
        )

        results.append({
            "event_id": event_id,
            "event_type": event_type,
            "summary": summary,
            "actors": actors,
            "objects": objects,
            "location": location,
            "confidence": confidence,
        })

    db.commit()
    logger.info("Extracted %d events from segment %s", len(results), segment_id)
    return results


# ---------------------------------------------------------------------------
# Combined extraction (all types from one segment)
# ---------------------------------------------------------------------------

def extract_all_from_segment(
    db: Session,
    segment_id: str,
    text_content: str,
    source_id: str,
    tenant_id: str,
) -> dict:
    """Run all extractors on a single segment. Returns summary of results."""

    # 1. Extract entities first (needed for relations)
    entities = extract_entities(db, text_content, source_id, segment_id, tenant_id)

    # Build entity name → id map for linking claims/events
    entity_map = {}
    for e in entities:
        entity_map[e["name"].lower()] = e["entity_id"]
        for alias in e.get("aliases", []):
            if alias:
                entity_map[alias.lower()] = e["entity_id"]

    # 2. Extract claims
    claims = extract_claims(db, text_content, source_id, segment_id, tenant_id, entity_map)

    # 3. Extract relations (needs entity list)
    relations = extract_relations(db, entities, text_content, source_id, segment_id, tenant_id)

    # 4. Extract events
    events = extract_events(db, text_content, source_id, segment_id, tenant_id, entity_map)

    return {
        "segment_id": segment_id,
        "entities": len(entities),
        "claims": len(claims),
        "relations": len(relations),
        "events": len(events),
        "entity_details": entities,
        "claim_details": claims,
        "relation_details": relations,
        "event_details": events,
    }
