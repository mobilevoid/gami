#!/usr/bin/env python3
"""Test extraction on sample segments."""
import sys, json, time, os
gami_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, gami_root)
os.chdir(gami_root)

from api.services.db import get_sync_db
from api.services.extraction import extract_all_from_segment
from sqlalchemy import text

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

db_gen = get_sync_db()
db = next(db_gen)

# Clean extraction tables
db.execute(text("DELETE FROM provenance"))
db.execute(text("DELETE FROM relations"))
db.execute(text("DELETE FROM events"))
db.execute(text("DELETE FROM claims"))
db.execute(text("DELETE FROM entities"))
db.commit()
print("Cleaned extraction tables.", flush=True)

# Select 5 diverse segments (short ones for speed)
segments = db.execute(text("""
    SELECT s.segment_id, s.source_id, s.owner_tenant_id, s.text,
           s.segment_type, src.source_type, src.title
    FROM segments s
    JOIN sources src ON s.source_id = src.source_id
    WHERE LENGTH(s.text) BETWEEN 150 AND 800
    AND s.segment_type IN ('section', 'paragraph')
    AND src.source_type = 'markdown'
    ORDER BY s.token_count DESC
    LIMIT 5
""")).fetchall()

print(f"Selected {len(segments)} segments for extraction.", flush=True)

total_entities = 0
total_claims = 0
total_relations = 0
total_events = 0

for i, seg in enumerate(segments):
    seg_id, src_id, tenant_id, txt, seg_type, src_type, title = seg
    print(f"\n=== [{i+1}/{len(segments)}] {seg_id} ({len(txt)} chars, {title}) ===", flush=True)
    print(f"Preview: {txt[:100]}...", flush=True)

    t0 = time.time()
    result = extract_all_from_segment(db, seg_id, txt, src_id, tenant_id)
    elapsed = time.time() - t0

    ne, nc, nr, nev = result["entities"], result["claims"], result["relations"], result["events"]
    total_entities += ne
    total_claims += nc
    total_relations += nr
    total_events += nev

    print(f"  Done in {elapsed:.1f}s: {ne} ent, {nc} clm, {nr} rel, {nev} evt", flush=True)

    for e in result.get('entity_details', [])[:5]:
        print(f"  ENT [{e['type']}] {e['name']}", flush=True)
    for c in result.get('claim_details', [])[:5]:
        print(f"  CLM {c['subject']} | {c['predicate']} | {c['object'][:40]}", flush=True)
    for r in result.get('relation_details', [])[:3]:
        print(f"  REL {r['from_entity']} --[{r['relation_type']}]--> {r['to_entity']}", flush=True)
    for ev in result.get('event_details', [])[:3]:
        print(f"  EVT [{ev['event_type']}] {ev['summary'][:70]}", flush=True)

print(f"\n=== TOTALS ===", flush=True)
print(f"Entities: {total_entities}", flush=True)
print(f"Claims:   {total_claims}", flush=True)
print(f"Relations:{total_relations}", flush=True)
print(f"Events:   {total_events}", flush=True)

# DB verification
print(f"\n=== DB State ===", flush=True)
for tbl in ['entities', 'claims', 'relations', 'events', 'provenance']:
    cnt = db.execute(text(f"SELECT count(*) FROM {tbl}")).scalar()
    print(f"  {tbl}: {cnt}", flush=True)

# Show some sample entities
print(f"\n=== Sample Entities ===", flush=True)
rows = db.execute(text("SELECT canonical_name, entity_type, description FROM entities LIMIT 20")).fetchall()
for r in rows:
    print(f"  [{r[1]}] {r[0]}: {(r[2] or '')[:60]}", flush=True)

# Show some sample claims
print(f"\n=== Sample Claims ===", flush=True)
rows = db.execute(text("SELECT summary_text FROM claims LIMIT 15")).fetchall()
for r in rows:
    print(f"  {r[0]}", flush=True)

try: next(db_gen)
except StopIteration: pass

print("\nDONE", flush=True)
