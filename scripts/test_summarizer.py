#!/usr/bin/env python3
"""Test summarization on sample sources."""
import sys, json, time, os
gami_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, gami_root)
os.chdir(gami_root)

from api.services.db import get_sync_db
from api.services.summarizer import summarize_source
from sqlalchemy import text

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

db_gen = get_sync_db()
db = next(db_gen)

# Clean summaries table
db.execute(text("DELETE FROM summaries"))
db.commit()
print("Cleaned summaries table.", flush=True)

# Select 3 interesting sources
sources = db.execute(text("""
    SELECT s.source_id, s.title, s.source_type, s.owner_tenant_id,
           COUNT(seg.segment_id) as seg_count
    FROM sources s
    JOIN segments seg ON s.source_id = seg.source_id
    WHERE s.source_type = 'markdown'
    GROUP BY s.source_id, s.title, s.source_type, s.owner_tenant_id
    HAVING COUNT(seg.segment_id) BETWEEN 3 AND 20
    ORDER BY COUNT(seg.segment_id) DESC
    LIMIT 3
""")).fetchall()

print(f"Selected {len(sources)} sources for summarization.", flush=True)

for i, src in enumerate(sources):
    src_id, title, src_type, tenant_id, seg_count = src
    print(f"\n=== [{i+1}/{len(sources)}] {title} ({seg_count} segments) ===", flush=True)

    t0 = time.time()
    result = summarize_source(db, src_id, tenant_id)
    elapsed = time.time() - t0

    print(f"  Done in {elapsed:.1f}s: {result}", flush=True)

# Show generated summaries
print(f"\n=== Generated Summaries ===", flush=True)
rows = db.execute(text("""
    SELECT summary_id, scope_type, scope_id, abstraction_level,
           LEFT(summary_text, 300) as preview
    FROM summaries
    ORDER BY created_at
""")).fetchall()

for r in rows:
    print(f"\n  [{r[3]}] {r[1]}/{r[2]}", flush=True)
    print(f"  {r[4]}...", flush=True)

cnt = db.execute(text("SELECT count(*) FROM summaries")).scalar()
print(f"\n=== Total summaries: {cnt} ===", flush=True)

try: next(db_gen)
except StopIteration: pass

print("\nDONE", flush=True)
