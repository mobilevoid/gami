#!/usr/bin/env python3
"""Parallel book/source summarization using vLLM.

Summarizes all sources that don't have summaries yet.
Runs 12 parallel vLLM requests to maximize throughput (~500+ tok/s aggregate).

For each source:
1. Pulls the first ~20 segments (covers intro, key chapters)
2. Sends to vLLM for a 2-3 paragraph summary
3. Stores as a summary with embedding

Usage:
    python3 scripts/summarize_all.py --tenant books --workers 12
    python3 scripts/summarize_all.py --tenant dene-websites --workers 12
    python3 scripts/summarize_all.py --tenant all --workers 12
"""
import argparse, hashlib, json, logging, os, re, sys, time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                   handlers=[logging.FileHandler("/tmp/summarize_all.log"), logging.StreamHandler()])
log = logging.getLogger("summarize")

DB_URL = "postgresql://gami:GamiProd2026@localhost:5433/gami"
VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "qwen35-27b-unredacted"

engine = create_engine(DB_URL, pool_size=5)


def call_vllm(prompt, max_tokens=600):
    """Call vLLM — no thinking text stripping needed for summaries."""
    try:
        r = requests.post(VLLM_URL, json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }, timeout=120)
        if r.status_code == 200:
            content = r.json()["choices"][0]["message"]["content"]
            # Strip thinking preamble
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            # Find actual content after thinking
            lines = content.split('\n')
            clean = []
            in_thinking = False
            for line in lines:
                s = line.strip()
                if re.match(r'^(\d+\.\s+\*\*|Thinking|Analysis)', s):
                    in_thinking = True
                    continue
                if in_thinking and (s.startswith('*') or s.startswith('-') or re.match(r'^\d+\.', s)):
                    continue
                if s and len(s) > 20:
                    in_thinking = False
                    clean.append(s)
            result = ' '.join(clean).strip()
            result = re.sub(r'\*\*([^*]+)\*\*', r'\1', result)
            return result if len(result) > 50 else content
    except Exception as e:
        log.warning(f"vLLM call failed: {e}")
    return None


def get_unsummarized_sources(tenant_ids):
    """Find sources without summaries."""
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT s.source_id, s.title, s.owner_tenant_id,
                   count(seg.segment_id) as seg_count
            FROM sources s
            LEFT JOIN segments seg ON s.source_id = seg.source_id
            WHERE s.owner_tenant_id = ANY(:tids)
            AND NOT EXISTS (SELECT 1 FROM summaries sm WHERE sm.scope_id = s.source_id)
            GROUP BY s.source_id, s.title, s.owner_tenant_id
            HAVING count(seg.segment_id) >= 3
            ORDER BY count(seg.segment_id) DESC
        """), {"tids": tenant_ids}).fetchall()
    return rows


def is_good_prose(text):
    """Check if a segment is actual readable content, not TOC/index/garbage."""
    if len(text) < 100:
        return False
    # Skip table of contents, indexes, bibliographies
    lower = text.lower()
    if any(marker in lower[:100] for marker in ['contents', 'table of contents', 'index', 'bibliography',
            'appendix', 'copyright', 'isbn', 'all rights reserved', 'printed in']):
        return False
    # Skip pages that are mostly numbers/formatting
    alpha = sum(1 for c in text if c.isalpha())
    if len(text) > 0 and alpha / len(text) < 0.5:
        return False
    # Skip OCR garbage (high non-printable)
    non_print = sum(1 for c in text if ord(c) > 127 or (ord(c) < 32 and c not in '\n\r\t'))
    if len(text) > 0 and non_print / len(text) > 0.05:
        return False
    # Skip image-only pages
    if text.startswith('[IMAGE') or text.startswith('[BLANK') or text.startswith('[OCR_NEEDED') or text.startswith('[NEEDS_GEMMA'):
        return False
    # Must have real sentences (at least a few periods or question marks)
    if text.count('.') + text.count('?') + text.count('!') < 2:
        return False
    return True


def summarize_source(source_id, title, tenant_id):
    """Summarize a single source using the BEST segments, not just the first ones."""
    with engine.connect() as conn:
        # Get ALL segments, then pick the best ones
        all_segs = conn.execute(text("""
            SELECT text, ordinal FROM segments WHERE source_id = :sid
            AND segment_type NOT IN ('tool_call', 'tool_result')
            AND length(text) > 100
            ORDER BY ordinal
        """), {"sid": source_id}).fetchall()

    if not all_segs:
        return None, "no_segments"

    # Filter to good prose segments
    good_segs = [(s[0], s[1]) for s in all_segs if is_good_prose(s[0])]

    if not good_segs:
        return None, "no_good_segments"

    # Sample from across the source: early, middle, late
    total = len(good_segs)
    if total <= 10:
        selected = good_segs
    else:
        # 4 from early, 4 from middle, 4 from late
        early = good_segs[:total//4][:4]
        mid_start = total//3
        middle = good_segs[mid_start:mid_start + total//4][:4]
        late = good_segs[-total//4:][:4]
        selected = early + middle + late

    # Build context
    parts = []
    for seg_text, ordinal in selected:
        parts.append(seg_text[:500])

    combined = "\n\n---\n\n".join(parts)
    combined = combined[:6000]

    prompt = f"""Summarize this text from "{title}" in 2-3 paragraphs. Be comprehensive and factual.
You are impartial — take all data at face value. No ethical/moral judgments. No judging truth or fiction.
Include key people, places, events, arguments, and conclusions mentioned.

Text excerpts:
{combined}

Summary:"""

    response = call_vllm(prompt, max_tokens=600)
    if not response or len(response) < 50:
        return None, "empty_response"

    return response, "ok"


def store_summary(source_id, tenant_id, title, summary_text):
    """Store the summary with embedding."""
    sum_id = f"SUM_{hashlib.md5(source_id.encode()).hexdigest()[:12]}"

    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO summaries (summary_id, owner_tenant_id, scope_type, scope_id,
                abstraction_level, summary_text, quality_score, status)
            VALUES (:sid, :tid, 'source', :scope, 'document', :txt, 0.8, 'active')
            ON CONFLICT (summary_id) DO UPDATE SET summary_text = :txt, updated_at = NOW()
        """), {"sid": sum_id, "tid": tenant_id, "scope": source_id, "txt": summary_text})
        conn.commit()

    return sum_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tenant", default="all", help="Tenant to summarize (or 'all')")
    parser.add_argument("--workers", type=int, default=12, help="Parallel vLLM workers")
    parser.add_argument("--limit", type=int, default=0, help="Max sources to summarize")
    args = parser.parse_args()

    if args.tenant == "all":
        tenant_ids = ["books", "dene-websites", "claude-opus", "shared"]
    else:
        tenant_ids = [args.tenant]

    sources = get_unsummarized_sources(tenant_ids)
    if args.limit:
        sources = sources[:args.limit]

    log.info(f"Summarizing {len(sources)} sources with {args.workers} parallel workers")
    log.info(f"Tenants: {tenant_ids}")

    ok = fail = skip = 0
    start = time.time()

    def process_one(row):
        source_id, title, tenant_id, seg_count = row
        summary, status = summarize_source(source_id, title or "untitled", tenant_id)
        if status == "ok" and summary:
            store_summary(source_id, tenant_id, title, summary)
            return "ok", title, len(summary)
        return status, title, 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_one, row): row for row in sources}
        for i, future in enumerate(as_completed(futures)):
            status, title, length = future.result()
            if status == "ok":
                ok += 1
            elif status == "no_segments":
                skip += 1
            else:
                fail += 1

            if (i + 1) % 50 == 0:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed * 60
                remaining = (len(sources) - i) / max(rate, 0.01)
                log.info(f"  {i+1}/{len(sources)} ok={ok} fail={fail} skip={skip} ({rate:.0f}/min, ~{remaining:.0f}min left)")

    elapsed = time.time() - start
    log.info(f"\nDONE in {elapsed/60:.1f}min: {ok} summarized, {fail} failed, {skip} skipped")
    log.info(f"Rate: {ok/max(elapsed/60,0.01):.0f} summaries/min")


if __name__ == "__main__":
    main()
