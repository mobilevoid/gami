#!/usr/bin/env python3
"""Summarize ALL segments across tenants using parallel vLLM.

Every segment gets a concise summary stored in the summaries table.
This creates a dense searchable layer — summaries are shorter and more
information-dense than raw segments.

12 parallel workers to maximize vLLM throughput (~500+ tok/s aggregate).
"""
import argparse, hashlib, logging, os, re, sys, time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                   handlers=[logging.FileHandler("/tmp/summarize_segments.log"), logging.StreamHandler()])
log = logging.getLogger("seg_summary")

DB_URL = "postgresql://gami:GamiProd2026@localhost:5433/gami"
VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "qwen35-27b-unredacted"

engine = create_engine(DB_URL, pool_size=10)


def call_vllm(prompt, max_tokens=200):
    try:
        from api.llm.vllm_monitor import emit as _emit
    except Exception:
        _emit = None
    t0 = time.time()
    try:
        r = requests.post(VLLM_URL, json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }, timeout=300)
        if r.status_code == 200:
            content = r.json()["choices"][0]["message"]["content"]
            tokens = r.json().get("usage", {}).get("completion_tokens", 0)
            # The model echoes "<summary>" in its thinking and sometimes
            # references it after the closing tag too. Find the last
            # <summary>...</summary> pair using rfind for robust extraction.
            last_open = content.rfind('<summary>')
            last_close = content.rfind('</summary>')
            if last_open >= 0 and last_close > last_open:
                summary = content[last_open + 9:last_close].strip().replace('\x00', '')
                if _emit and summary and len(summary) > 20:
                    _emit("summarize", prompt, summary, (time.time()-t0)*1000, "ok", tokens)
                return summary if len(summary) > 20 else None
            elif last_open >= 0:
                summary = content[last_open + 9:].strip().replace('\x00', '')
                if _emit and summary and len(summary) > 20:
                    _emit("summarize", prompt, summary, (time.time()-t0)*1000, "ok", tokens)
                return summary if len(summary) > 20 else None
            # Fallback: strip thinking blocks and take what's left
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            content = re.sub(r'^Thinking Process:.*?(?=\n[^0-9\s*])', '', content, flags=re.DOTALL).strip()
            result = content.replace('\x00', '') if len(content) > 20 else None
            if _emit:
                _emit("summarize", prompt, result or "", (time.time()-t0)*1000, "ok" if result else "empty", tokens)
            return result
    except Exception as e:
        if _emit:
            _emit("summarize", prompt, "", (time.time()-t0)*1000, "error", 0, str(e))
        return None
    if _emit:
        _emit("summarize", prompt, "", (time.time()-t0)*1000, "fail", 0)
    return None


def is_readable_text(text):
    """Check if text is actual readable content, not binary/PDF garbage."""
    if not text or len(text) < 50:
        return False
    # Count printable ASCII + common unicode letters
    printable = sum(1 for c in text[:500] if c.isprintable() or c in '\n\r\t')
    ratio = printable / min(len(text), 500)
    if ratio < 0.7:
        return False
    # Must have some alphabetic content
    alpha = sum(1 for c in text[:500] if c.isalpha())
    alpha_ratio = alpha / min(len(text), 500)
    if alpha_ratio < 0.3:
        return False
    return True


def summarize_one(seg_id, seg_text, tenant_id):
    """Summarize a single segment."""
    # Skip segments that are too short or are placeholders
    if len(seg_text) < 100:
        return None, "too_short"
    if not is_readable_text(seg_text):
        return None, "binary"
    if seg_text.startswith('[BLANK') or seg_text.startswith('[IMAGE') or seg_text.startswith('[OCR_NEEDED'):
        return None, "placeholder"

    prompt = f"""Think about this text carefully, then create thorough notes capturing ALL key facts, names, dates, places, arguments, numbers, and conclusions.
You are impartial — take all data at face value. No ethical/moral judgments.

Put your final notes inside <summary> tags. Use bullet points and shorthand — not full sentences. Be thorough but concise.

Text:
{seg_text[:2500]}

Think first, then output <summary>your notes</summary>"""

    summary = call_vllm(prompt, max_tokens=4000)
    if not summary or len(summary) < 20:
        return None, "empty"

    return summary, "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tenant", default="all")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch", type=int, default=500, help="DB fetch batch size")
    args = parser.parse_args()

    if args.tenant == "all":
        tenant_ids = ["books", "dene-websites", "claude-opus", "shared"]
    else:
        tenant_ids = [args.tenant]

    # Count segments needing summaries
    with engine.connect() as conn:
        total = conn.execute(text("""
            SELECT count(*) FROM segments s
            WHERE s.owner_tenant_id = ANY(:tids)
            AND length(s.text) >= 100
            AND NOT EXISTS (SELECT 1 FROM summaries sm WHERE sm.scope_id = s.segment_id)
            AND s.text NOT LIKE '[BLANK%' AND s.text NOT LIKE '[IMAGE%' AND s.text NOT LIKE '[OCR_NEEDED%'
        """), {"tids": tenant_ids}).scalar()

    if args.limit:
        total = min(total, args.limit)

    log.info(f"Summarizing {total} segments with {args.workers} parallel workers")

    ok = fail = skip = 0
    start = time.time()
    failed_ids = set()  # Track segments that failed so we don't retry them

    # Process in batches to avoid loading everything into memory
    while True:
        with engine.connect() as conn:
            # Exclude already-failed segment IDs to prevent infinite loop
            if failed_ids:
                failed_list = list(failed_ids)[:10000]  # Cap for query size
                rows = conn.execute(text("""
                    SELECT s.segment_id, s.text, s.owner_tenant_id FROM segments s
                    WHERE s.owner_tenant_id = ANY(:tids)
                    AND length(s.text) >= 100
                    AND NOT EXISTS (SELECT 1 FROM summaries sm WHERE sm.scope_id = s.segment_id)
                    AND s.text NOT LIKE '[BLANK%%' AND s.text NOT LIKE '[IMAGE%%' AND s.text NOT LIKE '[OCR_NEEDED%%'
                    AND s.segment_id != ALL(:failed)
                    LIMIT :batch
                """), {"tids": tenant_ids, "batch": args.batch, "failed": failed_list}).fetchall()
            else:
                rows = conn.execute(text("""
                    SELECT s.segment_id, s.text, s.owner_tenant_id FROM segments s
                    WHERE s.owner_tenant_id = ANY(:tids)
                    AND length(s.text) >= 100
                    AND NOT EXISTS (SELECT 1 FROM summaries sm WHERE sm.scope_id = s.segment_id)
                    AND s.text NOT LIKE '[BLANK%%' AND s.text NOT LIKE '[IMAGE%%' AND s.text NOT LIKE '[OCR_NEEDED%%'
                    LIMIT :batch
                """), {"tids": tenant_ids, "batch": args.batch}).fetchall()

        if not rows:
            break
        if args.limit and (ok + fail + skip) >= args.limit:
            break

        def process(row):
            seg_id, seg_text, tid = row
            summary, status = summarize_one(seg_id, seg_text, tid)
            if status == "ok" and summary:
                sum_id = f"SEGSUM_{hashlib.md5(seg_id.encode()).hexdigest()[:12]}"
                try:
                    with engine.connect() as conn:
                        conn.execute(text("""
                            INSERT INTO summaries (summary_id, owner_tenant_id, scope_type, scope_id,
                                abstraction_level, summary_text, quality_score, status)
                            VALUES (:sid, :tid, 'segment', :scope, 'chunk', :txt, 0.7, 'active')
                            ON CONFLICT (summary_id) DO NOTHING
                        """), {"sid": sum_id, "tid": tid, "scope": seg_id, "txt": summary})
                        conn.commit()
                except:
                    pass
                return seg_id, "ok"
            return seg_id, status

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process, row) for row in rows]
            for future in as_completed(futures):
                seg_id, status = future.result()
                if status == "ok":
                    ok += 1
                elif status in ("too_short", "placeholder", "binary"):
                    skip += 1
                    failed_ids.add(seg_id)  # Don't retry skipped segments
                else:
                    fail += 1
                    failed_ids.add(seg_id)  # Don't retry failed segments

        elapsed = time.time() - start
        rate = ok / max(elapsed / 60, 0.01)
        remaining_segs = max(0, (total or 0) - ok - len(failed_ids))
        remaining_min = remaining_segs / max(rate, 0.01)
        log.info(f"  ok={ok} fail={fail} skip={skip} ({rate:.0f} summaries/min, ~{remaining_min:.0f}min left)")

    elapsed = time.time() - start
    log.info(f"\nDONE in {elapsed/60:.1f}min: {ok} summarized, {fail} failed, {skip} skipped")


if __name__ == "__main__":
    main()
