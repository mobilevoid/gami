import os
#!/usr/bin/env python3
"""Ingest web scraped content into GAMI.

Handles:
- Text files (already extracted from HTML by scraper)
- Raw HTML files (strips tags)
- PDFs (uses existing PDF pipeline)

Per-file tracking, NUL byte safe, quality checked.
"""
import argparse, hashlib, logging, os, re, sys, time
from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                   handlers=[logging.FileHandler("/tmp/ingest_web.log"), logging.StreamHandler()])
log = logging.getLogger("web_ingest")

DB_URL = os.getenv("DATABASE_URL", "postgresql://gami:gami@localhost:5432/gami")
engine = create_engine(DB_URL, pool_size=3)


def strip_html(html):
    """Remove HTML tags, scripts, styles, nav elements."""
    text = re.sub(r'<(script|style|nav|footer|header|noscript)[^>]*>.*?</\1>', '', html, flags=re.DOTALL|re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    text = text.replace('&nbsp;', ' ').replace('&quot;', '"').replace('&#39;', "'")
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove NUL bytes
    text = text.replace('\x00', '')
    return text


def chunk_text(text, max_chars=3000):
    """Split text into chunks at paragraph boundaries."""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) > max_chars and current:
            chunks.append(current.strip())
            current = para
        else:
            current += "\n\n" + para if current else para
    if current.strip():
        chunks.append(current.strip())
    return chunks if chunks else [text]


def ingest_file(path, tenant_id):
    """Ingest a single text or HTML file."""
    fname = os.path.basename(path)
    source_id = f"SRC_WEB_{hashlib.md5(path.encode()).hexdigest()[:12]}"

    # Check if already ingested
    with engine.connect() as conn:
        exists = conn.execute(text("SELECT 1 FROM sources WHERE source_id = :sid"), {"sid": source_id}).fetchone()
        if exists:
            return 0, "skip"

    # Read file
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
    except Exception as e:
        return 0, f"read_error: {e}"

    # Determine if it's HTML or already text
    is_html = path.endswith('.html') or path.endswith('.htm') or '<html' in content[:500].lower()

    if is_html:
        # Extract title from HTML
        title_match = re.search(r'<title[^>]*>([^<]+)', content, re.IGNORECASE)
        title = title_match.group(1).strip() if title_match else fname
        content = strip_html(content)
    else:
        # Text file — might have URL/Title/Domain header from scraper
        lines = content.split('\n', 5)
        title = fname
        for line in lines[:4]:
            if line.startswith('Title: '):
                title = line[7:].strip()
                break
        # Remove header
        if '---' in content[:500]:
            content = content.split('---\n\n', 1)[-1] if '---' in content else content

    # Quality check
    content = content.replace('\x00', '')
    if len(content.strip()) < 50:
        return 0, "too_short"

    # Chunk the content
    chunks = chunk_text(content)

    # Create source
    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO sources (source_id, owner_tenant_id, source_type, title,
                raw_file_path, checksum, parse_status)
            VALUES (:sid, :tid, :stype, :title, :path, :cksum, 'parsed')
            ON CONFLICT DO NOTHING
        """), {
            "sid": source_id, "tid": tenant_id,
            "stype": "html" if is_html else "article",
            "title": title[:200], "path": path,
            "cksum": f"sha256:{hashlib.sha256(content[:1000].encode()).hexdigest()[:16]}",
        })

        # Insert segments
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 20:
                continue
            seg_id = f"SEG_{source_id}_{i}"
            conn.execute(text("""
                INSERT INTO segments (segment_id, source_id, owner_tenant_id, text,
                    segment_type, ordinal, depth, token_count)
                VALUES (:sid, :src, :tid, :txt, :stype, :ord, 0, :tc)
                ON CONFLICT (segment_id) DO NOTHING
            """), {
                "sid": seg_id, "src": source_id, "tid": tenant_id,
                "txt": chunk, "stype": "article" if not is_html else "html_page",
                "ord": i, "tc": len(chunk) // 4,
            })
        conn.commit()

    return len(chunks), "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tenant", default="websites")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dir", action="append", dest="dirs", help="Directories to scrape")
    args = parser.parse_args()

    # Collect all files from scrape directories
    # Example directories - customize for your setup:
    dirs = args.dirs or [
        # "/path/to/scraped/articles/",
        # "/path/to/scraped/blogs/",
    ]

    files = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for root, _, fnames in os.walk(d):
            for fname in fnames:
                if fname.endswith(('.txt', '.html', '.htm')):
                    files.append(os.path.join(root, fname))

    # Also PDFs from a specific directory (customize for your setup)
    pdf_dir = os.getenv("GAMI_PDF_DIR", "")
    if pdf_dir and os.path.isdir(pdf_dir):
        for fname in os.listdir(pdf_dir):
            if fname.endswith('.pdf'):
                files.append(os.path.join(pdf_dir, fname))

    log.info(f"Found {len(files)} files to ingest into '{args.tenant}'")

    ok = skip = errors = 0
    start = time.time()

    for i, path in enumerate(sorted(files)):
        if args.limit and i >= args.limit:
            break

        segs, status = ingest_file(path, args.tenant)

        if status == "ok":
            ok += 1
        elif status == "skip":
            skip += 1
        else:
            errors += 1

        if (i + 1) % 500 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            remaining = (len(files) - i) / max(rate, 0.01) / 60
            log.info(f"  {i+1}/{len(files)} ok={ok} skip={skip} err={errors} ({rate:.0f}/s, ~{remaining:.1f}min)")

    elapsed = time.time() - start
    log.info(f"DONE: {ok} ingested, {skip} skipped, {errors} errors in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
