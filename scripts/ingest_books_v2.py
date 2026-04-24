import os
#!/usr/bin/env python3
"""Book ingest v2 — per-page tracking, quality gate, parallel processing.

Phase 1: Digital books (fitz text only, no OCR)
Phase 2: Scanned books (tesseract, quality check, flag for Gemma3)

Per-page tracking: each page stored individually, resume-safe.
"""
import argparse, fitz, hashlib, logging, os, re, sys, time
import pytesseract
from PIL import Image
import io
from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parsers.pdf_parser import _clean_text, _extract_page_text_fitz, _extract_images_from_page
from parsers.chunker import _count_tokens

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                   handlers=[logging.FileHandler("/tmp/ingest_v2.log"), logging.StreamHandler()])
log = logging.getLogger("ingest_v2")

DB_URL = os.getenv("DATABASE_URL", "postgresql://gami:gami@localhost:5432/gami")
engine = create_engine(DB_URL, pool_size=3)
TENANT = os.getenv("GAMI_BOOK_TENANT", "books")
BOOK_DIR = os.getenv("GAMI_BOOK_DIR", "/path/to/your/books/")


def get_remaining_books():
    """Find books not yet fully ingested."""
    with engine.connect() as conn:
        existing = {r[0] for r in conn.execute(
            text("SELECT raw_file_path FROM sources WHERE owner_tenant_id = :tid AND parse_status = 'parsed'"),
            {"tid": TENANT}
        ).fetchall() if r[0]}
    
    all_pdfs = []
    for f in sorted(os.listdir(BOOK_DIR)):
        if not f.endswith('.pdf'):
            continue
        path = os.path.join(BOOK_DIR, f)
        if path in existing:
            continue
        all_pdfs.append(path)
    return all_pdfs


def classify_book(path):
    """Check if a book is digital or scanned."""
    try:
        doc = fitz.open(path)
        total_pages = doc.page_count
        text_chars = 0
        check_pages = min(5, total_pages)
        for i in range(check_pages):
            text_chars += len(doc[i].get_text("text").strip())
        doc.close()
        avg = text_chars / max(check_pages, 1)
        return "digital" if avg > 200 else "scanned", total_pages
    except:
        return "error", 0


def is_garbled(text):
    """Check if OCR text is garbled (too many non-word characters)."""
    if len(text) < 20:
        return False  # Too short to tell
    words = text.split()
    if not words:
        return True
    # Count words that are mostly alphabetic
    good_words = sum(1 for w in words if sum(c.isalpha() for c in w) > len(w) * 0.5)
    return good_words < len(words) * 0.3


def ingest_book(path, use_ocr=False):
    """Ingest a single book with per-page tracking."""
    fname = os.path.basename(path)
    source_id = f"SRC_PDF_{hashlib.md5(path.encode()).hexdigest()[:12]}"
    
    try:
        doc = fitz.open(path)
    except Exception as e:
        log.error(f"Cannot open {fname}: {e}")
        return 0, 0, []
    
    total_pages = doc.page_count
    
    # Create/update source
    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO sources (source_id, owner_tenant_id, source_type, title,
                raw_file_path, checksum, parse_status)
            VALUES (:sid, :tid, 'pdf', :title, :path, :cksum, 'parsing')
            ON CONFLICT (source_id) DO UPDATE SET parse_status = 'parsing'
        """), {
            "sid": source_id, "tid": TENANT, "title": fname.replace('.pdf', ''),
            "path": path, "cksum": f"sha256:{hashlib.sha256(open(path,'rb').read(8192)).hexdigest()}",
        })
        conn.commit()
        
        # Find already-processed pages
        done_pages = {r[0] for r in conn.execute(
            text("SELECT ordinal FROM segments WHERE source_id = :sid"),
            {"sid": source_id}
        ).fetchall()}
    
    new_segments = 0
    gemma_pages = []  # Pages that need Gemma3
    
    for pg_num in range(total_pages):
        if pg_num in done_pages:
            continue
        
        page = doc[pg_num]
        seg_id = f"SEG_{source_id}_{pg_num}"
        
        # Extract text
        page_text = _extract_page_text_fitz(page)
        
        # Extract image metadata
        images = _extract_images_from_page(page, pg_num + 1)
        if images:
            for img in images:
                page_text += f"\n\n[IMAGE: {img['width']}x{img['height']}px {img['format']} at page {img['page']}]"
        
        # OCR if needed
        if use_ocr and len(page_text.strip()) < 200:
            try:
                mat = fitz.Matrix(300/72, 300/72)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                ocr_text = pytesseract.image_to_string(img).strip()
                
                if len(ocr_text) > len(page_text) and not is_garbled(ocr_text):
                    page_text = ocr_text
                elif len(ocr_text) > len(page_text) and is_garbled(ocr_text):
                    # Garbled — flag for Gemma3
                    gemma_pages.append(pg_num)
                    page_text = f"[NEEDS_GEMMA: page {pg_num+1} — OCR produced garbled text]"
            except Exception as e:
                gemma_pages.append(pg_num)
                page_text = f"[NEEDS_GEMMA: page {pg_num+1} — OCR failed: {str(e)[:50]}]"
        
        elif len(page_text.strip()) < 50 and not use_ocr:
            # Digital book but sparse page — likely illustration
            if images:
                page_text = f"[NEEDS_GEMMA: page {pg_num+1} — illustration/diagram]"
                gemma_pages.append(pg_num)
            elif len(page_text.strip()) < 10:
                page_text = f"[BLANK PAGE: page {pg_num+1}]"
        
        # Clean text
        page_text = _clean_text(page_text) if len(page_text) > 50 else page_text
        
        # Store (strip NUL bytes — PostgreSQL rejects them in text columns)
        page_text = page_text.replace('\x00', '')
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO segments (segment_id, source_id, owner_tenant_id, text,
                    segment_type, page_start, page_end, ordinal, depth, token_count)
                VALUES (:sid, :src, :tid, :txt, 'page', :pg, :pg, :ord, 0, :tc)
                ON CONFLICT (segment_id) DO NOTHING
            """), {
                "sid": seg_id, "src": source_id, "tid": TENANT, "txt": page_text,
                "pg": pg_num + 1, "ord": pg_num, "tc": len(page_text) // 4,
            })
            conn.commit()
        new_segments += 1
    
    # Mark source as parsed
    with engine.connect() as conn:
        conn.execute(text("UPDATE sources SET parse_status = 'parsed' WHERE source_id = :sid"),
                    {"sid": source_id})
        conn.commit()
    
    doc.close()
    return new_segments, total_pages, gemma_pages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["digital", "scanned", "all"], default="all")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    
    remaining = get_remaining_books()
    log.info(f"Found {len(remaining)} books to process")
    
    # Classify
    digital = []
    scanned = []
    for path in remaining:
        btype, pages = classify_book(path)
        if btype == "digital":
            digital.append((path, pages))
        elif btype == "scanned":
            scanned.append((path, pages))
    
    log.info(f"Digital: {len(digital)}, Scanned: {len(scanned)}")
    
    total_new = 0
    all_gemma_pages = {}
    
    # Phase 1: Digital books (fast)
    if args.phase in ("digital", "all"):
        log.info(f"\n=== PHASE 1: {len(digital)} digital books ===")
        start = time.time()
        for i, (path, pages) in enumerate(digital):
            if args.limit and i >= args.limit:
                break
            t0 = time.time()
            segs, total, gemma = ingest_book(path, use_ocr=False)
            elapsed = time.time() - t0
            total_new += segs
            if gemma:
                all_gemma_pages[path] = gemma
            fname = os.path.basename(path)
            log.info(f"  [{i+1}/{len(digital)}] {fname}: {segs}/{total} pages in {elapsed:.1f}s"
                    + (f" ({len(gemma)} need Gemma3)" if gemma else ""))
        log.info(f"Phase 1 done: {total_new} segments in {time.time()-start:.0f}s")
    
    # Phase 2: Scanned books (tesseract + quality gate)
    if args.phase in ("scanned", "all"):
        log.info(f"\n=== PHASE 2: {len(scanned)} scanned books ===")
        start = time.time()
        for i, (path, pages) in enumerate(scanned):
            if args.limit and i >= args.limit:
                break
            t0 = time.time()
            segs, total, gemma = ingest_book(path, use_ocr=True)
            elapsed = time.time() - t0
            total_new += segs
            if gemma:
                all_gemma_pages[path] = gemma
            fname = os.path.basename(path)
            log.info(f"  [{i+1}/{len(scanned)}] {fname}: {segs}/{total} pages in {elapsed:.1f}s"
                    + (f" ({len(gemma)} need Gemma3)" if gemma else ""))
        log.info(f"Phase 2 done in {time.time()-start:.0f}s")
    
    # Report Gemma3 needs
    total_gemma = sum(len(v) for v in all_gemma_pages.values())
    if total_gemma:
        log.info(f"\n=== {total_gemma} pages flagged for Gemma3 across {len(all_gemma_pages)} books ===")
        # Save the list for Gemma3 processing
        import json
        with open("/tmp/gemma_pages_needed.json", "w") as f:
            json.dump({k: v for k, v in all_gemma_pages.items()}, f)
        log.info("Saved to /tmp/gemma_pages_needed.json")
    
    log.info(f"\nTOTAL: {total_new} new segments")


if __name__ == "__main__":
    main()
