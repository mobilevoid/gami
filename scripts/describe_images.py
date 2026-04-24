#!/usr/bin/env python3
"""Describe images and blank pages using a vision model.

Replaces [OCR_NEEDED] placeholders with actual descriptions of what's on the page.
Uses Ollama vision model (minicpm-v) to describe illustrations, diagrams, photos,
and identify truly blank pages.

Run: python3 scripts/describe_images.py --tenant books --limit 100
"""
import argparse
import base64
import logging
import os
import sys
import time

import fitz
import requests
from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                   handlers=[logging.FileHandler("/tmp/describe_images.log"), logging.StreamHandler()])
log = logging.getLogger("describe")

DB_URL = os.getenv("DATABASE_URL", "postgresql://gami:gami@localhost:5432/gami")
OLLAMA_URL = "http://localhost:11434"
VISION_MODEL = "gemma3:27b-it-q8_0"

engine = create_engine(DB_URL)


def describe_page_image(file_path: str, page_num: int) -> str:
    """Render a PDF page and describe it with the vision model."""
    try:
        doc = fitz.open(file_path)
        if page_num >= len(doc):
            doc.close()
            return "[BLANK PAGE]"

        page = doc[page_num]
        # Render at 150 DPI (good enough for description, saves memory)
        mat = fitz.Matrix(150 / 72, 150 / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("png")
        doc.close()

        # Encode as base64
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        # Call vision model
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": VISION_MODEL,
                "prompt": (
                    "Describe this book page in 1-2 sentences. "
                    "If it's a blank page, say 'Blank page'. "
                    "If it has illustrations, diagrams, maps, or photos, describe what they show. "
                    "If it has text that's hard to read, summarize what you can make out. "
                    "Be specific and factual."
                ),
                "images": [img_b64],
                "stream": False,
                "options": {"num_predict": 150},
            },
            timeout=60,
        )

        if response.status_code == 200:
            desc = response.json().get("response", "").strip()
            return desc if desc else "[BLANK PAGE]"
        else:
            return f"[VISION ERROR: {response.status_code}]"

    except Exception as e:
        return f"[VISION ERROR: {e}]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tenant", default="books")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Check vision model is available
    try:
        r = requests.post(f"{OLLAMA_URL}/api/generate",
                         json={"model": VISION_MODEL, "prompt": "test", "stream": False,
                               "options": {"num_predict": 5}}, timeout=30)
        if r.status_code != 200:
            log.error(f"Vision model not available: {r.status_code}")
            log.error(f"Pull it with: ollama pull {VISION_MODEL}")
            return
    except Exception as e:
        log.error(f"Ollama not responding: {e}")
        return

    log.info(f"Vision model {VISION_MODEL} ready")

    with engine.connect() as conn:
        # Get OCR_NEEDED segments with their source file paths
        rows = conn.execute(text("""
            SELECT seg.segment_id, seg.page_start, seg.text,
                   src.raw_file_path, src.title
            FROM segments seg
            JOIN sources src ON src.source_id = seg.source_id
            WHERE seg.owner_tenant_id = :tid
            AND seg.text LIKE '%[OCR_NEEDED%'
            ORDER BY src.title, seg.page_start
            LIMIT :lim
        """), {"tid": args.tenant, "lim": args.limit}).fetchall()

        log.info(f"Found {len(rows)} OCR_NEEDED pages to describe")

        described = 0
        blank = 0
        errors = 0
        start = time.time()

        for row in rows:
            seg_id = row.segment_id
            page_num = (row.page_start or 1) - 1  # 0-indexed
            file_path = row.raw_file_path
            title = row.title or "unknown"

            if not file_path or not os.path.exists(file_path):
                errors += 1
                continue

            if args.dry_run:
                log.info(f"  Would describe: {title} p{page_num + 1}")
                described += 1
                continue

            desc = describe_page_image(file_path, page_num)

            if "BLANK" in desc.upper() or len(desc) < 20:
                new_text = f"[BLANK PAGE: page {page_num + 1}]"
                blank += 1
            else:
                # Keep image metadata from original text
                image_tags = [line for line in row.text.split("\n") if "[IMAGE:" in line]
                image_section = "\n".join(image_tags) if image_tags else ""
                new_text = f"[PAGE DESCRIPTION: page {page_num + 1}] {desc}"
                if image_section:
                    new_text += f"\n{image_section}"

            conn.execute(text(
                "UPDATE segments SET text = :txt WHERE segment_id = :sid"
            ), {"txt": new_text, "sid": seg_id})
            described += 1

            if described % 10 == 0:
                conn.commit()
                elapsed = time.time() - start
                rate = described / elapsed
                remaining = (len(rows) - described) / max(rate, 0.01) / 60
                log.info(f"  {described}/{len(rows)} ({rate:.1f}/s, ~{remaining:.1f}min) "
                        f"blank={blank}, errors={errors}")

        conn.commit()
        elapsed = time.time() - start
        log.info(f"DONE: {described} described, {blank} blank, {errors} errors in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
