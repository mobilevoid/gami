"""PDF parser for GAMI — high-quality text extraction with OCR and image handling.

Extraction priority stack:
1. PyMuPDF (fitz) direct text extraction — fast, handles digital PDFs
2. GPU OCR via doctr (torch backend) — high quality for scanned pages
3. pdfplumber fallback — slower but handles edge cases

Image handling:
- Extracts images from each page via PyMuPDF
- Inserts [IMAGE: WxH at page N] placeholders inline
- Stores image metadata in segment metadata for future vision processing

Handles: scanned PDFs, mixed digital+scanned, multi-column, tables.
"""
import base64
import io
import logging
import os
import re
from typing import Optional

from parsers.base import BaseParser, ParsedSegment, ParseResult, register_parser
from parsers.chunker import chunk_segments

logger = logging.getLogger(__name__)

# Minimum characters per page to consider text extraction successful.
# Set high enough to avoid OCR'ing pages that just have headers/footers.
# Most book pages have 1000+ chars. Only OCR if we get very little text.
MIN_CHARS_PER_PAGE = 200

# Try importing PDF libraries
try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False
    logger.warning("PyMuPDF (fitz) not installed — pip install PyMuPDF")

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

# OCR backends
HAS_DOCTR = False
HAS_DOCTR_GPU = False
try:
    from doctr.io import DocumentFile
    from doctr.models import ocr_predictor
    import torch
    HAS_DOCTR = True
    HAS_DOCTR_GPU = torch.cuda.is_available()
    logger.info(f"doctr OCR available (GPU: {HAS_DOCTR_GPU})")
except ImportError:
    pass


def _get_ocr_model():
    """Lazy-load the doctr OCR model."""
    if not HAS_DOCTR:
        return None
    try:
        model = ocr_predictor(
            det_arch="db_resnet50",
            reco_arch="crnn_vgg16_bn",
            pretrained=True,
        )
        if HAS_DOCTR_GPU:
            model = model.cuda()
        return model
    except Exception as e:
        logger.warning(f"Failed to load doctr OCR model: {e}")
        return None


_ocr_model = None


def get_ocr():
    """Get the singleton OCR model."""
    global _ocr_model
    if _ocr_model is None:
        _ocr_model = _get_ocr_model()
    return _ocr_model


def _ocr_page_doctr(page_image_bytes: bytes) -> str:
    """OCR a single page image using doctr."""
    model = get_ocr()
    if model is None:
        return ""
    try:
        doc = DocumentFile.from_images([page_image_bytes])
        result = model(doc)
        # Extract text from doctr result
        lines = []
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    words = " ".join(w.value for w in line.words)
                    lines.append(words)
                lines.append("")  # Block separator
        return "\n".join(lines).strip()
    except Exception as e:
        logger.warning(f"doctr OCR failed: {e}")
        return ""


def _extract_page_text_fitz(page) -> str:
    """Extract text from a PyMuPDF page, handling multi-column layouts."""
    # Use "dict" extraction for better layout handling
    try:
        blocks = page.get_text("dict", sort=True)["blocks"]
        text_blocks = []
        for b in blocks:
            if b["type"] == 0:  # Text block
                block_text = ""
                for line in b.get("lines", []):
                    line_text = " ".join(
                        span["text"] for span in line.get("spans", [])
                    ).strip()
                    if line_text:
                        block_text += line_text + "\n"
                if block_text.strip():
                    text_blocks.append(block_text.strip())
        return "\n\n".join(text_blocks)
    except Exception:
        # Fallback to simple extraction
        return page.get_text("text").strip()


def _extract_images_from_page(page, page_num: int) -> list[dict]:
    """Extract image metadata from a PyMuPDF page."""
    images = []
    try:
        img_list = page.get_images(full=True)
        for img_idx, img_info in enumerate(img_list):
            xref = img_info[0]
            try:
                base_image = page.parent.extract_image(xref)
                if base_image:
                    width = base_image.get("width", 0)
                    height = base_image.get("height", 0)
                    img_format = base_image.get("ext", "unknown")
                    img_size = len(base_image.get("image", b""))

                    # Only include meaningful images (skip tiny icons/bullets)
                    if width > 50 and height > 50 and img_size > 1000:
                        images.append({
                            "page": page_num,
                            "index": img_idx,
                            "width": width,
                            "height": height,
                            "format": img_format,
                            "size_bytes": img_size,
                            "xref": xref,
                        })
            except Exception:
                continue
    except Exception as e:
        logger.debug(f"Image extraction failed on page {page_num}: {e}")
    return images


def _page_needs_ocr(text: str, page) -> bool:
    """Determine if a page needs OCR (scanned/image-only)."""
    if not text or len(text.strip()) < MIN_CHARS_PER_PAGE:
        return True
    # Check if page has images that dominate the area
    try:
        page_area = page.rect.width * page.rect.height
        img_list = page.get_images(full=True)
        if img_list and len(text.strip()) < 100:
            return True
    except Exception:
        pass
    return False


def _ocr_page_fitz(page) -> str:
    """Convert a PyMuPDF page to image bytes for OCR."""
    try:
        # Render page at 300 DPI for good OCR quality
        mat = fitz.Matrix(300 / 72, 300 / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("png")
        return img_bytes
    except Exception as e:
        logger.warning(f"Page render failed: {e}")
        return None


def _extract_with_pdfplumber(file_path: str, page_num: int) -> str:
    """Fallback: extract text using pdfplumber for a specific page."""
    if not HAS_PDFPLUMBER:
        return ""
    try:
        with pdfplumber.open(file_path) as pdf:
            if page_num < len(pdf.pages):
                page = pdf.pages[page_num]
                text = page.extract_text() or ""
                # Also try table extraction
                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        table_text = "\n".join(
                            " | ".join(str(cell or "") for cell in row)
                            for row in table
                        )
                        if table_text.strip():
                            text += f"\n\n[TABLE]\n{table_text}\n[/TABLE]"
                return text.strip()
    except Exception as e:
        logger.debug(f"pdfplumber failed on page {page_num}: {e}")
    return ""


def _clean_text(text: str) -> str:
    """Clean extracted PDF text — fix spacing artifacts, OCR issues, hyphenation."""
    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append('')
            continue

        # 1. Fix "R E L A T I O N S H I P" — pure single-char-spaced lines
        tokens = stripped.split()
        single_caps = sum(1 for t in tokens if len(t) == 1 and t.isupper())
        if len(tokens) > 3 and single_caps > len(tokens) * 0.5:
            fixed = re.sub(r'(?<=[A-Z]) (?=[A-Z](?:\s|$))', '', stripped)
            fixed = re.sub(r'  +', ' ', fixed)
            cleaned_lines.append(fixed)
            continue

        # 2. Fix "C ARLO  M ATTOGNO" — first letter split from uppercase word
        fixed = stripped
        for _ in range(3):
            new_fixed = re.sub(r'\b([A-Z]) ([A-Z]{2,})\b', r'\1\2', fixed)
            if new_fixed == fixed:
                break
            fixed = new_fixed
        fixed = re.sub(r'  +', ' ', fixed)
        cleaned_lines.append(fixed)

    text = '\n'.join(cleaned_lines)

    # 3. Fix hyphenated line breaks: "impor-\ntant" → "important"
    text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)

    # 4. Remove standalone page numbers
    text = re.sub(r'^\s*\d{1,4}\s*$', '', text, flags=re.MULTILINE)

    # 5. Remove single/double char garbage lines (OCR artifacts)
    text = re.sub(r'^.{1,2}$', '', text, flags=re.MULTILINE)

    # 6. Fix Unicode/OCR ligatures and smart quotes
    for old, new in [('ﬁ', 'fi'), ('ﬂ', 'fl'), ('ﬀ', 'ff'), ('ﬃ', 'ffi'), ('ﬄ', 'ffl'),
                     ('\xad', '-'), ('\u2019', "'"), ('\u2018', "'"),
                     ('\u201c', '"'), ('\u201d', '"')]:
        text = text.replace(old, new)

    # 7. Remove decorative lines (=====, -----, .....)
    text = re.sub(r'^\s*[.\-_=*]{5,}\s*$', '', text, flags=re.MULTILINE)

    # 8. Collapse excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def _clean_header_footer(pages_text: list[str]) -> list[str]:
    """Detect and remove repeated headers/footers across pages."""
    if len(pages_text) < 5:
        return pages_text

    # Find lines that repeat on >60% of pages (likely headers/footers)
    from collections import Counter
    first_lines = Counter()
    last_lines = Counter()
    for text in pages_text:
        lines = text.strip().split("\n")
        if lines:
            first_lines[lines[0].strip()] += 1
            last_lines[lines[-1].strip()] += 1

    threshold = len(pages_text) * 0.6
    header_patterns = {line for line, count in first_lines.items()
                       if count > threshold and len(line) < 100}
    footer_patterns = {line for line, count in last_lines.items()
                       if count > threshold and len(line) < 100}

    # Also detect page numbers
    page_num_re = re.compile(r'^\s*\d{1,4}\s*$')

    cleaned = []
    for text in pages_text:
        lines = text.strip().split("\n")
        # Remove header
        while lines and (lines[0].strip() in header_patterns or page_num_re.match(lines[0])):
            lines.pop(0)
        # Remove footer
        while lines and (lines[-1].strip() in footer_patterns or page_num_re.match(lines[-1])):
            lines.pop()
        cleaned.append("\n".join(lines))

    return cleaned


@register_parser("pdf")
class PdfParser(BaseParser):
    """High-quality PDF parser with OCR fallback and image extraction."""

    def can_parse(self, file_path: str, mime_type: str = None) -> bool:
        ext = os.path.splitext(file_path)[1].lower()
        return ext in (".pdf",) or (mime_type and "pdf" in mime_type)

    def parse(self, file_path: str, metadata: dict = None) -> ParseResult:
        if not HAS_FITZ:
            raise ImportError("PyMuPDF (fitz) is required: pip install PyMuPDF")

        metadata = metadata or {}
        fname = os.path.basename(file_path)
        logger.info(f"Parsing PDF: {fname}")

        doc = fitz.open(file_path)
        total_pages = len(doc)

        # Extract metadata
        pdf_meta = doc.metadata or {}
        title = (pdf_meta.get("title") or "").strip() or os.path.splitext(fname)[0]
        author = (pdf_meta.get("author") or "").strip() or None

        pages_text = []
        all_images = []
        ocr_pages = 0

        for page_num in range(total_pages):
            page = doc[page_num]

            # Step 1: Try direct text extraction
            text = _extract_page_text_fitz(page)

            # Step 2: Check if OCR is needed
            if _page_needs_ocr(text, page):
                # Try pdfplumber first (fast, no model loading)
                plumber_text = _extract_with_pdfplumber(file_path, page_num)
                if len(plumber_text) > len(text):
                    text = plumber_text

                # If still sparse and GPU OCR available, use doctr
                if len(text.strip()) < MIN_CHARS_PER_PAGE and HAS_DOCTR and HAS_DOCTR_GPU:
                    img_bytes = _ocr_page_fitz(page)
                    if img_bytes:
                        ocr_text = _ocr_page_doctr(img_bytes)
                        if len(ocr_text) > len(text):
                            text = ocr_text
                            ocr_pages += 1

                # If still no text, flag for future OCR (don't block on CPU OCR)
                if len(text.strip()) < MIN_CHARS_PER_PAGE:
                    text = f"[OCR_NEEDED: page {page_num + 1} — scanned image, text extraction failed]"
                    ocr_pages += 1

            # Step 3: Extract images and insert placeholders
            images = _extract_images_from_page(page, page_num + 1)
            if images:
                all_images.extend(images)
                # Insert image placeholders at end of page text
                for img in images:
                    text += f"\n\n[IMAGE: {img['width']}x{img['height']}px " \
                            f"{img['format']} at page {img['page']}]"

            pages_text.append(text)

        doc.close()

        # Step 4: Clean headers/footers
        pages_text = _clean_header_footer(pages_text)

        # Step 5: Clean text quality — fix spacing, OCR artifacts, hyphenation
        pages_text = [_clean_text(page) for page in pages_text]

        # Step 6: Build segments — one per page initially, then chunk
        segments = []
        for page_num, page_text in enumerate(pages_text):
            if not page_text or len(page_text.strip()) < 20:
                continue

            seg = ParsedSegment(
                text=page_text.strip(),
                segment_type="page",
                ordinal=page_num,
                depth=0,
                title_or_heading=title,
                page_start=page_num + 1,
                page_end=page_num + 1,
                metadata={
                    "source_file": fname,
                    "page_number": page_num + 1,
                    "total_pages": total_pages,
                    "images_on_page": [img for img in all_images if img["page"] == page_num + 1],
                },
            )
            segments.append(seg)

        # Step 6: Chunk large pages into smaller segments
        segments = chunk_segments(segments, threshold=1000)

        logger.info(
            f"  Parsed {fname}: {total_pages} pages → {len(segments)} segments, "
            f"{ocr_pages} OCR'd, {len(all_images)} images"
        )

        return ParseResult(
            title=title,
            source_type="pdf",
            segments=segments,
            author=author,
            metadata={
                "file_path": file_path,
                "file_size": os.path.getsize(file_path),
                "total_pages": total_pages,
                "ocr_pages": ocr_pages,
                "total_images": len(all_images),
                "image_details": all_images[:50],  # Cap metadata size
                "pdf_metadata": pdf_meta,
                "extraction_methods": {
                    "fitz": True,
                    "doctr_ocr": HAS_DOCTR,
                    "doctr_gpu": HAS_DOCTR_GPU,
                    "pdfplumber": HAS_PDFPLUMBER,
                },
                **(metadata or {}),
            },
        )
