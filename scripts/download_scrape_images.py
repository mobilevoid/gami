#!/usr/bin/env python3
"""Download images referenced in scraped HTML files.

Handles:
- Local/relative image URLs from original sites → download from site or Wayback
- Archive.org images → use Wayback CDX API with rate limiting
- Rewrites HTML src attributes to point to local files

Rate limits archive.org to ~15 req/min to avoid blocks.
"""
import argparse, hashlib, logging, os, re, sys, time
import requests
from urllib.parse import urljoin, urlparse
import threading

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                   handlers=[logging.FileHandler("/tmp/download_images.log"), logging.StreamHandler()])
log = logging.getLogger("images")

# Rate limiter for archive.org
_archive_lock = threading.Lock()
_archive_last = [0.0]
ARCHIVE_DELAY = 4.0  # seconds between archive.org requests (~15/min)

headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'}


def rate_limit_archive():
    """Enforce rate limit for archive.org requests."""
    with _archive_lock:
        now = time.time()
        wait = max(0, _archive_last[0] + ARCHIVE_DELAY - now)
        if wait > 0:
            time.sleep(wait)
        _archive_last[0] = time.time()


def download_image(url, save_path, is_archive=False):
    """Download a single image. Returns True on success."""
    if os.path.exists(save_path) and os.path.getsize(save_path) > 100:
        return True  # Already downloaded

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if is_archive:
        rate_limit_archive()

    try:
        r = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
        if r.status_code == 200 and len(r.content) > 100:
            with open(save_path, 'wb') as f:
                f.write(r.content)
            return True
    except Exception:
        pass
    return False


def get_wayback_url(original_url):
    """Get the Wayback Machine URL for an image."""
    rate_limit_archive()
    try:
        r = requests.get(
            f"https://archive.org/wayback/available?url={original_url}",
            timeout=15
        )
        if r.status_code == 200:
            snapshot = r.json().get("archived_snapshots", {}).get("closest", {})
            if snapshot and snapshot.get("available"):
                # Convert to raw image URL (add id_ to bypass wayback toolbar)
                wb_url = snapshot["url"]
                wb_url = re.sub(r'(web/\d+)/', r'\1id_/', wb_url)
                return wb_url
    except Exception:
        pass
    return None


def process_html_dir(html_dir, base_url, images_dir):
    """Download all images referenced in HTML files."""
    downloaded = skipped = failed = 0

    for fname in sorted(os.listdir(html_dir)):
        if not fname.endswith(('.html', '.htm')):
            continue

        fpath = os.path.join(html_dir, fname)
        with open(fpath, 'r', errors='replace') as f:
            html = f.read()

        modified = False
        for match in re.finditer(
            r'((?:src|SRC)\s*=\s*["\'])([^"\']+\.(?:jpg|jpeg|png|gif|bmp|svg|webp))(["\'])',
            html, re.IGNORECASE
        ):
            prefix, img_url, suffix = match.group(1), match.group(2), match.group(3)

            # Resolve to absolute URL
            if img_url.startswith('http'):
                abs_url = img_url
            else:
                abs_url = urljoin(base_url + "/" + fname, img_url)

            # Local save path
            img_hash = hashlib.md5(abs_url.encode()).hexdigest()[:10]
            ext = os.path.splitext(urlparse(abs_url).path)[1] or '.jpg'
            local_name = f"{img_hash}{ext}"
            local_path = os.path.join(images_dir, local_name)
            relative_path = os.path.relpath(local_path, html_dir)

            # Try downloading
            success = False

            # 1. Try original URL
            if download_image(abs_url, local_path):
                success = True
            else:
                # 2. Try Wayback Machine
                wb_url = get_wayback_url(abs_url)
                if wb_url and download_image(wb_url, local_path, is_archive=True):
                    success = True

            if success:
                # Rewrite HTML to point to local file
                html = html.replace(match.group(0), f'{prefix}{relative_path}{suffix}')
                modified = True
                downloaded += 1
            else:
                failed += 1

        if modified:
            with open(fpath, 'w', encoding='utf-8') as f:
                f.write(html)

        if (downloaded + failed) % 50 == 0 and downloaded + failed > 0:
            log.info(f"  {fname}: {downloaded} downloaded, {failed} failed, {skipped} skipped")

    return downloaded, failed


def main():
    parser = argparse.ArgumentParser(description="Download images from scraped HTML files")
    parser.add_argument("--html-dir", required=True, help="Directory containing HTML files")
    parser.add_argument("--images-dir", required=True, help="Directory to save images")
    parser.add_argument("--base-url", required=True, help="Base URL for resolving relative URLs")
    args = parser.parse_args()

    # Process single site from command-line arguments
    sites = [
        {
            "html_dir": args.html_dir,
            "images_dir": args.images_dir,
            "base_url": args.base_url,
        },
    ]

    total_downloaded = 0
    total_failed = 0

    for site in sites:
        if not os.path.isdir(site["html_dir"]):
            continue
        log.info(f"\n=== {site['base_url']} ===")
        os.makedirs(site["images_dir"], exist_ok=True)
        dl, fail = process_html_dir(site["html_dir"], site["base_url"], site["images_dir"])
        total_downloaded += dl
        total_failed += fail
        log.info(f"  {dl} downloaded, {fail} failed")

    log.info(f"\nTOTAL: {total_downloaded} images downloaded, {total_failed} failed")


if __name__ == "__main__":
    main()
