#!/usr/bin/env python3
"""Scan a folder and print AR bucket assignment counts as JSON.

Usage:
  python scan_ar_buckets.py --folder PATH --resolutions 512,768,1024
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ar_buckets import (
    BUCKET_STEP,
    assign_with_resolution_tiers,
    buckets_fingerprint,
    format_bucket_counts,
    normalize_resolutions,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def parse_resolutions(raw: str) -> list[int]:
    parts = [p.strip() for p in str(raw).replace(";", ",").split(",")]
    nums: list[int] = []
    for p in parts:
        if not p:
            continue
        try:
            nums.append(int(p))
        except ValueError:
            continue
    return normalize_resolutions(nums)


def main() -> int:
    p = argparse.ArgumentParser(description="Scan AR buckets for a dataset folder")
    p.add_argument("--folder", required=True)
    p.add_argument(
        "--resolutions",
        default="1024",
        help="Comma-separated enabled resolutions (e.g. 512,768,1024)",
    )
    args = p.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        print(json.dumps({"ok": False, "error": f"Not a folder: {folder}"}))
        return 2

    resolutions = parse_resolutions(args.resolutions)
    try:
        from PIL import Image
    except ImportError:
        print(json.dumps({"ok": False, "error": "Pillow required"}))
        return 3

    paths = sorted(
        x
        for x in folder.iterdir()
        if x.is_file() and x.suffix.lower() in IMAGE_EXTS
    )
    sizes: list[tuple[int, int]] = []
    errors: list[str] = []
    for path in paths:
        try:
            with Image.open(path) as im:
                sizes.append(im.size)
        except Exception as e:
            errors.append(f"{path.name}: {e}")

    assignments, counts, tier_counts, forced, _flags = assign_with_resolution_tiers(
        sizes, resolutions, step=BUCKET_STEP
    )
    used = {b: n for b, n in counts.items() if n > 0}
    out = {
        "ok": True,
        "folder": str(folder),
        "image_count": len(sizes),
        "used_buckets": len(used),
        "forced_upscale": forced,
        "resolutions": resolutions,
        "min_res": min(resolutions),
        "step": BUCKET_STEP,
        "fingerprint": buckets_fingerprint(resolutions, BUCKET_STEP),
        "tier_counts": {str(k): v for k, v in sorted(tier_counts.items())},
        "counts": {
            f"{w}x{h}": n
            for (w, h), n in sorted(used.items(), key=lambda kv: (-kv[1], kv[0]))
        },
        "counts_ordered": [
            {"bucket": k, "count": v} for k, v in format_bucket_counts(used)
        ],
        "errors": errors[:20],
    }
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
