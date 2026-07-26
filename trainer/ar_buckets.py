"""Aspect-ratio bucketing helpers for Captioer Krea2 LoRA training.

Always-on AR buckets under enabled resolution tiers (e.g. 512/768/1024).
Long side picks the closest tier without unnecessary upscale; AR picks WxH.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

BUCKET_STEP = 64


def _snap(v: int, step: int, lo: int, hi: int) -> int:
    step = max(1, int(step))
    v = int(round(v / step) * step)
    return max(lo, min(hi, v))


def normalize_resolutions(resolutions: Iterable[int]) -> list[int]:
    out = sorted({int(r) for r in resolutions if int(r) > 0})
    return out or [1024]


def generate_buckets(
    max_res: int,
    min_res: int = 512,
    step: int = BUCKET_STEP,
) -> list[tuple[int, int]]:
    """All (w,h) with sides in [min_res, max_res], multiples of step, area <= max_res^2."""
    step = max(1, int(step))
    max_res = max(step, int(max_res))
    min_res = max(step, min(int(min_res), max_res))
    min_res = _snap(min_res, step, step, max_res)
    max_res = _snap(max_res, step, min_res, max_res)
    max_area = max_res * max_res
    buckets: list[tuple[int, int]] = []
    for h in range(min_res, max_res + 1, step):
        for w in range(min_res, max_res + 1, step):
            if w * h <= max_area:
                buckets.append((w, h))
    if not buckets:
        buckets = [(max_res, max_res)]
    return buckets


def pick_resolution_tier(
    iw: int, ih: int, resolutions: list[int]
) -> tuple[int, bool]:
    """
    Pick enabled resolution tier for image (iw, ih).

    Returns (tier, allow_upscale):
    - Closest enabled res to long side
    - If that tier is larger than long side, use largest enabled <= long
    - If none fit, use min(enabled) and allow_upscale=True
    """
    R = normalize_resolutions(resolutions)
    long_side = max(int(iw), int(ih))
    closest = min(R, key=lambda r: (abs(r - long_side), r))
    if closest <= long_side:
        return closest, False
    lower = [r for r in R if r <= long_side]
    if lower:
        return max(lower), False
    return min(R), True


def closest_bucket(
    iw: int,
    ih: int,
    buckets: list[tuple[int, int]],
    *,
    no_upscale: bool = True,
) -> tuple[tuple[int, int], bool]:
    """
    Pick bucket for image size (iw, ih).

    Returns ((w, h), upscale_avoided_or_forced).
    """
    if iw <= 0 or ih <= 0 or not buckets:
        raise ValueError("invalid image size or empty buckets")
    ar = math.log(max(iw, 1) / max(ih, 1))

    def score(b: tuple[int, int]) -> tuple[float, int]:
        bw, bh = b
        return (abs(math.log(max(bw, 1) / max(bh, 1)) - ar), -(bw * bh))

    best = min(buckets, key=score)
    avoided = False

    if no_upscale and (best[0] > iw or best[1] > ih):
        fitting = [b for b in buckets if b[0] <= iw and b[1] <= ih]
        if fitting:
            best = min(fitting, key=score)
            avoided = True
        else:
            best = min(buckets, key=lambda b: b[0] * b[1])
            avoided = True

    return best, avoided


def assign_with_resolution_tiers(
    sizes: Iterable[tuple[int, int]],
    resolutions: list[int],
    *,
    step: int = BUCKET_STEP,
) -> tuple[
    list[tuple[int, int]],
    dict[tuple[int, int], int],
    dict[int, int],
    int,
    list[bool],
]:
    """
    Assign each (iw,ih) using resolution tiers then AR buckets.

    Returns:
      assignments, bucket_counts, tier_counts, forced_upscale_count, allow_upscale_flags
    """
    R = normalize_resolutions(resolutions)
    min_res = min(R)
    # Cache buckets per tier
    buckets_by_tier: dict[int, list[tuple[int, int]]] = {
        t: generate_buckets(t, min_res, step) for t in R
    }

    assignments: list[tuple[int, int]] = []
    counts: dict[tuple[int, int], int] = {}
    tier_counts: dict[int, int] = {}
    allow_flags: list[bool] = []
    forced_n = 0

    for iw, ih in sizes:
        tier, allow_up = pick_resolution_tier(iw, ih, R)
        buckets = buckets_by_tier[tier]
        b, _ = closest_bucket(iw, ih, buckets, no_upscale=not allow_up)
        if allow_up:
            forced_n += 1
        assignments.append(b)
        counts[b] = counts.get(b, 0) + 1
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        allow_flags.append(allow_up)

    return assignments, counts, tier_counts, forced_n, allow_flags


def assign_images_to_buckets(
    sizes: Iterable[tuple[int, int]],
    buckets: list[tuple[int, int]],
    *,
    no_upscale: bool = True,
) -> tuple[list[tuple[int, int]], dict[tuple[int, int], int], int]:
    """Legacy single-pool assign (kept for tests). Prefer assign_with_resolution_tiers."""
    assignments: list[tuple[int, int]] = []
    counts: dict[tuple[int, int], int] = {}
    avoided_n = 0
    for iw, ih in sizes:
        b, avoided = closest_bucket(iw, ih, buckets, no_upscale=no_upscale)
        assignments.append(b)
        counts[b] = counts.get(b, 0) + 1
        if avoided:
            avoided_n += 1
    return assignments, counts, avoided_n


def cover_scale(iw: int, ih: int, tw: int, th: int, *, no_upscale: bool) -> float:
    """Scale factor so resized image covers (tw, th)."""
    if iw <= 0 or ih <= 0:
        return 1.0
    scale = max(tw / iw, th / ih)
    if no_upscale:
        scale = min(scale, 1.0)
    return scale


@dataclass
class CropBox:
    left: int
    top: int
    right: int
    bottom: int


def compute_cover_crop(
    iw: int,
    ih: int,
    tw: int,
    th: int,
    *,
    no_upscale: bool = True,
    random_crop: bool = False,
    rng=None,
) -> tuple[tuple[int, int], CropBox]:
    """Return (resized_w, resized_h) and crop box in resized coordinates."""
    import random as _random

    scale = cover_scale(iw, ih, tw, th, no_upscale=no_upscale)
    rw = max(1, int(round(iw * scale)))
    rh = max(1, int(round(ih * scale)))
    crop_w = min(tw, rw)
    crop_h = min(th, rh)
    max_l = max(0, rw - crop_w)
    max_t = max(0, rh - crop_h)
    if random_crop and max_l + max_t > 0:
        r = rng if rng is not None else _random
        left = int(r.randint(0, max_l)) if max_l > 0 else 0
        top = int(r.randint(0, max_t)) if max_t > 0 else 0
    else:
        left = max_l // 2
        top = max_t // 2
    return (rw, rh), CropBox(left, top, left + crop_w, top + crop_h)


def prepare_image_for_bucket(
    img,
    tw: int,
    th: int,
    *,
    no_upscale: bool = True,
    random_crop: bool = False,
    rng=None,
):
    """PIL Image -> RGB PIL of exact size (tw, th) via cover scale + crop (+ pad)."""
    from PIL import Image

    if img.mode != "RGB":
        img = img.convert("RGB")
    iw, ih = img.size
    (rw, rh), box = compute_cover_crop(
        iw, ih, tw, th, no_upscale=no_upscale, random_crop=random_crop, rng=rng
    )
    resized = img.resize((rw, rh), Image.Resampling.BILINEAR)
    cropped = resized.crop((box.left, box.top, box.right, box.bottom))
    cw, ch = cropped.size
    if cw == tw and ch == th:
        return cropped
    canvas = Image.new("RGB", (tw, th))
    ox = (tw - cw) // 2
    oy = (th - ch) // 2
    canvas.paste(cropped, (ox, oy))
    return canvas


def format_bucket_counts(counts: dict[tuple[int, int], int]) -> list[tuple[str, int]]:
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
    return [(f"{w}x{h}", n) for (w, h), n in items]


def buckets_fingerprint(resolutions: list[int], step: int = BUCKET_STEP) -> str:
    R = normalize_resolutions(resolutions)
    joined = "-".join(str(r) for r in R)
    return f"ar_v2_res{joined}_step{int(step)}"
