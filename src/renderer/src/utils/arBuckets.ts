/** Aspect-ratio bucketing helpers — mirrors trainer/ar_buckets.py */

export const BUCKET_STEP = 64

export function normalizeResolutions(resolutions: Iterable<number>): number[] {
  const out = [
    ...new Set(
      [...resolutions]
        .map((r) => Math.round(Number(r)))
        .filter((r) => Number.isFinite(r) && r > 0)
    )
  ].sort((a, b) => a - b)
  return out.length > 0 ? out : [1024]
}

function snap(v: number, step: number, lo: number, hi: number): number {
  const s = Math.max(1, Math.round(step))
  const snapped = Math.round(v / s) * s
  return Math.max(lo, Math.min(hi, snapped))
}

export function generateBuckets(
  maxRes: number,
  minRes = 512,
  step = BUCKET_STEP
): Array<[number, number]> {
  const s = Math.max(1, Math.round(step))
  let maxR = Math.max(s, Math.round(maxRes))
  let minR = Math.max(s, Math.min(Math.round(minRes), maxR))
  minR = snap(minR, s, s, maxR)
  maxR = snap(maxR, s, minR, maxR)
  const maxArea = maxR * maxR
  const buckets: Array<[number, number]> = []
  for (let h = minR; h <= maxR; h += s) {
    for (let w = minR; w <= maxR; w += s) {
      if (w * h <= maxArea) buckets.push([w, h])
    }
  }
  if (buckets.length === 0) buckets.push([maxR, maxR])
  return buckets
}

export function pickResolutionTier(
  iw: number,
  ih: number,
  resolutions: number[]
): { tier: number; allowUpscale: boolean } {
  const R = normalizeResolutions(resolutions)
  const longSide = Math.max(Math.round(iw), Math.round(ih))
  const closest = R.reduce((best, r) => {
    const d = Math.abs(r - longSide)
    const bd = Math.abs(best - longSide)
    if (d < bd) return r
    if (d === bd && r < best) return r
    return best
  })
  if (closest <= longSide) return { tier: closest, allowUpscale: false }
  const lower = R.filter((r) => r <= longSide)
  if (lower.length > 0) return { tier: Math.max(...lower), allowUpscale: false }
  return { tier: Math.min(...R), allowUpscale: true }
}

export function closestBucket(
  iw: number,
  ih: number,
  buckets: Array<[number, number]>,
  noUpscale = true
): [number, number] {
  if (iw <= 0 || ih <= 0 || buckets.length === 0) {
    throw new Error('invalid image size or empty buckets')
  }
  const ar = Math.log(Math.max(iw, 1) / Math.max(ih, 1))

  const score = (b: [number, number]): [number, number] => {
    const [bw, bh] = b
    return [Math.abs(Math.log(Math.max(bw, 1) / Math.max(bh, 1)) - ar), -(bw * bh)]
  }

  const better = (a: [number, number], b: [number, number]): boolean => {
    if (a[0] !== b[0]) return a[0] < b[0]
    return a[1] < b[1]
  }

  let best = buckets[0]
  let bestScore = score(best)
  for (let i = 1; i < buckets.length; i++) {
    const s = score(buckets[i])
    if (better(s, bestScore)) {
      best = buckets[i]
      bestScore = s
    }
  }

  if (noUpscale && (best[0] > iw || best[1] > ih)) {
    const fitting = buckets.filter((b) => b[0] <= iw && b[1] <= ih)
    if (fitting.length > 0) {
      best = fitting[0]
      bestScore = score(best)
      for (let i = 1; i < fitting.length; i++) {
        const s = score(fitting[i])
        if (better(s, bestScore)) {
          best = fitting[i]
          bestScore = s
        }
      }
    } else {
      best = buckets.reduce((a, b) => (a[0] * a[1] <= b[0] * b[1] ? a : b))
    }
  }

  return best
}

export function assignBucket(
  iw: number,
  ih: number,
  resolutions: number[],
  step = BUCKET_STEP
): { w: number; h: number; tier: number; allowUpscale: boolean } {
  const R = normalizeResolutions(resolutions)
  const minRes = Math.min(...R)
  const { tier, allowUpscale } = pickResolutionTier(iw, ih, R)
  const buckets = generateBuckets(tier, minRes, step)
  const [w, h] = closestBucket(iw, ih, buckets, !allowUpscale)
  return { w, h, tier, allowUpscale }
}

export function coverScale(
  iw: number,
  ih: number,
  tw: number,
  th: number,
  noUpscale: boolean
): number {
  if (iw <= 0 || ih <= 0) return 1
  let scale = Math.max(tw / iw, th / ih)
  if (noUpscale) scale = Math.min(scale, 1)
  return scale
}

/** Crop rectangle in original image pixel coordinates (center cover crop). */
export function coverCropRect(
  iw: number,
  ih: number,
  tw: number,
  th: number,
  noUpscale = true
): { left: number; top: number; width: number; height: number } {
  const scale = coverScale(iw, ih, tw, th, noUpscale)
  const rw = Math.max(1, Math.round(iw * scale))
  const rh = Math.max(1, Math.round(ih * scale))
  const cropW = Math.min(tw, rw)
  const cropH = Math.min(th, rh)
  const maxL = Math.max(0, rw - cropW)
  const maxT = Math.max(0, rh - cropH)
  const leftR = Math.floor(maxL / 2)
  const topR = Math.floor(maxT / 2)

  // Map resized crop box back to original image space.
  const inv = scale > 0 ? 1 / scale : 1
  const left = leftR * inv
  const top = topR * inv
  const width = cropW * inv
  const height = cropH * inv

  return {
    left: Math.max(0, Math.min(iw, left)),
    top: Math.max(0, Math.min(ih, top)),
    width: Math.max(0, Math.min(iw - left, width)),
    height: Math.max(0, Math.min(ih - top, height))
  }
}
