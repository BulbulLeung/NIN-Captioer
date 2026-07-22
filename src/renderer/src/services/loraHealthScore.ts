import {
  aggregateCategoryDetails,
  CAPTION_CATEGORIES,
  emptyClassification,
  fitnessLevelForScore,
  type CaptionAnalysisResult,
  type CaptionCategoryId,
  type CaptionClassification,
  type FitnessLevel,
  type HealthBreakdownItem
} from './captionAnalysis'

const IDENTITY_DETAILS = new Set(['woman', 'man', 'girl', 'boy'])

export interface LoraHealthScoreResult {
  total: number
  level: FitnessLevel
  breakdown: HealthBreakdownItem[]
  strengths: string[]
  improvements: string[]
  summary: string
}

function clamp(n: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, n))
}

/** Flux.1-dev / Krea2 T5 fit band (not SDXL CLIP 77). */
const FLUX_T5_TOKEN_MIN = 15
const FLUX_T5_TOKEN_MAX = 512

function isInFluxT5TokenRange(tokens: number): boolean {
  return tokens >= FLUX_T5_TOKEN_MIN && tokens <= FLUX_T5_TOKEN_MAX
}

/**
 * Rough token estimate for Flux/Krea T5 fit checks.
 * Latin: whitespace-separated tokens; CJK runs: ceil(chars / 1.5).
 */
export function estimateClipTokens(text: string): number {
  const trimmed = text.trim()
  if (!trimmed) return 0

  let tokens = 0
  // Split into CJK runs vs other
  const parts = trimmed.split(/([\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+)/)
  for (const part of parts) {
    if (!part) continue
    if (/^[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+$/.test(part)) {
      tokens += Math.ceil(part.length / 1.5)
    } else {
      const words = part.match(/\S+/g)
      if (words) tokens += words.length
    }
  }
  return tokens
}

function countUniqueDetails(hits: CaptionClassification): number {
  let n = 0
  for (const cat of CAPTION_CATEGORIES) {
    n += new Set(hits[cat.id] ?? []).size
  }
  return n
}

function scoreDetailRichness(avg: number): number {
  if (avg >= 5) return 100
  if (avg >= 4) return 85
  if (avg >= 3) return 70
  if (avg >= 2) return 50
  if (avg >= 1) return 30
  return 0
}

/**
 * Character LoRA training health score (0–100) with transparent breakdown.
 *
 * Weights: subject 25%, diversity 20%, detail 15%, length 15%,
 * coverage 15%, problem penalty 10% (subtracted).
 */
export function calculateLoraHealthScore(input: {
  captions: string[]
  classifications: CaptionClassification[]
}): LoraHealthScoreResult {
  const { captions, classifications } = input
  const totalImages = captions.length

  const emptyBreakdown = (summary: string): LoraHealthScoreResult => ({
    total: 0,
    level: 'red',
    breakdown: [
      {
        id: 'subject',
        label: 'Subject completeness',
        weight: 0.25,
        score: 0,
        weightContribution: 0,
        notes: 'No images'
      },
      {
        id: 'diversity',
        label: 'Diversity & balance',
        weight: 0.2,
        score: 0,
        weightContribution: 0,
        notes: 'No images'
      },
      {
        id: 'detail',
        label: 'Detail richness',
        weight: 0.15,
        score: 0,
        weightContribution: 0,
        notes: 'No images'
      },
      {
        id: 'length',
        label: 'Length / token fit',
        weight: 0.15,
        score: 0,
        weightContribution: 0,
        notes: 'No images'
      },
      {
        id: 'coverage',
        label: 'Category coverage',
        weight: 0.15,
        score: 0,
        weightContribution: 0,
        notes: 'No images'
      },
      {
        id: 'penalty',
        label: 'Problem cases',
        weight: 0.1,
        score: 0,
        weightContribution: 0,
        notes: 'No images'
      }
    ],
    strengths: [],
    improvements: ['Add images and captions to score the dataset.'],
    summary
  })

  if (totalImages === 0) {
    return emptyBreakdown('No images to score.')
  }

  const nonEmptyIndexes: number[] = []
  const tokenCounts: number[] = []
  const captionKeys = new Map<string, number>()

  for (let i = 0; i < totalImages; i++) {
    const text = captions[i]?.trim() ?? ''
    if (!text) continue
    nonEmptyIndexes.push(i)
    tokenCounts.push(estimateClipTokens(text))
    const key = text.toLowerCase()
    captionKeys.set(key, (captionKeys.get(key) ?? 0) + 1)
  }

  const captionedCount = nonEmptyIndexes.length
  const emptyRatio = (totalImages - captionedCount) / totalImages

  if (captionedCount === 0) {
    const penalty = clamp(emptyRatio * 40, 0, 100)
    const total = clamp(0 - penalty * 0.1, 0, 100)
    return {
      total: Math.round(total),
      level: fitnessLevelForScore(Math.round(total)),
      breakdown: [
        {
          id: 'subject',
          label: 'Subject completeness',
          weight: 0.25,
          score: 0,
          weightContribution: 0,
          notes: 'All captions empty'
        },
        {
          id: 'diversity',
          label: 'Diversity & balance',
          weight: 0.2,
          score: 0,
          weightContribution: 0,
          notes: 'All captions empty'
        },
        {
          id: 'detail',
          label: 'Detail richness',
          weight: 0.15,
          score: 0,
          weightContribution: 0,
          notes: 'All captions empty'
        },
        {
          id: 'length',
          label: 'Length / token fit',
          weight: 0.15,
          score: 0,
          weightContribution: 0,
          notes: 'All captions empty'
        },
        {
          id: 'coverage',
          label: 'Category coverage',
          weight: 0.15,
          score: 0,
          weightContribution: 0,
          notes: 'All captions empty'
        },
        {
          id: 'penalty',
          label: 'Problem cases',
          weight: 0.1,
          score: Math.round(penalty),
          weightContribution: -Math.round(penalty * 0.1 * 10) / 10,
          notes: `${Math.round(emptyRatio * 100)}% missing captions`
        }
      ],
      strengths: [],
      improvements: ['Write captions for every image before training.'],
      summary: 'Dataset has no usable captions yet.'
    }
  }

  // --- 1. Subject completeness ---
  let subjectSum = 0
  let identityCount = 0
  let subjectAnyCount = 0
  for (const i of nonEmptyIndexes) {
    const hits = classifications[i] ?? emptyClassification()
    const subjectIds = [...new Set(hits.subject ?? [])]
    let s = 0
    if (subjectIds.length > 0) {
      s += 0.6
      subjectAnyCount += 1
    }
    if (subjectIds.some((id) => IDENTITY_DETAILS.has(id))) {
      s += 0.4
      identityCount += 1
    }
    subjectSum += s
  }
  const subjectScore = (subjectSum / captionedCount) * 100
  const subjectNotes =
    identityCount / captionedCount >= 0.8
      ? `Identity terms on ${identityCount}/${captionedCount} captions`
      : `Subject tags ${subjectAnyCount}/${captionedCount}; identity ${identityCount}/${captionedCount}`

  // --- 2. Diversity & balance ---
  const categoryDetailTotals: Record<CaptionCategoryId, Map<string, number>> = {
    subject: new Map(),
    camera: new Map(),
    clothing: new Map(),
    pose: new Map(),
    expression: new Map(),
    scene: new Map()
  }
  const categoryPresence: Record<CaptionCategoryId, number> = {
    subject: 0,
    camera: 0,
    clothing: 0,
    pose: 0,
    expression: 0,
    scene: 0
  }

  for (const i of nonEmptyIndexes) {
    const hits = classifications[i] ?? emptyClassification()
    for (const cat of CAPTION_CATEGORIES) {
      const ids = [...new Set(hits[cat.id] ?? [])]
      if (ids.length > 0) categoryPresence[cat.id] += 1
      for (const id of ids) {
        const map = categoryDetailTotals[cat.id]
        map.set(id, (map.get(id) ?? 0) + 1)
      }
    }
  }

  const diversityScores: number[] = []
  for (const cat of CAPTION_CATEGORIES) {
    const map = categoryDetailTotals[cat.id]
    const taxonomySize = Math.max(1, cat.details.length)
    const unique = map.size
    let catScore = (unique / taxonomySize) * 100
    let hitTotal = 0
    let maxCount = 0
    for (const c of map.values()) {
      hitTotal += c
      if (c > maxCount) maxCount = c
    }
    const maxShare = hitTotal > 0 ? maxCount / hitTotal : 0
    if (maxShare > 0.7) catScore *= 0.6
    diversityScores.push(catScore)
  }
  const diversityScore =
    diversityScores.reduce((a, b) => a + b, 0) / diversityScores.length
  const diversityNotes = `Avg taxonomy fill ${Math.round(diversityScore)}%; concentrated labels penalized`

  // --- 3. Detail richness ---
  let detailSum = 0
  for (const i of nonEmptyIndexes) {
    detailSum += countUniqueDetails(classifications[i] ?? emptyClassification())
  }
  const avgDetails = detailSum / captionedCount
  const detailScore = scoreDetailRichness(avgDetails)
  const detailNotes = `Avg ${avgDetails.toFixed(1)} unique details / caption`

  // --- 4. Length / token fit (Flux/Krea T5 pass rate) ---
  const outOfRange = tokenCounts.filter((t) => !isInFluxT5TokenRange(t)).length
  const inRangeRatio = (captionedCount - outOfRange) / captionedCount
  let lengthScore = 100 * inRangeRatio
  lengthScore *= 1 - emptyRatio * 0.5
  lengthScore = clamp(lengthScore, 0, 100)
  const lengthNotes = `${outOfRange}/${captionedCount} outside Flux/Krea T5 fit (ideal ${FLUX_T5_TOKEN_MIN}–${FLUX_T5_TOKEN_MAX})`

  // --- 5. Category coverage ---
  const coverageRates = CAPTION_CATEGORIES.map(
    (cat) => categoryPresence[cat.id] / captionedCount
  )
  const coverageScore = (coverageRates.reduce((a, b) => a + b, 0) / 6) * 100
  const weakCats = CAPTION_CATEGORIES.filter(
    (_, idx) => coverageRates[idx] < 0.5
  ).map((c) => c.label)
  const coverageNotes =
    weakCats.length === 0
      ? 'All six categories appear in ≥50% of captions'
      : `Weak: ${weakCats.slice(0, 3).join(', ')}${weakCats.length > 3 ? '…' : ''}`

  // --- 6. Problem penalty severity ---
  const veryShortRatio =
    tokenCounts.filter((t) => t < 5).length / captionedCount
  let duplicateExtras = 0
  for (const count of captionKeys.values()) {
    if (count > 1) duplicateExtras += count - 1
  }
  const duplicateRatio = duplicateExtras / captionedCount
  const penalty = clamp(
    emptyRatio * 40 + veryShortRatio * 30 + duplicateRatio * 30,
    0,
    100
  )
  const penaltyParts: string[] = []
  if (emptyRatio > 0) penaltyParts.push(`${Math.round(emptyRatio * 100)}% empty`)
  if (veryShortRatio > 0) {
    penaltyParts.push(`${Math.round(veryShortRatio * 100)}% very short`)
  }
  if (duplicateRatio > 0) {
    penaltyParts.push(`${Math.round(duplicateRatio * 100)}% duplicates`)
  }
  const penaltyNotes =
    penaltyParts.length > 0 ? penaltyParts.join('; ') : 'No major issues detected'

  const subjectW = subjectScore * 0.25
  const diversityW = diversityScore * 0.2
  const detailW = detailScore * 0.15
  const lengthW = lengthScore * 0.15
  const coverageW = coverageScore * 0.15
  const penaltyW = penalty * 0.1

  const total = Math.round(
    clamp(subjectW + diversityW + detailW + lengthW + coverageW - penaltyW, 0, 100)
  )

  const breakdown: HealthBreakdownItem[] = [
    {
      id: 'subject',
      label: 'Subject completeness',
      weight: 0.25,
      score: Math.round(subjectScore),
      weightContribution: Math.round(subjectW * 10) / 10,
      notes: subjectNotes
    },
    {
      id: 'diversity',
      label: 'Diversity & balance',
      weight: 0.2,
      score: Math.round(diversityScore),
      weightContribution: Math.round(diversityW * 10) / 10,
      notes: diversityNotes
    },
    {
      id: 'detail',
      label: 'Detail richness',
      weight: 0.15,
      score: Math.round(detailScore),
      weightContribution: Math.round(detailW * 10) / 10,
      notes: detailNotes
    },
    {
      id: 'length',
      label: 'Length / token fit',
      weight: 0.15,
      score: Math.round(lengthScore),
      weightContribution: Math.round(lengthW * 10) / 10,
      notes: lengthNotes
    },
    {
      id: 'coverage',
      label: 'Category coverage',
      weight: 0.15,
      score: Math.round(coverageScore),
      weightContribution: Math.round(coverageW * 10) / 10,
      notes: coverageNotes
    },
    {
      id: 'penalty',
      label: 'Problem cases',
      weight: 0.1,
      score: Math.round(penalty),
      weightContribution: -Math.round(penaltyW * 10) / 10,
      notes: penaltyNotes
    }
  ]

  const strengths: string[] = []
  const improvements: string[] = []

  for (const item of breakdown) {
    if (item.id === 'penalty') {
      if (item.score <= 10) strengths.push('Few dataset hygiene issues')
      else if (item.score >= 30) {
        improvements.push(`Fix issues: ${item.notes}`)
      }
      continue
    }
    if (item.score >= 75) strengths.push(`${item.label} looks solid (${item.score})`)
    else if (item.score < 55) {
      improvements.push(`Improve ${item.label.toLowerCase()}: ${item.notes}`)
    }
  }

  if (strengths.length === 0) {
    strengths.push('Keep iterating captions as analysis completes')
  }
  if (improvements.length === 0) {
    improvements.push('Dataset looks balanced — spot-check hard cases manually')
  }

  let summary: string
  if (total >= 70) {
    summary = 'Healthy for character LoRA training — minor polish only.'
  } else if (total >= 40) {
    summary = 'Usable but uneven — prioritize the improvements below.'
  } else {
    summary = 'Weak dataset health — fix missing/short captions and coverage first.'
  }

  return {
    total,
    level: fitnessLevelForScore(total),
    breakdown,
    strengths: strengths.slice(0, 4),
    improvements: improvements.slice(0, 4),
    summary
  }
}

/** Full analysis result: category pies + LoRA health score. */
export function buildCaptionAnalysisResult(
  captions: string[],
  classifications: CaptionClassification[]
): CaptionAnalysisResult {
  const base = aggregateCategoryDetails(captions, classifications)
  const health = calculateLoraHealthScore({ captions, classifications })
  return {
    ...base,
    fitnessScore: health.total,
    fitnessLevel: health.level,
    healthBreakdown: health.breakdown,
    strengths: health.strengths,
    improvements: health.improvements,
    summary: health.summary
  }
}
