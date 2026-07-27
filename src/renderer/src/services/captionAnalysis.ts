export type CaptionCategoryId =
  | 'subject'
  | 'camera'
  | 'clothing'
  | 'pose'
  | 'expression'
  | 'scene'

export interface DetailDef {
  id: string
  label: string
}

export interface CaptionCategoryDef {
  id: CaptionCategoryId
  label: string
  details: DetailDef[]
}

export interface DetailCount {
  detailId: string
  label: string
  count: number
}

/** Per-caption AI classification: category id → list of detail ids. */
export type CaptionClassification = Record<CaptionCategoryId, string[]>

export const DETAIL_COLORS = [
  '#5a9dff',
  '#7fd99a',
  '#e6c07b',
  '#c678dd',
  '#f07178',
  '#56b6c2',
  '#d19a66',
  '#61afef',
  '#98c379',
  '#e5c07b',
  '#be5046',
  '#abb2bf'
]

export const CAPTION_CATEGORIES: CaptionCategoryDef[] = [
  {
    id: 'subject',
    label: 'Subject',
    details: [
      { id: 'woman', label: 'woman' },
      { id: 'man', label: 'man' },
      { id: 'girl', label: 'girl' },
      { id: 'boy', label: 'boy' },
      { id: 'flat_chest', label: 'flat chest' },
      { id: 'small_breasts', label: 'small breasts' },
      { id: 'medium_breasts', label: 'medium breasts' },
      { id: 'large_breasts', label: 'large breasts' },
      { id: 'huge_breasts', label: 'huge breasts' }
    ]
  },
  {
    id: 'camera',
    label: 'Camera Angle',
    details: [
      { id: 'front_view', label: 'front view' },
      { id: 'side_view', label: 'side view' },
      { id: 'back_view', label: 'back view' },
      { id: 'from_above', label: 'from above' },
      { id: 'from_below', label: 'from below' },
      { id: 'close_up', label: 'close-up' },
      { id: 'looking_at_viewer', label: 'looking at viewer' },
      { id: 'cowboy_shot', label: 'cowboy shot' },
      { id: 'full_body', label: 'full body' },
      { id: 'upper_body', label: 'upper body' }
    ]
  },
  {
    id: 'clothing',
    label: 'Clothing / Accessories',
    details: [
      { id: 'dress', label: 'dress' },
      { id: 'shirt', label: 'shirt' },
      { id: 'skirt', label: 'skirt' },
      { id: 'lingerie', label: 'lingerie' },
      { id: 'jewelry', label: 'jewelry' },
      { id: 'glasses', label: 'glasses' },
      { id: 'shoes', label: 'shoes' },
      { id: 'hat', label: 'hat' },
      { id: 'stockings', label: 'stockings' }
    ]
  },
  {
    id: 'pose',
    label: 'Pose / Action',
    details: [
      { id: 'standing', label: 'standing' },
      { id: 'sitting', label: 'sitting' },
      { id: 'kneeling', label: 'kneeling' },
      { id: 'lying', label: 'lying' },
      { id: 'walking', label: 'walking' },
      { id: 'holding', label: 'holding' },
      { id: 'leaning', label: 'leaning' },
      { id: 'sexual_action', label: 'sexual action' }
    ]
  },
  {
    id: 'expression',
    label: 'Expression / Emotion',
    details: [
      { id: 'smiling', label: 'smiling' },
      { id: 'blushing', label: 'blushing' },
      { id: 'angry', label: 'angry' },
      { id: 'sad', label: 'sad' },
      { id: 'surprised', label: 'surprised' },
      { id: 'open_mouth', label: 'open mouth' },
      { id: 'ahegao', label: 'ahegao' },
      { id: 'looking', label: 'looking' }
    ]
  },
  {
    id: 'scene',
    label: 'Scene / Background',
    details: [
      { id: 'bedroom', label: 'bedroom' },
      { id: 'outdoors', label: 'outdoors' },
      { id: 'indoors', label: 'indoors' },
      { id: 'bathroom', label: 'bathroom' },
      { id: 'office', label: 'office' },
      { id: 'beach', label: 'beach' },
      { id: 'street', label: 'street' },
      { id: 'park', label: 'park' },
      { id: 'bed', label: 'bed' }
    ]
  }
]

export type FitnessLevel = 'green' | 'yellow' | 'red'

export type HealthMetricId =
  | 'subject'
  | 'diversity'
  | 'detail'
  | 'length'
  | 'coverage'
  | 'penalty'

export interface HealthBreakdownItem {
  id: HealthMetricId
  label: string
  weight: number
  score: number
  weightContribution: number
  notes: string
}

export interface CaptionAnalysisResult {
  totalImages: number
  captionedCount: number
  emptyCount: number
  categoryDetails: Record<CaptionCategoryId, DetailCount[]>
  fitnessScore: number
  fitnessLevel: FitnessLevel
  healthBreakdown: HealthBreakdownItem[]
  strengths: string[]
  improvements: string[]
  summary: string
}

export function emptyClassification(): CaptionClassification {
  return {
    subject: [],
    camera: [],
    clothing: [],
    pose: [],
    expression: [],
    scene: []
  }
}

function emptyCategoryDetails(): Record<CaptionCategoryId, DetailCount[]> {
  const out = {} as Record<CaptionCategoryId, DetailCount[]>
  for (const cat of CAPTION_CATEGORIES) {
    out[cat.id] = cat.details.map((d) => ({
      detailId: d.id,
      label: d.label,
      count: 0
    }))
  }
  return out
}

/** Build id/label → canonical detail id maps per category for AI output validation. */
export function buildDetailLookup(): Record<CaptionCategoryId, Map<string, string>> {
  const out = {} as Record<CaptionCategoryId, Map<string, string>>
  for (const cat of CAPTION_CATEGORIES) {
    const map = new Map<string, string>()
    for (const d of cat.details) {
      map.set(d.id.toLowerCase(), d.id)
      map.set(d.label.toLowerCase(), d.id)
      map.set(d.label.toLowerCase().replace(/[-\s]+/g, '_'), d.id)
    }
    out[cat.id] = map
  }
  return out
}

export function fitnessLevelForScore(score: number): FitnessLevel {
  if (score >= 70) return 'green'
  if (score >= 40) return 'yellow'
  return 'red'
}

/**
 * Aggregate AI classifications for each caption (aligned by index with `captions`).
 * Fitness fields are filled by buildCaptionAnalysisResult (lora health score).
 */
export function aggregateCategoryDetails(
  captions: string[],
  classifications: CaptionClassification[]
): Pick<
  CaptionAnalysisResult,
  'totalImages' | 'captionedCount' | 'emptyCount' | 'categoryDetails'
> {
  const totalImages = captions.length
  const categoryDetails = emptyCategoryDetails()

  let captionedCount = 0
  for (let i = 0; i < totalImages; i++) {
    const trimmed = captions[i]?.trim() ?? ''
    if (!trimmed) continue
    captionedCount += 1
    const hits = classifications[i] ?? emptyClassification()
    for (const cat of CAPTION_CATEGORIES) {
      const ids = [...new Set(hits[cat.id] ?? [])]
      for (const detail of categoryDetails[cat.id]) {
        if (ids.includes(detail.detailId)) {
          detail.count += 1
        }
      }
    }
  }

  return {
    totalImages,
    captionedCount,
    emptyCount: totalImages - captionedCount,
    categoryDetails
  }
}
