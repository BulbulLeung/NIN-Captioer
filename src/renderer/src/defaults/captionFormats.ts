/** Extensible caption format ids used by Auto Caption / reCaption. */
export type CaptionFormatId = 'natural' | 'wd14'

export interface CaptionFormatOption {
  id: CaptionFormatId
  label: string
}

/** Toolbar dropdown options — add new formats here and extend CaptionFormatId. */
export const CAPTION_FORMAT_OPTIONS: CaptionFormatOption[] = [
  { id: 'natural', label: 'Natural Language(Flux/Krea2)' },
  { id: 'wd14', label: 'Danbooru Tags(SD/XL)' }
]

export const DEFAULT_CAPTION_FORMAT: CaptionFormatId = 'natural'

export interface Wd14Settings {
  modelRepoId: string
  threshold: number
  characterThreshold: number
}

export const DEFAULT_WD14_SETTINGS: Wd14Settings = {
  modelRepoId: 'SmilingWolf/wd-swinv2-tagger-v3',
  threshold: 0.35,
  characterThreshold: 0.85
}

export function normalizeCaptionFormat(value: unknown): CaptionFormatId {
  if (value === 'wd14' || value === 'natural') return value
  return DEFAULT_CAPTION_FORMAT
}

export function normalizeWd14Settings(raw: unknown): Wd14Settings {
  const src =
    raw && typeof raw === 'object' ? (raw as Partial<Wd14Settings>) : ({} as Partial<Wd14Settings>)
  const threshold =
    typeof src.threshold === 'number' && Number.isFinite(src.threshold)
      ? Math.min(1, Math.max(0, src.threshold))
      : DEFAULT_WD14_SETTINGS.threshold
  const characterThreshold =
    typeof src.characterThreshold === 'number' && Number.isFinite(src.characterThreshold)
      ? Math.min(1, Math.max(0, src.characterThreshold))
      : DEFAULT_WD14_SETTINGS.characterThreshold
  const modelRepoId =
    typeof src.modelRepoId === 'string' && src.modelRepoId.trim()
      ? src.modelRepoId.trim()
      : DEFAULT_WD14_SETTINGS.modelRepoId
  return { modelRepoId, threshold, characterThreshold }
}
