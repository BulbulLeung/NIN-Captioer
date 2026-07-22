import { createDefaultCaptionPreset, DEFAULT_CAPTION_PRESET_ID } from './defaults/captionPresets'

export type TranslationProvider = 'lmstudio' | 'ollama'

export interface CaptionPreset {
  id: string
  name: string
  prompt: string
}

export type ListViewMode = 'list' | 'thumbnails'

export interface AppSettings {
  provider: TranslationProvider
  lmStudioBaseUrl: string
  ollamaBaseUrl: string
  model: string
  targetLanguage: string
  lastFolder: string | null
  datasetFolders: string[]
  captionPresets: CaptionPreset[]
  activeCaptionPresetId: string
  sidebarWidth: number
  rightPaneWidth: number
  /** When true, analyze captions in the background; when false, only while Analyze dialog is open. */
  autoAnalysis: boolean
  listViewMode: ListViewMode
  thumbnailWidth: number
}

export interface ImageItem {
  path: string
  name: string
  hasCaption: boolean
}

export interface LanguageOption {
  code: string
  label: string
}

export const LANGUAGES: LanguageOption[] = [
  { code: 'zh-TW', label: '繁體中文' },
  { code: 'zh-CN', label: '简体中文' },
  { code: 'ja', label: '日本語' },
  { code: 'ko', label: '한국어' },
  { code: 'fr', label: 'Français' },
  { code: 'de', label: 'Deutsch' },
  { code: 'es', label: 'Español' },
  { code: 'it', label: 'Italiano' },
  { code: 'pt', label: 'Português' },
  { code: 'ru', label: 'Русский' }
]

const defaultPreset = createDefaultCaptionPreset()

export const DEFAULT_SETTINGS: AppSettings = {
  provider: 'lmstudio',
  lmStudioBaseUrl: 'http://localhost:1234/v1',
  ollamaBaseUrl: 'http://localhost:11434',
  model: '',
  targetLanguage: 'zh-TW',
  lastFolder: null,
  datasetFolders: [],
  captionPresets: [defaultPreset],
  activeCaptionPresetId: DEFAULT_CAPTION_PRESET_ID,
  sidebarWidth: 260,
  rightPaneWidth: 380,
  autoAnalysis: true,
  listViewMode: 'list',
  thumbnailWidth: 96
}

const SIDEBAR_MIN = 160
const SIDEBAR_MAX = 480
const RIGHT_PANE_MIN = 280
const RIGHT_PANE_MAX = 720
const THUMB_MIN = 48
const THUMB_MAX = 160

function clamp(n: number, min: number, max: number): number {
  if (!Number.isFinite(n)) return min
  return Math.min(max, Math.max(min, Math.round(n)))
}

export function clampSidebarWidth(n: number): number {
  return clamp(n, SIDEBAR_MIN, SIDEBAR_MAX)
}

export function clampRightPaneWidth(n: number): number {
  return clamp(n, RIGHT_PANE_MIN, RIGHT_PANE_MAX)
}

export function clampThumbnailWidth(n: number): number {
  return clamp(n, THUMB_MIN, THUMB_MAX)
}

export function normalizeListViewMode(value: unknown): ListViewMode {
  return value === 'thumbnails' ? 'thumbnails' : 'list'
}

function normalizeDatasetFolders(raw: unknown, lastFolder: string | null): string[] {
  const seen = new Set<string>()
  const folders: string[] = []
  if (Array.isArray(raw)) {
    for (const item of raw) {
      if (typeof item !== 'string' || !item) continue
      if (seen.has(item)) continue
      seen.add(item)
      folders.push(item)
    }
  }
  if (lastFolder && !seen.has(lastFolder)) {
    folders.unshift(lastFolder)
  }
  return folders
}

export function normalizeSettings(raw: Partial<AppSettings> | null | undefined): AppSettings {
  const merged = { ...DEFAULT_SETTINGS, ...raw }
  let presets = Array.isArray(merged.captionPresets) ? merged.captionPresets.filter(Boolean) : []
  if (presets.length === 0) {
    presets = [createDefaultCaptionPreset()]
  }
  let activeId = merged.activeCaptionPresetId
  if (!presets.some((p) => p.id === activeId)) {
    activeId = presets[0].id
  }
  let lastFolder = merged.lastFolder ?? null
  const datasetFolders = normalizeDatasetFolders(merged.datasetFolders, lastFolder)
  if (!lastFolder && datasetFolders.length > 0) {
    lastFolder = datasetFolders[0]
  }
  if (lastFolder && !datasetFolders.includes(lastFolder) && datasetFolders.length > 0) {
    lastFolder = datasetFolders[0]
  }
  return {
    ...merged,
    captionPresets: presets,
    activeCaptionPresetId: activeId,
    model: merged.model ?? '',
    lastFolder,
    datasetFolders,
    sidebarWidth: clampSidebarWidth(merged.sidebarWidth ?? DEFAULT_SETTINGS.sidebarWidth),
    rightPaneWidth: clampRightPaneWidth(merged.rightPaneWidth ?? DEFAULT_SETTINGS.rightPaneWidth),
    autoAnalysis: merged.autoAnalysis !== false,
    listViewMode: normalizeListViewMode(merged.listViewMode),
    thumbnailWidth: clampThumbnailWidth(merged.thumbnailWidth ?? DEFAULT_SETTINGS.thumbnailWidth)
  }
}

declare global {
  interface Window {
    api: {
      openFolder: () => Promise<string | null>
      listImages: (dir: string) => Promise<ImageItem[]>
      readCaption: (imagePath: string) => Promise<string>
      writeCaption: (imagePath: string, text: string) => Promise<boolean>
      deleteImage: (imagePath: string) => Promise<{ ok: boolean }>
      readImageMeta: (imagePath: string) => Promise<{ positivePrompt: string }>
      readImageBase64: (imagePath: string) => Promise<{ mimeType: string; base64: string }>
      getSettings: () => Promise<AppSettings>
      setSettings: (settings: AppSettings) => Promise<boolean>
      toLocalUrl: (filePath: string) => string
    }
  }
}
