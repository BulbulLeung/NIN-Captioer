import { createDefaultCaptionPreset, DEFAULT_CAPTION_PRESET_ID } from './defaults/captionPresets'

export type TranslationProvider = 'lmstudio' | 'ollama'

export interface CaptionPreset {
  id: string
  name: string
  prompt: string
}

export interface AppSettings {
  provider: TranslationProvider
  lmStudioBaseUrl: string
  ollamaBaseUrl: string
  model: string
  targetLanguage: string
  lastFolder: string | null
  captionPresets: CaptionPreset[]
  activeCaptionPresetId: string
  sidebarWidth: number
  rightPaneWidth: number
  /** When true, analyze captions in the background; when false, only while Analyze dialog is open. */
  autoAnalysis: boolean
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
  captionPresets: [defaultPreset],
  activeCaptionPresetId: DEFAULT_CAPTION_PRESET_ID,
  sidebarWidth: 260,
  rightPaneWidth: 380,
  autoAnalysis: true
}

const SIDEBAR_MIN = 160
const SIDEBAR_MAX = 480
const RIGHT_PANE_MIN = 280
const RIGHT_PANE_MAX = 720

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
  return {
    ...merged,
    captionPresets: presets,
    activeCaptionPresetId: activeId,
    model: merged.model ?? '',
    lastFolder: merged.lastFolder ?? null,
    sidebarWidth: clampSidebarWidth(merged.sidebarWidth ?? DEFAULT_SETTINGS.sidebarWidth),
    rightPaneWidth: clampRightPaneWidth(merged.rightPaneWidth ?? DEFAULT_SETTINGS.rightPaneWidth),
    autoAnalysis: merged.autoAnalysis !== false
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
