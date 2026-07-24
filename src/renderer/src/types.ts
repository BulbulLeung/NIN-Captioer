import { createDefaultCaptionPreset, DEFAULT_CAPTION_PRESET_ID } from './defaults/captionPresets'
import {
  CAPTION_FORMAT_OPTIONS,
  DEFAULT_CAPTION_FORMAT,
  DEFAULT_WD14_SETTINGS,
  normalizeCaptionFormat,
  normalizeWd14Settings,
  type CaptionFormatId,
  type CaptionFormatOption,
  type Wd14Settings
} from './defaults/captionFormats'
import {
  DEFAULT_LORA_TRAIN_APP,
  DEFAULT_LORA_TRAIN_JOB,
  KREA2_RAW,
  KREA2_TURBO,
  normalizeActiveView,
  normalizeLoraTrainApp,
  normalizeLoraTrainJob,
  type ActiveView,
  type LoraTrainAppSettings,
  type LoraTrainJobConfig
} from './defaults/loraTrain'

export type {
  ActiveView,
  CaptionFormatId,
  CaptionFormatOption,
  LoraTrainAppSettings,
  LoraTrainJobConfig,
  Wd14Settings
}
export {
  CAPTION_FORMAT_OPTIONS,
  DEFAULT_CAPTION_FORMAT,
  DEFAULT_LORA_TRAIN_APP,
  DEFAULT_LORA_TRAIN_JOB,
  DEFAULT_WD14_SETTINGS,
  KREA2_RAW,
  KREA2_TURBO,
  normalizeActiveView,
  normalizeCaptionFormat,
  normalizeLoraTrainApp,
  normalizeLoraTrainJob,
  normalizeWd14Settings
}

export type TranslationProvider = 'lmstudio' | 'ollama'

export interface ResourceGpuVramApp {
  pid: number
  name: string
  memUsedMiB: number
  killable: boolean
}

export interface ResourceGpuStats {
  id: string
  name: string
  utilPercent: number
  memUsedMiB: number
  memTotalMiB: number
  tempC: number | null
  powerDrawW: number | null
  powerLimitW: number | null
  apps: ResourceGpuVramApp[]
}

export interface ResourceStats {
  cpuName: string
  cpuPercent: number
  ramUsedBytes: number
  ramTotalBytes: number
  gpu: ResourceGpuStats | null
}

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
  /** Auto Caption / reCaption output format (Natural VLM vs WD14 ONNX tags). */
  captionFormat: CaptionFormatId
  wd14: Wd14Settings
  sidebarWidth: number
  rightPaneWidth: number
  /** When true, analyze captions in the background; when false, only while Analyze dialog is open. */
  autoAnalysis: boolean
  listViewMode: ListViewMode
  thumbnailWidth: number
  activeView: ActiveView
  loraTrainJob: LoraTrainJobConfig
  loraTrainApp: LoraTrainAppSettings
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
  captionFormat: DEFAULT_CAPTION_FORMAT,
  wd14: { ...DEFAULT_WD14_SETTINGS },
  sidebarWidth: 260,
  rightPaneWidth: 380,
  autoAnalysis: true,
  listViewMode: 'list',
  thumbnailWidth: 96,
  activeView: 'datasetEdit',
  loraTrainJob: structuredClone(DEFAULT_LORA_TRAIN_JOB),
  loraTrainApp: { ...DEFAULT_LORA_TRAIN_APP }
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
  const loraTrainJob = normalizeLoraTrainJob(merged.loraTrainJob)
  // Seed dataset folder from DatasetEdit lastFolder when empty
  if (!loraTrainJob.datasets[0]?.folder_path && lastFolder) {
    loraTrainJob.datasets[0] = {
      ...loraTrainJob.datasets[0],
      folder_path: lastFolder
    }
  }
  return {
    ...merged,
    captionPresets: presets,
    activeCaptionPresetId: activeId,
    captionFormat: normalizeCaptionFormat(merged.captionFormat),
    wd14: normalizeWd14Settings(merged.wd14),
    model: merged.model ?? '',
    lastFolder,
    datasetFolders,
    sidebarWidth: clampSidebarWidth(merged.sidebarWidth ?? DEFAULT_SETTINGS.sidebarWidth),
    rightPaneWidth: clampRightPaneWidth(merged.rightPaneWidth ?? DEFAULT_SETTINGS.rightPaneWidth),
    autoAnalysis: merged.autoAnalysis !== false,
    listViewMode: normalizeListViewMode(merged.listViewMode),
    thumbnailWidth: clampThumbnailWidth(merged.thumbnailWidth ?? DEFAULT_SETTINGS.thumbnailWidth),
    activeView: normalizeActiveView(merged.activeView),
    loraTrainJob,
    loraTrainApp: normalizeLoraTrainApp(merged.loraTrainApp)
  }
}

declare global {
  interface Window {
    api: {
      openFolder: () => Promise<string | null>
      openFile: (opts?: {
        title?: string
        filters?: { name: string; extensions: string[] }[]
      }) => Promise<string | null>
      listImages: (dir: string) => Promise<ImageItem[]>
      readCaption: (imagePath: string) => Promise<string>
      writeCaption: (imagePath: string, text: string) => Promise<boolean>
      deleteImage: (imagePath: string) => Promise<{ ok: boolean }>
      readImageMeta: (imagePath: string) => Promise<{ positivePrompt: string }>
      readImageBase64: (imagePath: string) => Promise<{ mimeType: string; base64: string }>
      getSettings: () => Promise<AppSettings>
      setSettings: (settings: AppSettings) => Promise<boolean>
      saveTextFile: (opts: {
        defaultPath?: string
        content: string
        filters?: { name: string; extensions: string[] }[]
      }) => Promise<string | null>
      listGpuDevices: () => Promise<{ id: string; label: string }[]>
      getResourceStats: (deviceId?: string) => Promise<ResourceStats>
      killProcess: (pid: number) => Promise<{ ok: boolean; error?: string }>
      checkTrainEnv: (pythonPath?: string) => Promise<{ ok: boolean; message: string }>
      startTrain: (opts: {
        pythonPath?: string
        configJson: string
        device?: string
      }) => Promise<{ ok: boolean; error?: string; configPath?: string }>
      stopTrain: () => Promise<{ ok: boolean }>
      trainStatus: () => Promise<{ running: boolean }>
      onTrainLog: (
        cb: (payload: { line: string; stream: string }) => void
      ) => () => void
      onTrainProgress: (
        cb: (payload: { step: number; total: number; loss: number }) => void
      ) => () => void
      onTrainDone: (cb: (payload: { path: string }) => void) => () => void
      onTrainError: (cb: (payload: { message: string }) => void) => () => void
      defaultModelDownloadPath: () => Promise<string>
      checkModelStatus: (opts: {
        pythonPath?: string
        downloadPath?: string
        token?: string
        targets: { role: string; path: string }[]
      }) => Promise<{
        ok: boolean
        error?: string
        results: {
          role: string
          path: string
          repoId: string | null
          status: 'missing' | 'ready' | 'updateAvailable' | 'local' | 'error'
          localPath?: string | null
          localRevision?: string | null
          remoteRevision?: string | null
          message?: string | null
        }[]
        downloadPath?: string
      }>
      downloadModel: (opts: {
        pythonPath?: string
        downloadPath?: string
        token?: string
        repoId: string
      }) => Promise<{ ok: boolean; error?: string; downloadPath?: string }>
      cancelModelDownload: () => Promise<{ ok: boolean }>
      modelDownloadStatus: () => Promise<{ running: boolean }>
      onModelDownloadProgress: (
        cb: (payload: { repoId: string; pct: number }) => void
      ) => () => void
      onModelDownloadDone: (
        cb: (payload: { repoId: string; path: string; revision: string }) => void
      ) => () => void
      onModelDownloadError: (
        cb: (payload: { message: string; repoId: string }) => void
      ) => () => void
      ensureWd14Model: (opts: {
        pythonPath?: string
        downloadPath?: string
        token?: string
        repoId: string
      }) => Promise<{ ok: boolean; error?: string; modelDir?: string }>
      tagWd14: (opts: {
        pythonPath?: string
        modelDir: string
        threshold: number
        characterThreshold: number
        imagePaths: string[]
        ensure?: boolean
        downloadPath?: string
        token?: string
        repoId?: string
      }) => Promise<{
        ok: boolean
        error?: string
        results: { path: string; tags?: string; error?: string }[]
      }>
      cancelWd14: () => Promise<{ ok: boolean }>
      toLocalUrl: (filePath: string) => string
    }
  }
}
