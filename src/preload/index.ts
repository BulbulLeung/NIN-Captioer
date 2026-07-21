import { contextBridge, ipcRenderer } from 'electron'

export interface ImageItem {
  path: string
  name: string
  hasCaption: boolean
}

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
}

const api = {
  openFolder: (): Promise<string | null> => ipcRenderer.invoke('dialog:openFolder'),
  listImages: (dir: string): Promise<ImageItem[]> => ipcRenderer.invoke('fs:listImages', dir),
  readCaption: (imagePath: string): Promise<string> =>
    ipcRenderer.invoke('fs:readCaption', imagePath),
  writeCaption: (imagePath: string, text: string): Promise<boolean> =>
    ipcRenderer.invoke('fs:writeCaption', imagePath, text),
  deleteImage: (imagePath: string): Promise<{ ok: boolean }> =>
    ipcRenderer.invoke('fs:deleteImage', imagePath),
  readImageMeta: (imagePath: string): Promise<{ positivePrompt: string }> =>
    ipcRenderer.invoke('fs:readImageMeta', imagePath),
  readImageBase64: (imagePath: string): Promise<{ mimeType: string; base64: string }> =>
    ipcRenderer.invoke('fs:readImageBase64', imagePath),
  getSettings: (): Promise<AppSettings> => ipcRenderer.invoke('settings:get'),
  setSettings: (settings: AppSettings): Promise<boolean> =>
    ipcRenderer.invoke('settings:set', settings),
  toLocalUrl: (filePath: string): string => {
    const normalized = filePath.replace(/\\/g, '/')
    const win = normalized.match(/^([A-Za-z]:)(\/.*)?$/)
    if (win) {
      const drive = win[1]
      const rest = (win[2] || '')
        .split('/')
        .map((seg) => encodeURIComponent(seg))
        .join('/')
      return `local-file:///${drive}${rest}`
    }
    return `local-file:///${normalized.split('/').map(encodeURIComponent).join('/')}`
  }
}

contextBridge.exposeInMainWorld('api', api)

export type ElectronApi = typeof api
