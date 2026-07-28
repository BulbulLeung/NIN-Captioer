import { contextBridge, ipcRenderer, type IpcRendererEvent } from 'electron'

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
  datasetFolders: string[]
  captionPresets: CaptionPreset[]
  activeCaptionPresetId: string
  appendPositivePrompt?: boolean
  captionFormat?: string
  wd14?: {
    modelRepoId: string
    threshold: number
    characterThreshold: number
  }
  sidebarWidth: number
  rightPaneWidth: number
  autoAnalysis?: boolean
  listViewMode?: string
  thumbnailWidth?: number
  bucketPreview?: boolean
  activeView?: string
  loraTrainJob?: unknown
  loraTrainApp?: unknown
  uiGpuMode?: 'auto' | 'onboard' | 'software'
  /** @deprecated Prefer uiGpuMode; kept for reading older settings.json */
  disableUiGpu?: boolean
}

const api = {
  openFolder: (): Promise<string | null> => ipcRenderer.invoke('dialog:openFolder'),
  openFile: (opts?: {
    title?: string
    filters?: { name: string; extensions: string[] }[]
  }): Promise<string | null> => ipcRenderer.invoke('dialog:openFile', opts),
  listImages: (dir: string): Promise<ImageItem[]> => ipcRenderer.invoke('fs:listImages', dir),
  scanArBuckets: (opts: {
    folder: string
    resolutions: number[]
    pythonPath?: string
  }): Promise<{
    ok: boolean
    error?: string
    imageCount?: number
    forcedUpscale?: number
    countsOrdered?: { bucket: string; count: number }[]
  }> => ipcRenderer.invoke('dataset:scanArBuckets', opts),
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
  relaunchApp: (): Promise<boolean> => ipcRenderer.invoke('app:relaunch'),
  listGpuDevices: (): Promise<{ id: string; label: string }[]> =>
    ipcRenderer.invoke('gpu:listDevices'),
  getResourceStats: (deviceId?: string): Promise<{
    cpuName: string
    cpuPercent: number
    ramUsedBytes: number
    ramTotalBytes: number
    gpu: null | {
      id: string
      name: string
      utilPercent: number
      memUsedMiB: number
      memTotalMiB: number
      tempC: number | null
      powerDrawW: number | null
      powerLimitW: number | null
      apps: { pid: number; name: string; memUsedMiB: number; killable: boolean }[]
    }
  }> => ipcRenderer.invoke('system:getResourceStats', deviceId),
  killProcess: (pid: number): Promise<{ ok: boolean; error?: string }> =>
    ipcRenderer.invoke('system:killProcess', pid),
  defaultDownloadFolder: (): Promise<string> =>
    ipcRenderer.invoke('download:defaultFolder'),
  probePython: (
    pythonPath?: string
  ): Promise<{
    status: 'ready' | 'missingPython' | 'missingPackages' | 'error'
    message: string
    pythonPath?: string
    version?: string
    cuda?: boolean
    krea?: boolean
    missing?: string[]
  }> => ipcRenderer.invoke('python:probe', pythonPath),
  installPython: (opts?: {
    installPath?: string
  }): Promise<{ ok: boolean; pythonPath?: string; message: string }> =>
    ipcRenderer.invoke('python:install', opts),
  cancelPythonInstall: (): Promise<{ ok: boolean }> =>
    ipcRenderer.invoke('python:cancelInstall'),
  onPythonInstallProgress: (
    cb: (payload: { stage: string; message: string; pct: number }) => void
  ) => {
    const listener = (
      _e: IpcRendererEvent,
      payload: { stage: string; message: string; pct: number }
    ) => cb(payload)
    ipcRenderer.on('python:installProgress', listener)
    return () => ipcRenderer.removeListener('python:installProgress', listener)
  },
  startTrain: (opts: {
    pythonPath?: string
    configJson: string
    device?: string
  }): Promise<{ ok: boolean; error?: string; configPath?: string }> =>
    ipcRenderer.invoke('train:start', opts),
  stopTrain: (): Promise<{ ok: boolean }> => ipcRenderer.invoke('train:stop'),
  trainStatus: (): Promise<{ running: boolean }> => ipcRenderer.invoke('train:status'),
  listTrainCheckpoints: (opts: {
    trainingFolder: string
    jobName: string
  }): Promise<{
    ok: boolean
    error?: string
    checkpoints: { step: number; path: string }[]
  }> => ipcRenderer.invoke('train:listCheckpoints', opts),
  listTrainSamples: (opts: {
    trainingFolder: string
    jobName: string
  }): Promise<{
    ok: boolean
    error?: string
    samples: { path: string; name: string; mtimeMs: number; step?: number; promptIndex: number }[]
  }> => ipcRenderer.invoke('train:listSamples', opts),
  onTrainLog: (cb: (payload: { line: string; stream: string }) => void) => {
    const listener = (_e: IpcRendererEvent, payload: { line: string; stream: string }) =>
      cb(payload)
    ipcRenderer.on('train:log', listener)
    return () => ipcRenderer.removeListener('train:log', listener)
  },
  onTrainProgress: (
    cb: (payload: { step: number; total: number; loss: number }) => void
  ) => {
    const listener = (
      _e: IpcRendererEvent,
      payload: { step: number; total: number; loss: number }
    ) => cb(payload)
    ipcRenderer.on('train:progress', listener)
    return () => ipcRenderer.removeListener('train:progress', listener)
  },
  onTrainDone: (cb: (payload: { path: string }) => void) => {
    const listener = (_e: IpcRendererEvent, payload: { path: string }) => cb(payload)
    ipcRenderer.on('train:done', listener)
    return () => ipcRenderer.removeListener('train:done', listener)
  },
  onTrainError: (cb: (payload: { message: string }) => void) => {
    const listener = (_e: IpcRendererEvent, payload: { message: string }) => cb(payload)
    ipcRenderer.on('train:error', listener)
    return () => ipcRenderer.removeListener('train:error', listener)
  },
  checkModelStatus: (opts: {
    pythonPath?: string
    downloadPath?: string
    token?: string
    targets: { role: string; path: string }[]
  }): Promise<{
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
  }> => ipcRenderer.invoke('model:checkStatus', opts),
  downloadModel: (opts: {
    pythonPath?: string
    downloadPath?: string
    token?: string
    repoId: string
  }): Promise<{ ok: boolean; error?: string; downloadPath?: string }> =>
    ipcRenderer.invoke('model:download', opts),
  cancelModelDownload: (): Promise<{ ok: boolean }> =>
    ipcRenderer.invoke('model:cancelDownload'),
  onModelDownloadProgress: (cb: (payload: {
    repoId: string
    pct: number
    done?: number
    total?: number
  }) => void) => {
    const listener = (
      _e: IpcRendererEvent,
      payload: { repoId: string; pct: number; done?: number; total?: number }
    ) => cb(payload)
    ipcRenderer.on('model:downloadProgress', listener)
    return () => ipcRenderer.removeListener('model:downloadProgress', listener)
  },
  onModelDownloadDone: (
    cb: (payload: { repoId: string; path: string; revision: string }) => void
  ) => {
    const listener = (
      _e: IpcRendererEvent,
      payload: { repoId: string; path: string; revision: string }
    ) => cb(payload)
    ipcRenderer.on('model:downloadDone', listener)
    return () => ipcRenderer.removeListener('model:downloadDone', listener)
  },
  onModelDownloadError: (cb: (payload: { message: string; repoId: string }) => void) => {
    const listener = (_e: IpcRendererEvent, payload: { message: string; repoId: string }) =>
      cb(payload)
    ipcRenderer.on('model:downloadError', listener)
    return () => ipcRenderer.removeListener('model:downloadError', listener)
  },
  ensureWd14Model: (opts: {
    pythonPath?: string
    downloadPath?: string
    token?: string
    repoId: string
  }): Promise<{ ok: boolean; error?: string; modelDir?: string }> =>
    ipcRenderer.invoke('wd14:ensureModel', opts),
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
  }): Promise<{
    ok: boolean
    error?: string
    results: { path: string; tags?: string; error?: string }[]
  }> => ipcRenderer.invoke('wd14:tag', opts),
  cancelWd14: (): Promise<{ ok: boolean }> => ipcRenderer.invoke('wd14:cancel'),
  toLocalUrl: (filePath: string): string => {
    const normalized = filePath.replace(/\\/g, '/')
    const encoded = normalized.split('/').map((seg) => encodeURIComponent(seg)).join('/')
    return `local-file://local/${encoded}`
  }
}

contextBridge.exposeInMainWorld('api', api)

export type ElectronApi = typeof api
