import { app, BrowserWindow, dialog, ipcMain, protocol, screen, Menu, shell } from 'electron'
import { join, dirname, basename, extname } from 'path'
import { readFileSync, writeFileSync } from 'fs'
import { readFile, writeFile, readdir, access, constants } from 'fs/promises'

const IMAGE_EXTS = new Set(['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'])

type TranslationProvider = 'lmstudio' | 'ollama'

interface CaptionPreset {
  id: string
  name: string
  prompt: string
}

interface WindowState {
  width: number
  height: number
  x: number | null
  y: number | null
  isMaximized: boolean
}

interface AppSettings {
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
  autoAnalysis: boolean
  windowWidth: number
  windowHeight: number
  windowX: number | null
  windowY: number | null
  windowMaximized: boolean
}

const DEFAULT_WINDOW: WindowState = {
  width: 1280,
  height: 840,
  x: null,
  y: null,
  isMaximized: false
}

const DEFAULT_SETTINGS: AppSettings = {
  provider: 'lmstudio',
  lmStudioBaseUrl: 'http://localhost:1234/v1',
  ollamaBaseUrl: 'http://localhost:11434',
  model: '',
  targetLanguage: 'zh-TW',
  lastFolder: null,
  datasetFolders: [],
  captionPresets: [],
  activeCaptionPresetId: '',
  sidebarWidth: 260,
  rightPaneWidth: 380,
  autoAnalysis: true,
  windowWidth: DEFAULT_WINDOW.width,
  windowHeight: DEFAULT_WINDOW.height,
  windowX: null,
  windowY: null,
  windowMaximized: false
}

function settingsPath(): string {
  return join(app.getPath('userData'), 'settings.json')
}

async function loadSettings(): Promise<AppSettings> {
  try {
    const raw = await readFile(settingsPath(), 'utf-8')
    return { ...DEFAULT_SETTINGS, ...JSON.parse(raw) }
  } catch {
    return { ...DEFAULT_SETTINGS }
  }
}

async function saveSettings(settings: AppSettings): Promise<void> {
  await writeFile(settingsPath(), JSON.stringify(settings, null, 2), 'utf-8')
}

function getWindowState(settings: AppSettings): WindowState {
  return {
    width: settings.windowWidth || DEFAULT_WINDOW.width,
    height: settings.windowHeight || DEFAULT_WINDOW.height,
    x: settings.windowX ?? null,
    y: settings.windowY ?? null,
    isMaximized: Boolean(settings.windowMaximized)
  }
}

function isVisibleOnAnyDisplay(bounds: { x: number; y: number; width: number; height: number }): boolean {
  const displays = screen.getAllDisplays()
  return displays.some((d) => {
    const a = d.workArea
    const overlapX = Math.max(0, Math.min(bounds.x + bounds.width, a.x + a.width) - Math.max(bounds.x, a.x))
    const overlapY = Math.max(0, Math.min(bounds.y + bounds.height, a.y + a.height) - Math.max(bounds.y, a.y))
    return overlapX >= 80 && overlapY >= 80
  })
}

async function persistWindowState(win: BrowserWindow): Promise<void> {
  const isMaximized = win.isMaximized()
  const bounds = isMaximized ? win.getNormalBounds() : win.getBounds()
  const current = await loadSettings()
  await saveSettings({
    ...current,
    windowWidth: bounds.width,
    windowHeight: bounds.height,
    windowX: bounds.x,
    windowY: bounds.y,
    windowMaximized: isMaximized
  })
}

function captionPathForImage(imagePath: string): string {
  const dir = dirname(imagePath)
  const stem = basename(imagePath, extname(imagePath))
  return join(dir, `${stem}.txt`)
}

async function fileExists(path: string): Promise<boolean> {
  try {
    await access(path, constants.F_OK)
    return true
  } catch {
    return false
  }
}

function mimeForExt(ext: string): string {
  switch (ext.toLowerCase()) {
    case '.jpg':
    case '.jpeg':
      return 'image/jpeg'
    case '.png':
      return 'image/png'
    case '.webp':
      return 'image/webp'
    case '.gif':
      return 'image/gif'
    case '.bmp':
      return 'image/bmp'
    default:
      return 'application/octet-stream'
  }
}

/** Extract tEXt / uncompressed iTXt key-value pairs from a PNG buffer. */
function extractPngTextChunks(buf: Buffer): Record<string, string> {
  const signature = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])
  if (buf.length < 8 || !buf.subarray(0, 8).equals(signature)) return {}

  const texts: Record<string, string> = {}
  let offset = 8

  while (offset + 8 <= buf.length) {
    const length = buf.readUInt32BE(offset)
    const type = buf.toString('ascii', offset + 4, offset + 8)
    const dataStart = offset + 8
    const dataEnd = dataStart + length
    if (dataEnd + 4 > buf.length) break

    if (type === 'tEXt') {
      const data = buf.subarray(dataStart, dataEnd)
      const nullIdx = data.indexOf(0)
      if (nullIdx > 0) {
        const key = data.toString('latin1', 0, nullIdx)
        const value = data.toString('latin1', nullIdx + 1)
        texts[key] = value
      }
    } else if (type === 'iTXt') {
      const data = buf.subarray(dataStart, dataEnd)
      let p = 0
      const nextNull = () => {
        const i = data.indexOf(0, p)
        if (i < 0) return null
        const s = data.toString('utf8', p, i)
        p = i + 1
        return s
      }
      const key = nextNull()
      if (key) {
        const compressionFlag = data[p]
        p += 1
        p += 1 // compression method
        nextNull() // language
        nextNull() // translated keyword
        if (compressionFlag === 0 && p <= data.length) {
          texts[key] = data.toString('utf8', p)
        }
      }
    }

    if (type === 'IEND') break
    offset = dataEnd + 4
  }

  return texts
}

function positivePromptFromParameters(params: string): string {
  const match = params.match(/\nNegative prompt:/i)
  if (match && match.index !== undefined) {
    return params.slice(0, match.index).trim()
  }
  return params.trim()
}

function extractPositivePrompt(texts: Record<string, string>): string {
  if (texts.prompt?.trim()) return texts.prompt.trim()
  if (texts.parameters?.trim()) return positivePromptFromParameters(texts.parameters)
  // Some tools use Title / Description
  if (texts.Description?.trim()) return texts.Description.trim()
  return ''
}

let mainWindow: BrowserWindow | null = null
let saveWindowTimer: ReturnType<typeof setTimeout> | null = null

function scheduleSaveWindowState(win: BrowserWindow): void {
  if (saveWindowTimer) clearTimeout(saveWindowTimer)
  saveWindowTimer = setTimeout(() => {
    saveWindowTimer = null
    void persistWindowState(win)
  }, 400)
}

async function createWindow(): Promise<void> {
  const settings = await loadSettings()
  const saved = getWindowState(settings)

  const options: Electron.BrowserWindowConstructorOptions = {
    width: Math.max(900, saved.width),
    height: Math.max(600, saved.height),
    minWidth: 900,
    minHeight: 600,
    title: `${app.getName()} Ver${app.getVersion()}`,
    show: false,
    autoHideMenuBar: true,
    // Dev: build/icon.ico. Packaged Windows builds use the exe icon from electron-builder.
    ...(!app.isPackaged
      ? { icon: join(__dirname, '../../build/icon.ico') }
      : {}),
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false
    }
  }

  if (
    saved.x !== null &&
    saved.y !== null &&
    isVisibleOnAnyDisplay({
      x: saved.x,
      y: saved.y,
      width: options.width!,
      height: options.height!
    })
  ) {
    options.x = saved.x
    options.y = saved.y
  }

  mainWindow = new BrowserWindow(options)
  mainWindow.setMenuBarVisibility(false)
  Menu.setApplicationMenu(null)
  // Keep BrowserWindow title (from package.json version); ignore document.title from index.html
  mainWindow.on('page-title-updated', (event) => {
    event.preventDefault()
  })

  mainWindow.once('ready-to-show', () => {
    if (!mainWindow) return
    if (saved.isMaximized) mainWindow.maximize()
    mainWindow.show()
  })

  mainWindow.on('resize', () => {
    if (mainWindow && !mainWindow.isMinimized()) scheduleSaveWindowState(mainWindow)
  })
  mainWindow.on('move', () => {
    if (mainWindow && !mainWindow.isMinimized()) scheduleSaveWindowState(mainWindow)
  })
  mainWindow.on('maximize', () => {
    if (mainWindow) scheduleSaveWindowState(mainWindow)
  })
  mainWindow.on('unmaximize', () => {
    if (mainWindow) scheduleSaveWindowState(mainWindow)
  })
  mainWindow.on('close', () => {
    if (saveWindowTimer) {
      clearTimeout(saveWindowTimer)
      saveWindowTimer = null
    }
    if (!mainWindow) return
    const win = mainWindow
    const isMaximized = win.isMaximized()
    const bounds = isMaximized ? win.getNormalBounds() : win.getBounds()
    try {
      let current: AppSettings = { ...DEFAULT_SETTINGS }
      try {
        current = { ...DEFAULT_SETTINGS, ...JSON.parse(readFileSync(settingsPath(), 'utf-8')) }
      } catch {
        // use defaults
      }
      writeFileSync(
        settingsPath(),
        JSON.stringify(
          {
            ...current,
            windowWidth: bounds.width,
            windowHeight: bounds.height,
            windowX: bounds.x,
            windowY: bounds.y,
            windowMaximized: isMaximized
          },
          null,
          2
        ),
        'utf-8'
      )
    } catch {
      // Best-effort; resize handlers already debounce-save
    }
  })

  if (process.env.ELECTRON_RENDERER_URL) {
    mainWindow.loadURL(process.env.ELECTRON_RENDERER_URL)
  } else {
    mainWindow.loadFile(join(__dirname, '../renderer/index.html'))
  }
}

protocol.registerSchemesAsPrivileged([
  {
    scheme: 'local-file',
    privileges: {
      standard: true,
      secure: true,
      supportFetchAPI: true,
      bypassCSP: true,
      stream: true
    }
  }
])

app.whenReady().then(async () => {
  protocol.handle('local-file', async (request) => {
    const parsed = new URL(request.url)
    let filePath = decodeURIComponent(parsed.pathname)
    if (filePath.startsWith('/')) filePath = filePath.slice(1)
    try {
      const buf = await readFile(filePath)
      const mime = mimeForExt(extname(filePath))
      return new Response(buf, { headers: { 'Content-Type': mime } })
    } catch {
      return new Response('Not Found', { status: 404 })
    }
  })

  ipcMain.handle('dialog:openFolder', async () => {
    const result = await dialog.showOpenDialog(mainWindow!, {
      properties: ['openDirectory']
    })
    if (result.canceled || result.filePaths.length === 0) return null
    return result.filePaths[0]
  })

  ipcMain.handle('fs:listImages', async (_event, dir: string) => {
    const entries = await readdir(dir, { withFileTypes: true })
    const images: { path: string; name: string; hasCaption: boolean }[] = []

    for (const entry of entries) {
      if (!entry.isFile()) continue
      const ext = extname(entry.name).toLowerCase()
      if (!IMAGE_EXTS.has(ext)) continue
      const imagePath = join(dir, entry.name)
      const hasCaption = await fileExists(captionPathForImage(imagePath))
      images.push({ path: imagePath, name: entry.name, hasCaption })
    }

    images.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }))
    return images
  })

  ipcMain.handle('fs:readCaption', async (_event, imagePath: string) => {
    const txtPath = captionPathForImage(imagePath)
    try {
      return await readFile(txtPath, 'utf-8')
    } catch {
      return ''
    }
  })

  ipcMain.handle('fs:writeCaption', async (_event, imagePath: string, text: string) => {
    const txtPath = captionPathForImage(imagePath)
    await writeFile(txtPath, text, 'utf-8')
    return true
  })

  ipcMain.handle('fs:deleteImage', async (_event, imagePath: string) => {
    await shell.trashItem(imagePath)
    const txtPath = captionPathForImage(imagePath)
    try {
      await shell.trashItem(txtPath)
    } catch {
      // Caption file may not exist
    }
    return { ok: true }
  })

  ipcMain.handle('fs:readImageMeta', async (_event, imagePath: string) => {
    const ext = extname(imagePath).toLowerCase()
    if (ext !== '.png') {
      return { positivePrompt: '' }
    }
    try {
      const buf = await readFile(imagePath)
      const texts = extractPngTextChunks(buf)
      return { positivePrompt: extractPositivePrompt(texts) }
    } catch {
      return { positivePrompt: '' }
    }
  })

  ipcMain.handle('fs:readImageBase64', async (_event, imagePath: string) => {
    const buf = await readFile(imagePath)
    const ext = extname(imagePath)
    return {
      mimeType: mimeForExt(ext),
      base64: buf.toString('base64')
    }
  })

  ipcMain.handle('settings:get', async () => loadSettings())

  ipcMain.handle('settings:set', async (_event, settings: Partial<AppSettings>) => {
    const current = await loadSettings()
    await saveSettings({
      ...current,
      ...settings,
      // Window geometry is owned by main; never let renderer wipe it
      windowWidth: current.windowWidth,
      windowHeight: current.windowHeight,
      windowX: current.windowX,
      windowY: current.windowY,
      windowMaximized: current.windowMaximized
    })
    return true
  })

  await createWindow()

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) void createWindow()
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})
