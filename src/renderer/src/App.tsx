import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties, type PointerEvent as ReactPointerEvent } from 'react'
import { AnalysisDialog } from './components/AnalysisDialog'
import { CaptionEditor } from './components/CaptionEditor'
import { DeleteConfirmDialog } from './components/DeleteConfirmDialog'
import { ImageList } from './components/ImageList'
import { ImagePreview } from './components/ImagePreview'
import { SettingsDialog } from './components/SettingsDialog'
import { UnsavedDialog } from './components/UnsavedDialog'
import { useBidirectionalTranslate } from './hooks/useBidirectionalTranslate'
import { generateCaptionForImage } from './services/captioning'
import type { AppSettings, ImageItem } from './types'
import {
  clampRightPaneWidth,
  clampSidebarWidth,
  normalizeSettings
} from './types'

type ConfirmAction = 'save' | 'discard' | 'cancel'

export default function App() {
  const [folder, setFolder] = useState<string | null>(null)
  const [images, setImages] = useState<ImageItem[]>([])
  const [selectedPath, setSelectedPath] = useState<string | null>(null)
  const [english, setEnglish] = useState('')
  const [savedEnglish, setSavedEnglish] = useState('')
  const [translated, setTranslated] = useState('')
  const [settings, setSettings] = useState<AppSettings>(() => normalizeSettings(null))
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [analysisOpen, setAnalysisOpen] = useState(false)
  const [unsavedOpen, setUnsavedOpen] = useState(false)
  const [deleteOpen, setDeleteOpen] = useState(false)
  const [status, setStatus] = useState('')
  const [batchCaptioning, setBatchCaptioning] = useState(false)
  const [singleCaptioning, setSingleCaptioning] = useState(false)
  const [draggingPane, setDraggingPane] = useState<'sidebar' | 'right' | null>(null)
  const unsavedResolver = useRef<((action: ConfirmAction) => void) | null>(null)
  const captionAbortRef = useRef<AbortController | null>(null)
  const batchCancelRef = useRef(false)
  const settingsRef = useRef(settings)
  settingsRef.current = settings
  const selectedPathRef = useRef(selectedPath)
  selectedPathRef.current = selectedPath

  const dirty = english !== savedEnglish
  const dirtyPaths = useMemo(() => {
    const set = new Set<string>()
    if (selectedPath && dirty) set.add(selectedPath)
    return set
  }, [selectedPath, dirty])

  const missingCount = useMemo(
    () => images.filter((img) => !img.hasCaption).length,
    [images]
  )

  const selectedFileName = useMemo(() => {
    if (!selectedPath) return ''
    return selectedPath.split(/[/\\]/).pop() ?? selectedPath
  }, [selectedPath])

  const {
    translating,
    error,
    setError,
    cancelInFlight,
    scheduleEnglishToTarget,
    scheduleTargetToEnglish,
    translateEnglishToTargetNow,
    setEnglishSnapshot
  } = useBidirectionalTranslate({
    settings,
    setEnglish: (value) => {
      setEnglish(value)
      setEnglishSnapshot(value)
    },
    setTranslated,
    enabled: Boolean(selectedPath)
  })

  const applyCaptionToUi = useCallback(
    async (imagePath: string, caption: string) => {
      await window.api.writeCaption(imagePath, caption)
      setImages((prev) =>
        prev.map((img) => (img.path === imagePath ? { ...img, hasCaption: true } : img))
      )
      if (selectedPathRef.current === imagePath) {
        setEnglish(caption)
        setSavedEnglish(caption)
        setEnglishSnapshot(caption)
        setTranslated('')
        if (caption.trim()) translateEnglishToTargetNow(caption)
      }
    },
    [setEnglishSnapshot, translateEnglishToTargetNow]
  )

  const loadImage = useCallback(
    async (imagePath: string) => {
      cancelInFlight()
      const caption = await window.api.readCaption(imagePath)
      setSelectedPath(imagePath)
      setEnglish(caption)
      setSavedEnglish(caption)
      setTranslated('')
      setEnglishSnapshot(caption)
      setError(null)
      if (caption.trim()) {
        translateEnglishToTargetNow(caption)
      }
    },
    [cancelInFlight, setEnglishSnapshot, setError, translateEnglishToTargetNow]
  )

  const loadFolder = useCallback(
    async (dir: string, persist = true) => {
      cancelInFlight()
      setTranslated('')
      setEnglish('')
      setSavedEnglish('')
      setSelectedPath(null)
      try {
        const list = await window.api.listImages(dir)
        setFolder(dir)
        setImages(list)
        if (list.length > 0) {
          await loadImage(list[0].path)
        }
        setStatus(`${list.length} image(s)`)
        if (persist) {
          setSettings((prev) => {
            const next = normalizeSettings({ ...prev, lastFolder: dir })
            void window.api.setSettings(next)
            return next
          })
        }
        return true
      } catch {
        setStatus(`Could not open folder: ${dir}`)
        return false
      }
    },
    [cancelInFlight, loadImage]
  )

  useEffect(() => {
    void window.api.getSettings().then(async (s) => {
      const next = normalizeSettings(s)
      setSettings(next)
      if (next.lastFolder) {
        await loadFolder(next.lastFolder, false)
      }
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps -- run once on mount
  }, [])

  useEffect(() => {
    setEnglishSnapshot(english)
  }, [english, setEnglishSnapshot])

  const askUnsaved = useCallback((): Promise<ConfirmAction> => {
    return new Promise((resolve) => {
      unsavedResolver.current = resolve
      setUnsavedOpen(true)
    })
  }, [])

  const resolveUnsaved = (action: ConfirmAction) => {
    setUnsavedOpen(false)
    unsavedResolver.current?.(action)
    unsavedResolver.current = null
  }

  const saveCurrent = useCallback(async () => {
    if (!selectedPath) return false
    try {
      await window.api.writeCaption(selectedPath, english)
      setSavedEnglish(english)
      setImages((prev) =>
        prev.map((img) =>
          img.path === selectedPath ? { ...img, hasCaption: true } : img
        )
      )
      setStatus('Caption saved')
      window.setTimeout(() => setStatus(''), 2000)
      return true
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      return false
    }
  }, [selectedPath, english, setError])

  const ensureCanLeave = useCallback(async (): Promise<boolean> => {
    if (!dirty) return true
    const action = await askUnsaved()
    if (action === 'cancel') return false
    if (action === 'save') {
      return saveCurrent()
    }
    return true
  }, [dirty, askUnsaved, saveCurrent])

  const openFolder = async () => {
    if (!(await ensureCanLeave())) return
    const dir = await window.api.openFolder()
    if (!dir) return
    await loadFolder(dir, true)
  }

  const selectImage = async (path: string) => {
    if (path === selectedPath) return
    if (!(await ensureCanLeave())) return
    await loadImage(path)
  }

  const onEnglishChange = (value: string) => {
    setEnglish(value)
    setEnglishSnapshot(value)
    scheduleEnglishToTarget(value)
  }

  const onTranslatedChange = (value: string) => {
    setTranslated(value)
    scheduleTargetToEnglish(value)
  }

  const onLanguageChange = async (code: string) => {
    const next = normalizeSettings({ ...settings, targetLanguage: code })
    setSettings(next)
    await window.api.setSettings(next)
  }

  const onSaveSettings = async (next: AppSettings) => {
    const normalized = normalizeSettings(next)
    await window.api.setSettings(normalized)
    setSettings(normalized)
  }

  const onAutoSaveSettings = useCallback(async (next: AppSettings) => {
    const normalized = normalizeSettings(next)
    setSettings(normalized)
    await window.api.setSettings(normalized)
  }, [])

  const stopBatchCaption = () => {
    batchCancelRef.current = true
    captionAbortRef.current?.abort()
    captionAbortRef.current = null
  }

  const runCaptionForPath = async (imagePath: string): Promise<void> => {
    captionAbortRef.current?.abort()
    const ac = new AbortController()
    captionAbortRef.current = ac
    const caption = await generateCaptionForImage(settingsRef.current, imagePath, ac.signal)
    await applyCaptionToUi(imagePath, caption)
  }

  const startAutoCaption = async () => {
    const targets = images.filter((img) => !img.hasCaption)
    if (targets.length === 0) {
      setStatus('No images missing captions')
      return
    }
    if (!(await ensureCanLeave())) return

    batchCancelRef.current = false
    setBatchCaptioning(true)
    setError(null)
    let done = 0
    let failed = 0

    try {
      for (const img of targets) {
        if (batchCancelRef.current) break
        setStatus(`Captioning ${done + 1}/${targets.length}: ${img.name}`)
        try {
          await runCaptionForPath(img.path)
          done += 1
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err)
          if (msg === 'Caption cancelled') break
          failed += 1
          setError(`${img.name}: ${msg}`)
        }
      }
      if (batchCancelRef.current) {
        setStatus(`Auto Caption cancelled (${done} done)`)
      } else {
        setStatus(
          failed > 0
            ? `Auto Caption finished: ${done} ok, ${failed} failed`
            : `Auto Caption finished: ${done} image(s)`
        )
      }
    } finally {
      setBatchCaptioning(false)
      captionAbortRef.current = null
      window.setTimeout(() => setStatus(''), 4000)
    }
  }

  const reCaptionCurrent = async () => {
    if (!selectedPath) return
    if (!(await ensureCanLeave())) return
    setSingleCaptioning(true)
    setError(null)
    setStatus(`reCaption: ${selectedPath.split(/[/\\]/).pop()}`)
    try {
      await runCaptionForPath(selectedPath)
      setStatus('reCaption done')
      window.setTimeout(() => setStatus(''), 2000)
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err)
      if (msg !== 'Caption cancelled') setError(msg)
    } finally {
      setSingleCaptioning(false)
      captionAbortRef.current = null
    }
  }

  const clearEditor = useCallback(() => {
    cancelInFlight()
    setSelectedPath(null)
    setEnglish('')
    setSavedEnglish('')
    setTranslated('')
    setEnglishSnapshot('')
    setError(null)
  }, [cancelInFlight, setEnglishSnapshot, setError])

  const confirmDeleteSelected = useCallback(async () => {
    if (!selectedPath) {
      setDeleteOpen(false)
      return
    }
    const pathToDelete = selectedPath
    const name = pathToDelete.split(/[/\\]/).pop() ?? pathToDelete
    const idx = images.findIndex((img) => img.path === pathToDelete)
    const neighbor =
      images[idx + 1]?.path ?? (idx > 0 ? images[idx - 1]?.path : undefined)

    setDeleteOpen(false)
    try {
      await window.api.deleteImage(pathToDelete)
      setImages((prev) => prev.filter((img) => img.path !== pathToDelete))
      if (neighbor) {
        await loadImage(neighbor)
      } else {
        clearEditor()
      }
      setStatus(`Deleted ${name}`)
      window.setTimeout(() => setStatus(''), 2500)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }, [selectedPath, images, loadImage, clearEditor, setError])

  useEffect(() => {
    const isCaptionFocused = () => {
      const el = document.activeElement
      if (!el) return false
      const tag = el.tagName
      if (tag === 'TEXTAREA' || tag === 'INPUT' || tag === 'SELECT') return true
      return el.getAttribute('contenteditable') === 'true'
    }

    const onKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 's') {
        e.preventDefault()
        void saveCurrent()
        return
      }

      if (isCaptionFocused()) return
      if (e.ctrlKey || e.metaKey || e.altKey) return
      if (deleteOpen || unsavedOpen || settingsOpen) return

      // Space/Enter would activate a previously focused control (e.g. list button).
      if (e.key === ' ' || e.key === 'Spacebar' || e.key === 'Enter') {
        e.preventDefault()
        return
      }

      if (e.key === 'Delete') {
        if (!selectedPath) return
        e.preventDefault()
        setDeleteOpen(true)
        return
      }

      const idx = selectedPath ? images.findIndex((img) => img.path === selectedPath) : -1
      if (images.length === 0) return

      if (e.key === 'ArrowDown' || e.key === 'ArrowRight') {
        e.preventDefault()
        const next = idx < 0 ? 0 : Math.min(idx + 1, images.length - 1)
        if (next !== idx) void selectImage(images[next].path)
        return
      }

      if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') {
        e.preventDefault()
        const prev = idx < 0 ? 0 : Math.max(idx - 1, 0)
        if (prev !== idx) void selectImage(images[prev].path)
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [
    saveCurrent,
    images,
    selectedPath,
    selectImage,
    deleteOpen,
    unsavedOpen,
    settingsOpen
  ])

  const imageUrl = selectedPath ? window.api.toLocalUrl(selectedPath) : null
  const captionBusy = batchCaptioning || singleCaptioning

  const persistPaneWidths = useCallback((next: AppSettings) => {
    const normalized = normalizeSettings(next)
    setSettings(normalized)
    void window.api.setSettings(normalized)
  }, [])

  const startPaneResize = (which: 'sidebar' | 'right') => (e: ReactPointerEvent<HTMLButtonElement>) => {
    e.preventDefault()
    const startX = e.clientX
    const startSidebar = settingsRef.current.sidebarWidth
    const startRight = settingsRef.current.rightPaneWidth
    let nextSidebar = startSidebar
    let nextRight = startRight
    const target = e.currentTarget
    target.setPointerCapture(e.pointerId)
    setDraggingPane(which)
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'

    const onMove = (ev: PointerEvent) => {
      const dx = ev.clientX - startX
      if (which === 'sidebar') {
        nextSidebar = clampSidebarWidth(startSidebar + dx)
        setSettings((prev) => ({ ...prev, sidebarWidth: nextSidebar }))
      } else {
        nextRight = clampRightPaneWidth(startRight - dx)
        setSettings((prev) => ({ ...prev, rightPaneWidth: nextRight }))
      }
    }

    const onUp = (ev: PointerEvent) => {
      target.releasePointerCapture(ev.pointerId)
      target.removeEventListener('pointermove', onMove)
      target.removeEventListener('pointerup', onUp)
      target.removeEventListener('pointercancel', onUp)
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
      setDraggingPane(null)
      persistPaneWidths({
        ...settingsRef.current,
        sidebarWidth: nextSidebar,
        rightPaneWidth: nextRight
      })
    }

    target.addEventListener('pointermove', onMove)
    target.addEventListener('pointerup', onUp)
    target.addEventListener('pointercancel', onUp)
  }

  return (
    <div className="app">
      <header className="toolbar">
        <div className="toolbar-left">
          <button type="button" className="primary" onClick={() => void openFolder()}>
            Open folder
          </button>
          {batchCaptioning ? (
            <button type="button" onClick={stopBatchCaption}>
              Cancel Auto Caption
            </button>
          ) : (
            <button
              type="button"
              onClick={() => void startAutoCaption()}
              disabled={!folder || missingCount === 0 || captionBusy}
              title={
                missingCount === 0
                  ? 'All images already have .txt captions'
                  : `Caption ${missingCount} image(s) without .txt`
              }
            >
              Auto Caption{missingCount > 0 ? ` (${missingCount})` : ''}
            </button>
          )}
          <button type="button" onClick={() => setSettingsOpen(true)}>
            Settings
          </button>
          <button
            type="button"
            disabled={!folder || images.length === 0}
            onClick={() => setAnalysisOpen(true)}
          >
            Analyze
          </button>
        </div>
        <div className="toolbar-right">
          {folder && (
            <span className="folder-path" title={folder}>
              {folder}
            </span>
          )}
          {status && <span className="status-msg">{status}</span>}
          {dirty && <span className="dirty-flag">Unsaved</span>}
        </div>
      </header>

      <div
        className="main"
        style={
          {
            '--sidebar-width': `${settings.sidebarWidth}px`,
            '--right-pane-width': `${settings.rightPaneWidth}px`
          } as CSSProperties
        }
      >
        <aside className="sidebar">
          <ImageList
            images={images}
            selectedPath={selectedPath}
            dirtyPaths={dirtyPaths}
            onSelect={(path) => void selectImage(path)}
          />
        </aside>

        <button
          type="button"
          className={`pane-resizer${draggingPane === 'sidebar' ? ' dragging' : ''}`}
          aria-label="Resize image list"
          title="Drag to resize"
          onPointerDown={startPaneResize('sidebar')}
        />

        <section className="center-pane">
          <ImagePreview imagePath={selectedPath} imageUrl={imageUrl} />
        </section>

        <button
          type="button"
          className={`pane-resizer${draggingPane === 'right' ? ' dragging' : ''}`}
          aria-label="Resize caption pane"
          title="Drag to resize"
          onPointerDown={startPaneResize('right')}
        />

        <section className="right-pane">
          <CaptionEditor
            english={english}
            translated={translated}
            targetLanguage={settings.targetLanguage}
            translating={translating}
            captioning={captionBusy}
            error={error}
            canSave={Boolean(selectedPath) && dirty && !captionBusy}
            canRecaption={Boolean(selectedPath) && !captionBusy}
            onEnglishChange={onEnglishChange}
            onTranslatedChange={onTranslatedChange}
            onLanguageChange={(code) => void onLanguageChange(code)}
            onDismissError={() => setError(null)}
            onSave={() => void saveCurrent()}
            onRecaption={() => void reCaptionCurrent()}
          />
        </section>
      </div>

      <SettingsDialog
        open={settingsOpen}
        settings={settings}
        onClose={() => setSettingsOpen(false)}
        onSave={onSaveSettings}
        onAutoSave={onAutoSaveSettings}
      />

      <AnalysisDialog
        open={analysisOpen}
        images={images}
        settings={settings}
        onClose={() => setAnalysisOpen(false)}
      />

      <UnsavedDialog
        open={unsavedOpen}
        onSave={() => resolveUnsaved('save')}
        onDiscard={() => resolveUnsaved('discard')}
        onCancel={() => resolveUnsaved('cancel')}
      />

      <DeleteConfirmDialog
        open={deleteOpen}
        fileName={selectedFileName}
        onConfirm={() => void confirmDeleteSelected()}
        onCancel={() => setDeleteOpen(false)}
      />
    </div>
  )
}
