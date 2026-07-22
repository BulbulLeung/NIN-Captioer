import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties, type PointerEvent as ReactPointerEvent } from 'react'
import { AnalysisDialog } from './components/AnalysisDialog'
import { CaptionEditor } from './components/CaptionEditor'
import { DeleteConfirmDialog } from './components/DeleteConfirmDialog'
import { ImageList } from './components/ImageList'
import { ImagePreview } from './components/ImagePreview'
import { MissingDatasetFolderDialog } from './components/MissingDatasetFolderDialog'
import { RemoveDatasetFolderDialog } from './components/RemoveDatasetFolderDialog'
import { SettingsDialog } from './components/SettingsDialog'
import { UnsavedDialog } from './components/UnsavedDialog'
import { useBidirectionalTranslate } from './hooks/useBidirectionalTranslate'
import { useCaptionAnalysis } from './hooks/useCaptionAnalysis'
import { generateCaptionForImage } from './services/captioning'
import type { AppSettings, ImageItem, ListViewMode } from './types'
import {
  clampRightPaneWidth,
  clampSidebarWidth,
  clampThumbnailWidth,
  normalizeSettings
} from './types'

type ConfirmAction = 'save' | 'discard' | 'cancel'

function folderLabel(dir: string): string {
  return dir.split(/[/\\]/).pop() ?? dir
}

function getThumbnailColumns(): number {
  const items = document.querySelectorAll('.image-list.thumbnails > li')
  if (items.length === 0) return 1
  const firstTop = (items[0] as HTMLElement).offsetTop
  let cols = 0
  for (const el of items) {
    if ((el as HTMLElement).offsetTop !== firstTop) break
    cols++
  }
  return Math.max(1, cols)
}

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
  const [removeDatasetOpen, setRemoveDatasetOpen] = useState(false)
  const [missingDatasetOpen, setMissingDatasetOpen] = useState(false)
  const [missingDatasetPath, setMissingDatasetPath] = useState<string | null>(null)
  const [datasetMenuOpen, setDatasetMenuOpen] = useState(false)
  const [status, setStatus] = useState('')
  const [batchCaptioning, setBatchCaptioning] = useState(false)
  const [singleCaptioning, setSingleCaptioning] = useState(false)
  const [captioningPath, setCaptioningPath] = useState<string | null>(null)
  const [draggingPane, setDraggingPane] = useState<'sidebar' | 'right' | null>(null)
  const unsavedResolver = useRef<((action: ConfirmAction) => void) | null>(null)
  const captionAbortRef = useRef<AbortController | null>(null)
  const batchCancelRef = useRef(false)
  const datasetMenuRef = useRef<HTMLDivElement | null>(null)
  const settingsRef = useRef(settings)
  settingsRef.current = settings
  const selectedPathRef = useRef(selectedPath)
  selectedPathRef.current = selectedPath

  const dirty = english !== savedEnglish
  const captionBusy = batchCaptioning || singleCaptioning
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
    translatingPath,
    error,
    setError,
    cancelInFlight,
    scheduleEnglishToTarget,
    scheduleTargetToEnglish,
    translateEnglishToTargetNow,
    setEnglishSnapshot
  } = useBidirectionalTranslate({
    settings,
    selectedPath,
    setEnglish: (value) => {
      setEnglish(value)
      setEnglishSnapshot(value)
    },
    setTranslated,
    enabled: Boolean(selectedPath)
  })

  const busyPaths = useMemo(() => {
    const s = new Set<string>()
    if (captioningPath) s.add(captioningPath)
    if (translating && translatingPath) s.add(translatingPath)
    return s
  }, [captioningPath, translating, translatingPath])

  const aiBusy = captionBusy || translating
  const analysisEnabled = settings.autoAnalysis || analysisOpen
  const {
    result: analysisResult,
    analyzing: analysisAnalyzing,
    progress: analysisProgress,
    error: analysisError,
    invalidate: invalidateAnalysis
  } = useCaptionAnalysis(images, settings, aiBusy, analysisEnabled)

  const applyCaptionToUi = useCallback(
    async (imagePath: string, caption: string) => {
      await window.api.writeCaption(imagePath, caption)
      setImages((prev) =>
        prev.map((img) => (img.path === imagePath ? { ...img, hasCaption: true } : img))
      )
      invalidateAnalysis(imagePath, caption)
      if (selectedPathRef.current === imagePath) {
        setEnglish(caption)
        setSavedEnglish(caption)
        setEnglishSnapshot(caption)
        setTranslated('')
        if (caption.trim()) translateEnglishToTargetNow(caption, imagePath)
      }
    },
    [invalidateAnalysis, setEnglishSnapshot, translateEnglishToTargetNow]
  )

  const loadImage = useCallback(
    async (imagePath: string) => {
      cancelInFlight()
      const caption = await window.api.readCaption(imagePath)
      selectedPathRef.current = imagePath
      setSelectedPath(imagePath)
      setEnglish(caption)
      setSavedEnglish(caption)
      setTranslated('')
      setEnglishSnapshot(caption)
      setError(null)
      if (caption.trim()) {
        translateEnglishToTargetNow(caption, imagePath)
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
        setStatus('')
        if (persist) {
          setSettings((prev) => {
            const folders = prev.datasetFolders.includes(dir)
              ? prev.datasetFolders
              : [...prev.datasetFolders, dir]
            const next = normalizeSettings({
              ...prev,
              lastFolder: dir,
              datasetFolders: folders
            })
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

  useEffect(() => {
    if (!datasetMenuOpen) return
    const onPointerDown = (e: PointerEvent) => {
      const root = datasetMenuRef.current
      if (!root) return
      if (e.target instanceof Node && root.contains(e.target)) return
      setDatasetMenuOpen(false)
    }
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setDatasetMenuOpen(false)
    }
    window.addEventListener('pointerdown', onPointerDown, true)
    window.addEventListener('keydown', onKeyDown, true)
    return () => {
      window.removeEventListener('pointerdown', onPointerDown, true)
      window.removeEventListener('keydown', onKeyDown, true)
    }
  }, [datasetMenuOpen])

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
      invalidateAnalysis(selectedPath, english)
      setStatus('Caption saved')
      window.setTimeout(() => setStatus(''), 2000)
      return true
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      return false
    }
  }, [selectedPath, english, invalidateAnalysis, setError])

  const ensureCanLeave = useCallback(async (): Promise<boolean> => {
    if (!dirty) return true
    const action = await askUnsaved()
    if (action === 'cancel') return false
    if (action === 'save') {
      return saveCurrent()
    }
    return true
  }, [dirty, askUnsaved, saveCurrent])

  const addDatasetFolder = async () => {
    if (!(await ensureCanLeave())) return
    const dir = await window.api.openFolder()
    if (!dir) return
    await loadFolder(dir, true)
  }

  const switchDatasetFolder = async (dir: string) => {
    setDatasetMenuOpen(false)
    if (!dir || dir === folder) return
    if (!(await ensureCanLeave())) return
    const ok = await loadFolder(dir, true)
    if (!ok) {
      setMissingDatasetPath(dir)
      setMissingDatasetOpen(true)
    }
  }

  const cancelMissingDatasetFolder = () => {
    setMissingDatasetOpen(false)
    setMissingDatasetPath(null)
  }

  const confirmDeleteMissingDatasetFolder = () => {
    const removing = missingDatasetPath
    setMissingDatasetOpen(false)
    setMissingDatasetPath(null)
    if (!removing) return

    setSettings((prev) => {
      const remaining = prev.datasetFolders.filter((f) => f !== removing)
      const next = normalizeSettings({
        ...prev,
        datasetFolders: remaining,
        lastFolder: prev.lastFolder === removing ? (remaining[0] ?? null) : prev.lastFolder
      })
      void window.api.setSettings(next)
      return next
    })
  }

  const requestRemoveDatasetFolder = () => {
    if (!folder) return
    setRemoveDatasetOpen(true)
  }

  const confirmRemoveDatasetFolder = async () => {
    setRemoveDatasetOpen(false)
    if (!folder) return
    if (!(await ensureCanLeave())) return
    const removing = folder
    const remaining = settingsRef.current.datasetFolders.filter((f) => f !== removing)
    const nextFolder = remaining[0] ?? null

    cancelInFlight()
    setTranslated('')
    setEnglish('')
    setSavedEnglish('')
    setSelectedPath(null)

    setSettings((prev) => {
      const next = normalizeSettings({
        ...prev,
        datasetFolders: remaining,
        lastFolder: nextFolder
      })
      void window.api.setSettings(next)
      return next
    })

    if (nextFolder) {
      await loadFolder(nextFolder, false)
    } else {
      setFolder(null)
      setImages([])
    }
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
    setCaptioningPath(imagePath)
    try {
      const caption = await generateCaptionForImage(settingsRef.current, imagePath, ac.signal)
      await applyCaptionToUi(imagePath, caption)
    } finally {
      setCaptioningPath(null)
    }
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
      if (msg !== 'Caption cancelled') {
        setStatus('')
        setError(msg)
      }
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
      if (tag === 'TEXTAREA' || tag === 'SELECT') return true
      if (tag === 'INPUT') {
        const type = (el as HTMLInputElement).type
        return type !== 'range'
      }
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
      if (deleteOpen || unsavedOpen || settingsOpen || removeDatasetOpen || missingDatasetOpen) return

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
      if (
        e.key !== 'ArrowDown' &&
        e.key !== 'ArrowRight' &&
        e.key !== 'ArrowUp' &&
        e.key !== 'ArrowLeft'
      ) {
        return
      }

      e.preventDefault()
      const active = document.activeElement
      if (active instanceof HTMLElement && active.classList.contains('list-toolbar-slider')) {
        active.blur()
      }
      if (idx < 0) {
        void selectImage(images[0].path)
        return
      }

      let next = idx

      if (settingsRef.current.listViewMode === 'thumbnails') {
        const cols = getThumbnailColumns()
        if (e.key === 'ArrowLeft') next = Math.max(0, idx - 1)
        else if (e.key === 'ArrowRight') next = Math.min(images.length - 1, idx + 1)
        else if (e.key === 'ArrowUp') next = idx >= cols ? idx - cols : idx
        else if (e.key === 'ArrowDown') next = idx + cols < images.length ? idx + cols : idx
      } else if (e.key === 'ArrowDown' || e.key === 'ArrowRight') {
        next = Math.min(idx + 1, images.length - 1)
      } else {
        next = Math.max(idx - 1, 0)
      }

      if (next !== idx) void selectImage(images[next].path)
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
    settingsOpen,
    removeDatasetOpen,
    missingDatasetOpen
  ])

  const imageUrl = selectedPath ? window.api.toLocalUrl(selectedPath) : null
  const toolbarStatus =
    status ||
    error ||
    (translating
      ? 'Translating…'
      : analysisAnalyzing
        ? analysisProgress
          ? `Analyzing ${analysisProgress.done}/${analysisProgress.total}…`
          : 'Analyzing…'
        : '')
  const toolbarStatusIsError = Boolean(error && !status)

  const persistPaneWidths = useCallback((next: AppSettings) => {
    const normalized = normalizeSettings(next)
    setSettings(normalized)
    void window.api.setSettings(normalized)
  }, [])

  const persistListView = useCallback(
    (patch: Partial<Pick<AppSettings, 'listViewMode' | 'thumbnailWidth'>>) => {
      const next = normalizeSettings({ ...settingsRef.current, ...patch })
      setSettings(next)
      void window.api.setSettings(next)
    },
    []
  )

  const setListViewMode = (mode: ListViewMode) => {
    if (settings.listViewMode === mode) return
    persistListView({ listViewMode: mode })
  }

  const onThumbnailWidthChange = (value: number) => {
    const thumbnailWidth = clampThumbnailWidth(value)
    setSettings((prev) => ({ ...prev, thumbnailWidth }))
  }

  const onThumbnailWidthCommit = (value: number) => {
    persistListView({ thumbnailWidth: clampThumbnailWidth(value) })
  }

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
      <header
        className="toolbar"
        onMouseDown={(e) => {
          const target = e.target
          if (!(target instanceof Element)) return
          if (target.closest('button')) e.preventDefault()
        }}
      >
        <div className="toolbar-left">
          <div className="toolbar-dataset" ref={datasetMenuRef}>
            <button
              type="button"
              className="toolbar-dataset-trigger"
              disabled={settings.datasetFolders.length === 0}
              title={folder ?? 'No dataset folder'}
              aria-label="Dataset folder"
              aria-haspopup="listbox"
              aria-expanded={datasetMenuOpen}
              onClick={() => setDatasetMenuOpen((open) => !open)}
            >
              <span className="toolbar-dataset-label">
                {folder ? folderLabel(folder) : 'No dataset folder'}
              </span>
              <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true">
                <path fill="currentColor" d="M2 3.5h6L5 7.5 2 3.5z" />
              </svg>
            </button>
            {datasetMenuOpen && settings.datasetFolders.length > 0 && (
              <ul className="toolbar-dataset-menu" role="listbox" aria-label="Dataset folders">
                {settings.datasetFolders.map((dir) => (
                  <li key={dir} role="presentation">
                    <button
                      type="button"
                      role="option"
                      className={
                        dir === folder
                          ? 'toolbar-dataset-option active'
                          : 'toolbar-dataset-option'
                      }
                      aria-selected={dir === folder}
                      title={dir}
                      onClick={() => void switchDatasetFolder(dir)}
                    >
                      {folderLabel(dir)}
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
          <button
            type="button"
            className="primary toolbar-icon-btn"
            title="Add Dataset Folder"
            aria-label="Add Dataset Folder"
            onClick={() => void addDatasetFolder()}
          >
            <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
              <path
                fill="currentColor"
                d="M6.25 1.5h1.5v4.75H12.5v1.5H7.75V12.5h-1.5V7.75H1.5v-1.5h4.75V1.5z"
              />
            </svg>
          </button>
          <button
            type="button"
            className="toolbar-icon-btn danger"
            disabled={!folder}
            title="Remove dataset folder"
            aria-label="Remove dataset folder"
            onClick={requestRemoveDatasetFolder}
          >
            <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
              <path
                fill="currentColor"
                d="M5 1.5h4l.5 1H12v1.5H2V2.5h2.5L5 1.5zM3.5 5h7l-.6 7.2A1 1 0 0 1 8.9 13H5.1a1 1 0 0 1-1-.8L3.5 5zm2 1.5v5h1.25v-5H5.5zm2.75 0v5H9.5v-5H8.25z"
              />
            </svg>
          </button>
          <button type="button" onClick={() => setSettingsOpen(true)}>
            Settings
          </button>
          <button
            type="button"
            className={analysisAnalyzing ? 'analyze-btn is-analyzing' : 'analyze-btn'}
            disabled={!folder || images.length === 0}
            onClick={() => setAnalysisOpen(true)}
          >
            <span className="analyze-btn-label">Analyze</span>
          </button>
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
          <div className="list-toolbar">
            <div className="list-toolbar-left">
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
            </div>
            <div className="list-toolbar-right">
              <button
                type="button"
                className={`list-toolbar-btn${settings.listViewMode === 'list' ? ' active' : ''}`}
                aria-pressed={settings.listViewMode === 'list'}
                title="List view"
                onClick={() => setListViewMode('list')}
              >
                <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
                  <path
                    fill="currentColor"
                    d="M1 2.5h12v1.5H1V2.5zm0 4h12V8H1V6.5zm0 4h12V12H1v-1.5z"
                  />
                </svg>
              </button>
              <button
                type="button"
                className={`list-toolbar-btn${settings.listViewMode === 'thumbnails' ? ' active' : ''}`}
                aria-pressed={settings.listViewMode === 'thumbnails'}
                title="Thumbnail view"
                onClick={() => setListViewMode('thumbnails')}
              >
                <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
                  <path
                    fill="currentColor"
                    d="M1 1h5v5H1V1zm7 0h5v5H8V1zM1 8h5v5H1V8zm7 0h5v5H8V8z"
                  />
                </svg>
              </button>
              <input
                type="range"
                className="list-toolbar-slider"
                min={48}
                max={160}
                step={4}
                value={settings.thumbnailWidth}
                disabled={settings.listViewMode !== 'thumbnails'}
                title="Thumbnail width"
                aria-label="Thumbnail width"
                onChange={(e) => onThumbnailWidthChange(Number(e.target.value))}
                onPointerUp={(e) => {
                  onThumbnailWidthCommit(Number(e.currentTarget.value))
                  e.currentTarget.blur()
                }}
                onKeyUp={(e) => {
                  onThumbnailWidthCommit(Number(e.currentTarget.value))
                  e.currentTarget.blur()
                }}
              />
            </div>
          </div>
          <div className="sidebar-list">
            <ImageList
              images={images}
              selectedPath={selectedPath}
              dirtyPaths={dirtyPaths}
              busyPaths={busyPaths}
              viewMode={settings.listViewMode}
              thumbnailWidth={settings.thumbnailWidth}
              onSelect={(path) => void selectImage(path)}
            />
          </div>
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
            translating={Boolean(translating && translatingPath === selectedPath)}
            captioning={captioningPath !== null && captioningPath === selectedPath}
            canSave={Boolean(selectedPath) && dirty && !captionBusy}
            canRecaption={Boolean(selectedPath) && !captionBusy}
            onEnglishChange={onEnglishChange}
            onTranslatedChange={onTranslatedChange}
            onLanguageChange={(code) => void onLanguageChange(code)}
            onSave={() => void saveCurrent()}
            onRecaption={() => void reCaptionCurrent()}
          />
        </section>
      </div>

      <footer className="system-bar">
        <div className="system-bar-left">
          {folder && (
            <span className="folder-path" title={folder}>
              {folder}
            </span>
          )}
          {folder && <span className="image-count">{images.length} image(s)</span>}
        </div>
        <div className="system-bar-right">
          {toolbarStatus && (
            <span className={`status-msg${toolbarStatusIsError ? ' is-error' : ''}`}>
              {toolbarStatus}
            </span>
          )}
          {dirty && <span className="dirty-flag">Unsaved</span>}
        </div>
      </footer>

      <SettingsDialog
        open={settingsOpen}
        settings={settings}
        onClose={() => setSettingsOpen(false)}
        onSave={onSaveSettings}
        onAutoSave={onAutoSaveSettings}
      />

      <AnalysisDialog
        open={analysisOpen}
        imageCount={images.length}
        analyzing={analysisAnalyzing}
        progress={analysisProgress}
        error={analysisError}
        result={analysisResult}
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

      <RemoveDatasetFolderDialog
        open={removeDatasetOpen}
        folderName={folder ? folderLabel(folder) : ''}
        onConfirm={() => void confirmRemoveDatasetFolder()}
        onCancel={() => setRemoveDatasetOpen(false)}
      />

      <MissingDatasetFolderDialog
        open={missingDatasetOpen}
        folderName={missingDatasetPath ? folderLabel(missingDatasetPath) : ''}
        onConfirm={confirmDeleteMissingDatasetFolder}
        onCancel={cancelMissingDatasetFolder}
      />
    </div>
  )
}
