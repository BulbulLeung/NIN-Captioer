import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { AppSettings, ImageItem } from '../types'
import {
  emptyClassification,
  type CaptionAnalysisResult,
  type CaptionClassification
} from '../services/captionAnalysis'
import { classifyCaption } from '../services/captionClassify'
import { buildCaptionAnalysisResult } from '../services/loraHealthScore'

/** Wait after interactive AI finishes before resuming analysis. */
const IDLE_RESUME_MS = 800

interface CacheEntry {
  captionText: string
  classification: CaptionClassification
  /** True when classification matches captionText (incl. empty captions). */
  fresh: boolean
}

export function useCaptionAnalysis(
  images: ImageItem[],
  settings: AppSettings,
  aiBusy: boolean,
  analysisEnabled: boolean
) {
  const [result, setResult] = useState<CaptionAnalysisResult | null>(null)
  const [analyzing, setAnalyzing] = useState(false)
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null)
  const [error, setError] = useState<string | null>(null)

  const cacheRef = useRef<Map<string, CacheEntry>>(new Map())
  const queueRef = useRef<string[]>([])
  const abortRef = useRef<AbortController | null>(null)
  const runningRef = useRef(false)
  const syncGenRef = useRef(0)
  const resumeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const imagesRef = useRef(images)
  const settingsRef = useRef(settings)
  const aiBusyRef = useRef(aiBusy)
  const analysisEnabledRef = useRef(analysisEnabled)
  imagesRef.current = images
  settingsRef.current = settings
  aiBusyRef.current = aiBusy
  analysisEnabledRef.current = analysisEnabled

  const imagePathsKey = useMemo(
    () => images.map((img) => img.path).join('\0'),
    [images]
  )

  const clearResumeTimer = useCallback(() => {
    if (resumeTimerRef.current) {
      clearTimeout(resumeTimerRef.current)
      resumeTimerRef.current = null
    }
  }, [])

  const rebuildResult = useCallback(() => {
    const imgs = imagesRef.current
    if (imgs.length === 0) {
      setResult(null)
      return
    }
    const captions: string[] = []
    const classifications: CaptionClassification[] = []
    for (const img of imgs) {
      const entry = cacheRef.current.get(img.path)
      captions.push(entry?.captionText ?? '')
      classifications.push(entry?.classification ?? emptyClassification())
    }
    setResult(buildCaptionAnalysisResult(captions, classifications))
  }, [])

  const enqueue = useCallback((paths: string[]) => {
    const q = queueRef.current
    for (const p of paths) {
      if (!q.includes(p)) q.push(p)
    }
  }, [])

  const updateProgress = useCallback(() => {
    const nonEmpty = imagesRef.current
      .map((img) => img.path)
      .filter((p) => {
        const e = cacheRef.current.get(p)
        return Boolean(e?.captionText.trim())
      })
    const total = nonEmpty.length
    if (total === 0) {
      setProgress(null)
      return
    }
    const done = nonEmpty.filter((p) => cacheRef.current.get(p)?.fresh).length
    setProgress({ done, total })
  }, [])

  const pump = useCallback(async () => {
    if (runningRef.current) return
    if (!analysisEnabledRef.current) return
    if (aiBusyRef.current) return
    runningRef.current = true

    try {
      while (true) {
        if (!analysisEnabledRef.current || aiBusyRef.current) {
          setAnalyzing(false)
          setProgress(null)
          break
        }

        if (!settingsRef.current.model.trim()) {
          setAnalyzing(false)
          setProgress(null)
          if (imagesRef.current.length > 0) {
            setError('Model is required in Settings before analyzing captions.')
          }
          break
        }

        const pathSet = new Set(imagesRef.current.map((img) => img.path))
        queueRef.current = queueRef.current.filter((p) => {
          if (!pathSet.has(p)) return false
          const e = cacheRef.current.get(p)
          return Boolean(e && !e.fresh && e.captionText.trim())
        })

        if (queueRef.current.length === 0) {
          setAnalyzing(false)
          setProgress(null)
          break
        }

        setAnalyzing(true)
        setError(null)
        updateProgress()

        const path = queueRef.current.shift()!
        const entry = cacheRef.current.get(path)
        if (!entry || entry.fresh || !entry.captionText.trim()) continue

        const abort = new AbortController()
        abortRef.current = abort
        try {
          const classification = await classifyCaption(
            settingsRef.current,
            entry.captionText,
            abort.signal
          )
          const current = cacheRef.current.get(path)
          if (current && current.captionText === entry.captionText) {
            cacheRef.current.set(path, {
              captionText: entry.captionText,
              classification,
              fresh: true
            })
            rebuildResult()
            updateProgress()
          }
        } catch (err) {
          if (abort.signal.aborted) {
            const current = cacheRef.current.get(path)
            if (current && !current.fresh && current.captionText.trim()) {
              enqueue([path])
            }
            break
          }
          setError(err instanceof Error ? err.message : String(err))
          setAnalyzing(false)
          setProgress(null)
          break
        } finally {
          if (abortRef.current === abort) abortRef.current = null
        }
      }
    } finally {
      runningRef.current = false
      if (
        analysisEnabledRef.current &&
        !aiBusyRef.current &&
        queueRef.current.length > 0 &&
        settingsRef.current.model.trim()
      ) {
        void pump()
      }
    }
  }, [enqueue, rebuildResult, updateProgress])

  /** Immediate pump when interactive AI is already idle and analysis is enabled. */
  const tryPumpSoon = useCallback(() => {
    if (!analysisEnabledRef.current || aiBusyRef.current) return
    void pump()
  }, [pump])

  /** After interactive AI finishes, wait before resuming analysis. */
  const schedulePumpAfterIdle = useCallback(() => {
    clearResumeTimer()
    if (!analysisEnabledRef.current || aiBusyRef.current) return
    resumeTimerRef.current = setTimeout(() => {
      resumeTimerRef.current = null
      if (analysisEnabledRef.current && !aiBusyRef.current) void pump()
    }, IDLE_RESUME_MS)
  }, [clearResumeTimer, pump])

  // Sync cache when image set changes (folder open / delete).
  useEffect(() => {
    const paths = imagePathsKey ? imagePathsKey.split('\0') : []
    const pathSet = new Set(paths)

    for (const key of [...cacheRef.current.keys()]) {
      if (!pathSet.has(key)) cacheRef.current.delete(key)
    }
    queueRef.current = queueRef.current.filter((p) => pathSet.has(p))

    if (paths.length === 0) {
      clearResumeTimer()
      abortRef.current?.abort('cancel')
      abortRef.current = null
      setResult(null)
      setAnalyzing(false)
      setProgress(null)
      setError(null)
      return
    }

    const gen = ++syncGenRef.current
    void (async () => {
      const captions = await Promise.all(paths.map((p) => window.api.readCaption(p)))
      if (gen !== syncGenRef.current) return

      const toQueue: string[] = []
      paths.forEach((path, i) => {
        const text = captions[i]
        const existing = cacheRef.current.get(path)
        if (existing && existing.captionText === text && existing.fresh) {
          return
        }
        if (!text.trim()) {
          cacheRef.current.set(path, {
            captionText: text,
            classification: emptyClassification(),
            fresh: true
          })
          return
        }
        if (existing && existing.captionText === text && !existing.fresh) {
          toQueue.push(path)
          return
        }
        cacheRef.current.set(path, {
          captionText: text,
          classification: emptyClassification(),
          fresh: false
        })
        toQueue.push(path)
      })

      rebuildResult()
      enqueue(toQueue)
      tryPumpSoon()
    })()
  }, [imagePathsKey, clearResumeTimer, enqueue, rebuildResult, tryPumpSoon])

  // Gate: pause when analysis disabled (e.g. dialog closed with autoAnalysis off).
  useEffect(() => {
    if (!analysisEnabled) {
      clearResumeTimer()
      abortRef.current?.abort('cancel')
      setAnalyzing(false)
      setProgress(null)
      return
    }
    tryPumpSoon()
  }, [analysisEnabled, clearResumeTimer, tryPumpSoon])

  // Yield to interactive AI; resume after idle cooldown.
  useEffect(() => {
    if (!analysisEnabled) return
    if (aiBusy) {
      clearResumeTimer()
      abortRef.current?.abort('cancel')
      setAnalyzing(false)
      setProgress(null)
      return
    }
    schedulePumpAfterIdle()
    return () => clearResumeTimer()
  }, [aiBusy, analysisEnabled, clearResumeTimer, schedulePumpAfterIdle])

  // Resume when model becomes available.
  useEffect(() => {
    if (!analysisEnabled || !settings.model.trim() || aiBusy) return
    const stale: string[] = []
    for (const img of imagesRef.current) {
      const e = cacheRef.current.get(img.path)
      if (e && !e.fresh && e.captionText.trim()) stale.push(img.path)
    }
    enqueue(stale)
    tryPumpSoon()
  }, [settings.model, aiBusy, analysisEnabled, enqueue, tryPumpSoon])

  useEffect(() => {
    return () => clearResumeTimer()
  }, [clearResumeTimer])

  const invalidate = useCallback(
    (imagePath: string, newCaption: string) => {
      const existing = cacheRef.current.get(imagePath)
      if (existing && existing.captionText === newCaption && existing.fresh) {
        return
      }

      if (!newCaption.trim()) {
        cacheRef.current.set(imagePath, {
          captionText: newCaption,
          classification: emptyClassification(),
          fresh: true
        })
        queueRef.current = queueRef.current.filter((p) => p !== imagePath)
        rebuildResult()
        return
      }

      cacheRef.current.set(imagePath, {
        captionText: newCaption,
        classification: emptyClassification(),
        fresh: false
      })
      rebuildResult()
      enqueue([imagePath])
      tryPumpSoon()
    },
    [enqueue, rebuildResult, tryPumpSoon]
  )

  return { result, analyzing, progress, error, invalidate }
}
