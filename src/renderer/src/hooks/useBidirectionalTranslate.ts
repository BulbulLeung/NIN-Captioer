import { useCallback, useEffect, useRef, useState } from 'react'
import type { AppSettings } from '../types'
import { translateText } from '../services/translation'

const DEBOUNCE_MS = 700

interface Options {
  settings: AppSettings
  selectedPath: string | null
  setEnglish: (value: string) => void
  setTranslated: (value: string) => void
  enabled: boolean
}

export function useBidirectionalTranslate({
  settings,
  selectedPath,
  setEnglish,
  setTranslated,
  enabled
}: Options) {
  const [translating, setTranslating] = useState(false)
  const [translatingPath, setTranslatingPath] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const requestId = useRef(0)
  const enTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const trTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const settingsRef = useRef(settings)
  settingsRef.current = settings
  const selectedPathRef = useRef(selectedPath)
  selectedPathRef.current = selectedPath

  const clearTimers = useCallback(() => {
    if (enTimer.current) clearTimeout(enTimer.current)
    if (trTimer.current) clearTimeout(trTimer.current)
    enTimer.current = null
    trTimer.current = null
  }, [])

  const cancelInFlight = useCallback(() => {
    clearTimers()
    abortRef.current?.abort()
    abortRef.current = null
    requestId.current += 1
    setTranslating(false)
    setTranslatingPath(null)
  }, [clearTimers])

  const runTranslate = useCallback(
    async (
      text: string,
      direction: 'en-to-target' | 'target-to-en',
      imagePath?: string | null
    ) => {
      // Do not gate on `enabled` — loadImage may translate before selectedPath re-renders.
      abortRef.current?.abort()
      const ac = new AbortController()
      abortRef.current = ac
      const id = ++requestId.current
      const pathForRequest =
        imagePath !== undefined ? imagePath : selectedPathRef.current
      setTranslating(true)
      setTranslatingPath(pathForRequest)
      setError(null)
      try {
        const result = await translateText(settingsRef.current, text, direction, ac.signal)
        if (id !== requestId.current) return
        if (direction === 'en-to-target') {
          setTranslated(result)
        } else {
          setEnglish(result)
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err)
        if (id !== requestId.current) return
        if (message === 'Translation cancelled') return
        setError(message)
      } finally {
        if (id === requestId.current) {
          setTranslating(false)
          setTranslatingPath(null)
        }
      }
    },
    [setEnglish, setTranslated]
  )

  const scheduleEnglishToTarget = useCallback(
    (text: string) => {
      if (!enabled) return
      if (enTimer.current) clearTimeout(enTimer.current)
      const pathAtSchedule = selectedPathRef.current
      enTimer.current = setTimeout(() => {
        void runTranslate(text, 'en-to-target', pathAtSchedule)
      }, DEBOUNCE_MS)
    },
    [enabled, runTranslate]
  )

  const scheduleTargetToEnglish = useCallback(
    (text: string) => {
      if (!enabled) return
      if (trTimer.current) clearTimeout(trTimer.current)
      const pathAtSchedule = selectedPathRef.current
      trTimer.current = setTimeout(() => {
        void runTranslate(text, 'target-to-en', pathAtSchedule)
      }, DEBOUNCE_MS)
    },
    [enabled, runTranslate]
  )

  const translateEnglishToTargetNow = useCallback(
    (text: string, imagePath?: string | null) => {
      clearTimers()
      void runTranslate(text, 'en-to-target', imagePath)
    },
    [clearTimers, runTranslate]
  )

  const langKey = `${settings.targetLanguage}|${settings.provider}|${settings.model}|${settings.lmStudioBaseUrl}|${settings.ollamaBaseUrl}`
  const prevLangKey = useRef<string | null>(null)
  const englishSnapshot = useRef('')

  const setEnglishSnapshot = useCallback((text: string) => {
    englishSnapshot.current = text
  }, [])

  useEffect(() => {
    if (!enabled) return
    if (prevLangKey.current === null) {
      prevLangKey.current = langKey
      return
    }
    if (prevLangKey.current === langKey) return
    prevLangKey.current = langKey
    translateEnglishToTargetNow(englishSnapshot.current)
  }, [langKey, enabled, translateEnglishToTargetNow])

  return {
    translating,
    translatingPath,
    error,
    setError,
    cancelInFlight,
    scheduleEnglishToTarget,
    scheduleTargetToEnglish,
    translateEnglishToTargetNow,
    setEnglishSnapshot
  }
}
