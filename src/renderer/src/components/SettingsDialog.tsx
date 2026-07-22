import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { AppSettings, CaptionPreset, TranslationProvider } from '../types'
import { DEFAULT_SETTINGS, normalizeSettings } from '../types'
import { createDefaultCaptionPreset } from '../defaults/captionPresets'
import { listModels, testConnection } from '../services/translation'

interface Props {
  open: boolean
  settings: AppSettings
  onClose: () => void
  onSave: (settings: AppSettings) => Promise<void>
  /** Persist settings immediately (used for caption prompt auto-save). */
  onAutoSave: (settings: AppSettings) => Promise<void>
}

function newPresetId(): string {
  return `preset-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`
}

export function SettingsDialog({ open, settings, onClose, onSave, onAutoSave }: Props) {
  const [draft, setDraft] = useState<AppSettings>(settings)
  const [models, setModels] = useState<string[]>([])
  const [loadingModels, setLoadingModels] = useState(false)
  const [modelsError, setModelsError] = useState<string | null>(null)
  const [testing, setTesting] = useState(false)
  const [testResult, setTestResult] = useState<string | null>(null)
  const [testError, setTestError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const fetchId = useRef(0)
  const urlDebounce = useRef<ReturnType<typeof setTimeout> | null>(null)
  const promptDebounce = useRef<ReturnType<typeof setTimeout> | null>(null)
  const skipUrlFetch = useRef(false)
  const skipPromptSave = useRef(false)
  const draftRef = useRef(draft)
  draftRef.current = draft

  const baseUrl =
    draft.provider === 'lmstudio' ? draft.lmStudioBaseUrl : draft.ollamaBaseUrl

  const activePreset = useMemo(() => {
    return (
      draft.captionPresets.find((p) => p.id === draft.activeCaptionPresetId) ??
      draft.captionPresets[0] ??
      null
    )
  }, [draft.captionPresets, draft.activeCaptionPresetId])

  const fetchModels = useCallback(async (next: AppSettings) => {
    const id = ++fetchId.current
    setLoadingModels(true)
    setModelsError(null)
    try {
      const list = await listModels(next)
      if (id !== fetchId.current) return
      const sorted = [...list].sort((a, b) => {
        const score = (name: string) => {
          const m = name.toLowerCase().match(/(\d+(?:\.\d+)?)b/)
          return m ? parseFloat(m[1]) : 999
        }
        return score(a) - score(b) || a.localeCompare(b)
      })
      setModels(sorted)
      if (list.length > 0) {
        setDraft((prev) => {
          if (!prev.model) return { ...prev, model: list[0] }
          return prev
        })
      }
    } catch (err) {
      if (id !== fetchId.current) return
      setModels([])
      setModelsError(err instanceof Error ? err.message : String(err))
    } finally {
      if (id === fetchId.current) setLoadingModels(false)
    }
  }, [])

  useEffect(() => {
    if (!open) return
    skipUrlFetch.current = true
    skipPromptSave.current = true
    setDraft(normalizeSettings(settings))
    setTestResult(null)
    setTestError(null)
    setModelsError(null)
    void fetchModels(normalizeSettings(settings))
  }, [open, settings, fetchModels])

  useEffect(() => {
    if (!open) return
    if (skipUrlFetch.current) {
      skipUrlFetch.current = false
      return
    }
    if (urlDebounce.current) clearTimeout(urlDebounce.current)
    urlDebounce.current = setTimeout(() => {
      void fetchModels(draftRef.current)
    }, 500)
    return () => {
      if (urlDebounce.current) clearTimeout(urlDebounce.current)
    }
  }, [open, draft.provider, baseUrl, fetchModels])

  // Auto-save caption presets / active id while settings stay open
  useEffect(() => {
    if (!open) return
    if (skipPromptSave.current) {
      skipPromptSave.current = false
      return
    }
    if (promptDebounce.current) clearTimeout(promptDebounce.current)
    promptDebounce.current = setTimeout(() => {
      const next = normalizeSettings(draftRef.current)
      void onAutoSave(next)
    }, 500)
    return () => {
      if (promptDebounce.current) clearTimeout(promptDebounce.current)
    }
  }, [open, draft.captionPresets, draft.activeCaptionPresetId, onAutoSave])

  const modelOptions = useMemo(() => {
    if (draft.model && !models.includes(draft.model)) return [draft.model, ...models]
    return models
  }, [draft.model, models])

  if (!open) return null

  const setProvider = (provider: TranslationProvider) => {
    setDraft((prev) => ({ ...prev, provider }))
    setTestResult(null)
    setTestError(null)
  }

  const updateActivePreset = (patch: Partial<CaptionPreset>) => {
    setDraft((prev) => ({
      ...prev,
      captionPresets: prev.captionPresets.map((p) =>
        p.id === prev.activeCaptionPresetId ? { ...p, ...patch } : p
      )
    }))
  }

  const addPreset = () => {
    const preset: CaptionPreset = {
      id: newPresetId(),
      name: `Preset ${draft.captionPresets.length + 1}`,
      prompt: activePreset?.prompt ?? createDefaultCaptionPreset().prompt
    }
    setDraft((prev) => ({
      ...prev,
      captionPresets: [...prev.captionPresets, preset],
      activeCaptionPresetId: preset.id
    }))
  }

  const deletePreset = () => {
    setDraft((prev) => {
      if (prev.captionPresets.length <= 1) {
        const fallback = createDefaultCaptionPreset()
        return {
          ...prev,
          captionPresets: [fallback],
          activeCaptionPresetId: fallback.id
        }
      }
      const remaining = prev.captionPresets.filter((p) => p.id !== prev.activeCaptionPresetId)
      return {
        ...prev,
        captionPresets: remaining,
        activeCaptionPresetId: remaining[0].id
      }
    })
  }

  const handleTest = async () => {
    setTesting(true)
    setTestResult(null)
    setTestError(null)
    try {
      const msg = await testConnection(draft)
      setTestResult(msg)
      const list = await listModels(draft)
      setModels(list)
      setModelsError(null)
    } catch (err) {
      setTestError(err instanceof Error ? err.message : String(err))
    } finally {
      setTesting(false)
    }
  }

  const handleSave = async () => {
    setSaving(true)
    try {
      await onSave(normalizeSettings(draft))
      onClose()
    } finally {
      setSaving(false)
    }
  }

  const resetDefaults = () => {
    setDraft({
      ...DEFAULT_SETTINGS,
      targetLanguage: draft.targetLanguage,
      model: draft.model,
      lastFolder: draft.lastFolder,
      datasetFolders: draft.datasetFolders,
      captionPresets: draft.captionPresets,
      activeCaptionPresetId: draft.activeCaptionPresetId,
      sidebarWidth: draft.sidebarWidth,
      rightPaneWidth: draft.rightPaneWidth,
      listViewMode: draft.listViewMode,
      thumbnailWidth: draft.thumbnailWidth
    })
  }

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal modal-wide"
        role="dialog"
        aria-labelledby="settings-title"
        onClick={(e) => e.stopPropagation()}
      >
        <h2 id="settings-title">Settings</h2>

        <label className="field">
          <span>Translation provider</span>
          <div className="radio-row">
            <label>
              <input
                type="radio"
                name="provider"
                checked={draft.provider === 'lmstudio'}
                onChange={() => setProvider('lmstudio')}
              />
              LM Studio
            </label>
            <label>
              <input
                type="radio"
                name="provider"
                checked={draft.provider === 'ollama'}
                onChange={() => setProvider('ollama')}
              />
              Ollama
            </label>
          </div>
        </label>

        {draft.provider === 'lmstudio' ? (
          <label className="field">
            <span>LM Studio Base URL</span>
            <input
              type="text"
              value={draft.lmStudioBaseUrl}
              onChange={(e) =>
                setDraft((prev) => ({ ...prev, lmStudioBaseUrl: e.target.value }))
              }
            />
          </label>
        ) : (
          <>
            <label className="field">
              <span>Ollama Base URL</span>
              <input
                type="text"
                value={draft.ollamaBaseUrl}
                onChange={(e) =>
                  setDraft((prev) => ({ ...prev, ollamaBaseUrl: e.target.value }))
                }
              />
            </label>
            <p className="field-hint">
              Interactive translation / captioning has priority; analysis pauses until idle.
              Optional: <code>OLLAMA_NUM_PARALLEL=2</code> for hard concurrency (restart
              Ollama).
            </p>
          </>
        )}

        <div className="field">
          <span>Model</span>
          <div className="model-row">
            <select
              value={draft.model}
              onChange={(e) => setDraft((prev) => ({ ...prev, model: e.target.value }))}
              disabled={loadingModels && modelOptions.length === 0}
              aria-label="Select model"
            >
              {modelOptions.length === 0 ? (
                <option value="">{loadingModels ? 'Detecting…' : 'No models available'}</option>
              ) : (
                modelOptions.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))
              )}
            </select>
            <button
              type="button"
              onClick={() => void fetchModels(draft)}
              disabled={loadingModels}
            >
              {loadingModels ? 'Detecting…' : 'Refresh'}
            </button>
          </div>
          {modelsError && <p className="field-hint warn">{modelsError}</p>}
          <p className="field-hint">Used for translation, Auto Caption, and reCaption.</p>
        </div>

        <label className="field checkbox-field">
          <span className="checkbox-row">
            <input
              type="checkbox"
              checked={draft.autoAnalysis}
              onChange={(e) =>
                setDraft((prev) => ({ ...prev, autoAnalysis: e.target.checked }))
              }
            />
            Auto analysis
          </span>
          <p className="field-hint">
            On: analyze captions in the background. Off: only while the Analyze dialog is
            open.
          </p>
        </label>

        <div className="settings-section">
          <h3>Auto Caption prompt</h3>
          <div className="field">
            <span>Preset</span>
            <div className="model-row">
              <select
                value={draft.activeCaptionPresetId}
                onChange={(e) =>
                  setDraft((prev) => ({ ...prev, activeCaptionPresetId: e.target.value }))
                }
                aria-label="Caption prompt preset"
              >
                {draft.captionPresets.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name}
                  </option>
                ))}
              </select>
              <button type="button" onClick={addPreset}>
                Add
              </button>
              <button type="button" onClick={deletePreset}>
                Delete
              </button>
            </div>
          </div>
          <label className="field">
            <span>Preset name</span>
            <input
              type="text"
              value={activePreset?.name ?? ''}
              onChange={(e) => updateActivePreset({ name: e.target.value })}
            />
          </label>
          <label className="field">
            <span>Prompt (auto-saved; PNG Info is appended at runtime)</span>
            <textarea
              className="prompt-textarea"
              value={activePreset?.prompt ?? ''}
              onChange={(e) => updateActivePreset({ prompt: e.target.value })}
              spellCheck={false}
            />
          </label>
        </div>

        <div className="modal-actions">
          <button type="button" onClick={resetDefaults}>
            Reset default URLs
          </button>
          <button type="button" onClick={handleTest} disabled={testing}>
            {testing ? 'Testing…' : 'Test connection'}
          </button>
          <div className="spacer" />
          <button type="button" onClick={onClose}>
            Cancel
          </button>
          <button type="button" className="primary" onClick={handleSave} disabled={saving}>
            {saving ? 'Saving…' : 'Save'}
          </button>
        </div>

        {testResult && <p className="test-ok">{testResult}</p>}
        {testError && <p className="test-err">{testError}</p>}
      </div>
    </div>
  )
}
