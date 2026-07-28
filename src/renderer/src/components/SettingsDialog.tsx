import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { AppSettings, CaptionPreset, TranslationProvider } from '../types'
import { DEFAULT_SETTINGS, normalizeSettings } from '../types'
import { createDefaultCaptionPreset } from '../defaults/captionPresets'
import { listModels, testConnection } from '../services/translation'
import {
  FALLBACK_WD14_MODEL_REPOS,
  listWd14ModelReposOrFallback
} from '../services/wd14Models'
import { DownloadFolderField } from './DownloadFolderField'
import { PythonExecutableField } from './PythonExecutableField'

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
  const [wd14Repos, setWd14Repos] = useState<string[]>([...FALLBACK_WD14_MODEL_REPOS])
  const [loadingWd14Repos, setLoadingWd14Repos] = useState(false)
  const [wd14ReposError, setWd14ReposError] = useState<string | null>(null)
  const [wd14ReposFromNetwork, setWd14ReposFromNetwork] = useState(false)
  const [testing, setTesting] = useState(false)
  const [testResult, setTestResult] = useState<string | null>(null)
  const [testError, setTestError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const fetchId = useRef(0)
  const wd14FetchId = useRef(0)
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

  const wd14RepoOptions = useMemo(() => {
    const current = draft.wd14.modelRepoId.trim()
    if (current && !wd14Repos.includes(current)) {
      return [current, ...wd14Repos]
    }
    return wd14Repos
  }, [draft.wd14.modelRepoId, wd14Repos])

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

  const fetchWd14Repos = useCallback(async () => {
    const id = ++wd14FetchId.current
    setLoadingWd14Repos(true)
    setWd14ReposError(null)
    try {
      const { repos, fromNetwork, error } = await listWd14ModelReposOrFallback()
      if (id !== wd14FetchId.current) return
      setWd14Repos(repos)
      setWd14ReposFromNetwork(fromNetwork)
      if (error && !fromNetwork) {
        setWd14ReposError(`Using offline list: ${error}`)
      }
      // Keep the user's current selection even if Hub list changes; options merge it in.
    } finally {
      if (id === wd14FetchId.current) setLoadingWd14Repos(false)
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
    setWd14ReposError(null)
    void fetchModels(normalizeSettings(settings))
    void fetchWd14Repos()
  }, [open, settings, fetchModels, fetchWd14Repos])

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

  // Auto-save caption presets / active id / appendPositivePrompt while settings stay open
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
  }, [open, draft.captionPresets, draft.activeCaptionPresetId, draft.appendPositivePrompt, onAutoSave])

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
      const next = normalizeSettings(draft)
      const needsRestart = next.uiGpuMode !== settings.uiGpuMode
      await onSave(next)
      if (needsRestart) {
        const restart = window.confirm(
          'UI GPU rendering setting changed. Restart Captioer now for it to take effect?'
        )
        if (restart) {
          await window.api.relaunchApp()
          return
        }
      }
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
      captionFormat: draft.captionFormat,
      wd14: draft.wd14,
      sidebarWidth: draft.sidebarWidth,
      rightPaneWidth: draft.rightPaneWidth,
      listViewMode: draft.listViewMode,
      thumbnailWidth: draft.thumbnailWidth,
      bucketPreview: draft.bucketPreview,
      activeView: draft.activeView,
      loraTrainJobs: draft.loraTrainJobs,
      activeLoraTrainJobId: draft.activeLoraTrainJobId,
      loraTrainApp: draft.loraTrainApp,
      uiGpuMode: draft.uiGpuMode
    })
  }

  return (
    <div
      className="modal-backdrop"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose()
      }}
    >
      <div
        className="modal modal-wide"
        role="dialog"
        aria-labelledby="settings-title"
        onMouseDown={(e) => e.stopPropagation()}
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
          <p className="field-hint">
            Used for translation, Natural Auto Caption / reCaption, and analysis.
          </p>
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
          <h3>Display</h3>
          <label className="field">
            <span>UI GPU rendering</span>
            <div className="radio-row">
              <label>
                <input
                  type="radio"
                  name="uiGpuMode"
                  checked={draft.uiGpuMode === 'auto'}
                  onChange={() => setDraft((prev) => ({ ...prev, uiGpuMode: 'auto' }))}
                />
                Auto
              </label>
              <label>
                <input
                  type="radio"
                  name="uiGpuMode"
                  checked={draft.uiGpuMode === 'onboard'}
                  onChange={() => setDraft((prev) => ({ ...prev, uiGpuMode: 'onboard' }))}
                />
                Onboard GPU
              </label>
              <label>
                <input
                  type="radio"
                  name="uiGpuMode"
                  checked={draft.uiGpuMode === 'software'}
                  onChange={() => setDraft((prev) => ({ ...prev, uiGpuMode: 'software' }))}
                />
                Software
              </label>
            </div>
            <p className="field-hint">
              Auto: OS/driver chooses. Onboard: force integrated GPU (frees dedicated VRAM when
              dual-GPU). Software: CPU/RAM only. Does not affect training or AI. Requires
              restart. Windows Graphics settings for Captioer may override Onboard.
            </p>
          </label>
        </div>

        <div className="settings-section">
          <h3>WD14 Tagging</h3>
          <DownloadFolderField
            value={draft.loraTrainApp.downloadFolder}
            onChange={(downloadFolder) =>
              setDraft((prev) => ({
                ...prev,
                loraTrainApp: { ...prev.loraTrainApp, downloadFolder }
              }))
            }
          />
          <PythonExecutableField
            value={draft.loraTrainApp.pythonPath}
            onChange={(pythonPath) =>
              setDraft((prev) => ({
                ...prev,
                loraTrainApp: { ...prev.loraTrainApp, pythonPath }
              }))
            }
            downloadFolder={draft.loraTrainApp.downloadFolder}
            enabled={open}
          />
          <p className="field-hint">
            Used when Caption format is set to Danbooru Tags(SD/XL). Download installs WD14 +
            training packages into the download folder&apos;s python venv.
          </p>
          <label className="field">
            <span>Model repo</span>
            <div className="model-row">
              <select
                value={draft.wd14.modelRepoId}
                onChange={(e) =>
                  setDraft((prev) => ({
                    ...prev,
                    wd14: { ...prev.wd14, modelRepoId: e.target.value }
                  }))
                }
                disabled={loadingWd14Repos && wd14RepoOptions.length === 0}
                aria-label="WD14 model repo"
              >
                {wd14RepoOptions.length === 0 ? (
                  <option value="">
                    {loadingWd14Repos ? 'Loading…' : 'No models available'}
                  </option>
                ) : (
                  wd14RepoOptions.map((repo) => (
                    <option key={repo} value={repo}>
                      {repo.replace(/^SmilingWolf\//, '')}
                    </option>
                  ))
                )}
              </select>
              <button
                type="button"
                onClick={() => void fetchWd14Repos()}
                disabled={loadingWd14Repos}
              >
                {loadingWd14Repos ? 'Loading…' : 'Refresh'}
              </button>
            </div>
            {wd14ReposError && <p className="field-hint warn">{wd14ReposError}</p>}
            <p className="field-hint">
              {wd14ReposFromNetwork
                ? 'Listed from Hugging Face (ONNX + selected_tags.csv).'
                : 'Offline fallback list — click Refresh when online.'}{' '}
              Full id: <code>{draft.wd14.modelRepoId || '—'}</code>
            </p>
          </label>
          <div className="model-row">
            <label className="field" style={{ flex: 1 }}>
              <span>Threshold</span>
              <input
                type="number"
                min={0}
                max={1}
                step={0.05}
                value={draft.wd14.threshold}
                onChange={(e) =>
                  setDraft((prev) => ({
                    ...prev,
                    wd14: {
                      ...prev.wd14,
                      threshold: Number(e.target.value)
                    }
                  }))
                }
              />
            </label>
            <label className="field" style={{ flex: 1 }}>
              <span>Character threshold</span>
              <input
                type="number"
                min={0}
                max={1}
                step={0.05}
                value={draft.wd14.characterThreshold}
                onChange={(e) =>
                  setDraft((prev) => ({
                    ...prev,
                    wd14: {
                      ...prev.wd14,
                      characterThreshold: Number(e.target.value)
                    }
                  }))
                }
              />
            </label>
          </div>
        </div>

        <div className="settings-section">
          <h3>Auto Caption prompt</h3>
          <p className="field-hint">
            Used only for Natural Language(Flux/Krea2) format. Danbooru Tags ignores these
            prompts.
          </p>
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
            <span>Caption Prompt</span>
            <textarea
              className="prompt-textarea"
              value={activePreset?.prompt ?? ''}
              onChange={(e) => updateActivePreset({ prompt: e.target.value })}
              spellCheck={false}
            />
          </label>
          <div className="field">
            <div
              className={`lora-toggle${draft.appendPositivePrompt ? ' is-on' : ''}`}
            >
              <button
                type="button"
                role="switch"
                className="lora-switch"
                aria-checked={draft.appendPositivePrompt}
                aria-label="Append positive Prompt"
                onClick={() =>
                  setDraft((prev) => ({
                    ...prev,
                    appendPositivePrompt: !prev.appendPositivePrompt
                  }))
                }
              >
                <span className="lora-switch-knob" aria-hidden="true" />
              </button>
              <span className="lora-toggle-label">Append positive Prompt</span>
            </div>
            <p className="field-hint">
              On: append PNG Info positive prompt to the bottom of the Auto Caption prompt
              at runtime.
            </p>
          </div>
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
