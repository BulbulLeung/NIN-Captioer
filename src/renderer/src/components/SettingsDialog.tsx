import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { AppSettings, CaptionPreset, TranslationProvider } from '../types'
import { DEFAULT_SETTINGS, modelDownloadPathFromDownloadFolder, normalizeSettings } from '../types'
import { createDefaultCaptionPreset } from '../defaults/captionPresets'
import { listModels, testConnection } from '../services/translation'
import {
  FALLBACK_WD14_MODEL_REPOS,
  listWd14ModelReposOrFallback
} from '../services/wd14Models'
import { ComfyUiBatField } from './ComfyUiBatField'
import { DownloadFolderField } from './DownloadFolderField'
import { PythonExecutableField } from './PythonExecutableField'
import { parentDir } from '../services/comfyGenerate'

type SettingsTab = 'general' | 'datasetEdit' | 'loraTrain' | 'loraTest'

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

function tabFromActiveView(view: AppSettings['activeView']): SettingsTab {
  if (view === 'datasetEdit') return 'datasetEdit'
  if (view === 'loraTrain') return 'loraTrain'
  if (view === 'loraTest') return 'loraTest'
  return 'general'
}

const TABS: { id: SettingsTab; label: string }[] = [
  { id: 'general', label: 'General' },
  { id: 'datasetEdit', label: 'Dataset Edit' },
  { id: 'loraTrain', label: 'Lora Train' },
  { id: 'loraTest', label: 'Lora Test' }
]

type ModelPathKey = 'ditPath' | 'vaePath' | 't5Path'

type LoraTestModelSpec = {
  label: string
  filename: string
}

const LORA_TEST_MODEL_REPO = 'AlperKTS/Krea2_FP8'
const LORA_TEST_MODEL_SPECS: Record<ModelPathKey, LoraTestModelSpec> = {
  ditPath: {
    label: 'Checkpoint Folder',
    filename: 'krea2_turbo_fp8.safetensors'
  },
  vaePath: {
    label: 'VAE',
    filename: 'qwen_image_vae.safetensors'
  },
  t5Path: {
    label: 'Text Encoder',
    filename: 'qwen3vl_4b_fp8_scaled.safetensors'
  }
}

function isSafetensorsPath(value: string): boolean {
  return value.trim().toLowerCase().endsWith('.safetensors')
}

export function SettingsDialog({ open, settings, onClose, onSave, onAutoSave }: Props) {
  const [draft, setDraft] = useState<AppSettings>(settings)
  const [tab, setTab] = useState<SettingsTab>(() => tabFromActiveView(settings.activeView))
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
  const [loraTestDownloadKey, setLoraTestDownloadKey] = useState<ModelPathKey | null>(null)
  const [loraTestDownloadPct, setLoraTestDownloadPct] = useState(0)
  const [loraTestDownloadError, setLoraTestDownloadError] = useState<string | null>(null)
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
    } finally {
      if (id === wd14FetchId.current) setLoadingWd14Repos(false)
    }
  }, [])

  useEffect(() => {
    if (!open) return
    skipUrlFetch.current = true
    skipPromptSave.current = true
    setDraft(normalizeSettings(settings))
    setTab(tabFromActiveView(settings.activeView))
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

  useEffect(() => {
    const offProgress = window.api.onModelDownloadProgress((payload) => {
      if (payload.repoId !== LORA_TEST_MODEL_REPO) return
      setLoraTestDownloadPct(payload.pct)
    })
    const offDone = window.api.onModelDownloadDone((payload) => {
      if (payload.repoId !== LORA_TEST_MODEL_REPO) return
      const activeKey = loraTestDownloadKey
      setLoraTestDownloadKey(null)
      setLoraTestDownloadPct(100)
      if (!activeKey || !payload.filePath) return
      setDraft((prev) => {
        const nextDraft = { ...prev.loraTestDraft, [activeKey]: payload.filePath }
        if (activeKey === 'ditPath') {
          nextDraft.checkpointFolder = parentDir(payload.filePath)
        }
        return { ...prev, loraTestDraft: nextDraft }
      })
    })
    const offError = window.api.onModelDownloadError((payload) => {
      if (payload.repoId !== LORA_TEST_MODEL_REPO) return
      setLoraTestDownloadKey(null)
      setLoraTestDownloadError(payload.message)
    })
    return () => {
      offProgress()
      offDone()
      offError()
    }
  }, [loraTestDownloadKey])

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

  const browseSafetensors = async (key: ModelPathKey, title: string) => {
    const file = await window.api.openFile({
      title,
      filters: [
        { name: 'Safetensors', extensions: ['safetensors'] },
        { name: 'All Files', extensions: ['*'] }
      ]
    })
    if (!file) return
    setDraft((prev) => ({
      ...prev,
      loraTestDraft: { ...prev.loraTestDraft, [key]: file }
    }))
  }

  const browseCheckpointFolder = async () => {
    const dir = await window.api.openFolder()
    if (!dir) return
    setDraft((prev) => ({
      ...prev,
      loraTestDraft: { ...prev.loraTestDraft, checkpointFolder: dir }
    }))
  }

  const downloadLoraTestModel = async (key: ModelPathKey) => {
    if (loraTestDownloadKey) return
    setLoraTestDownloadError(null)
    setLoraTestDownloadKey(key)
    setLoraTestDownloadPct(0)
    const spec = LORA_TEST_MODEL_SPECS[key]
    const result = await window.api.downloadModel({
      pythonPath: draft.loraTrainApp.pythonPath,
      downloadPath: modelDownloadPathFromDownloadFolder(draft.loraTrainApp.downloadFolder),
      token: draft.loraTrainApp.huggingfaceToken,
      repoId: LORA_TEST_MODEL_REPO,
      fileName: spec.filename
    })
    if (!result.ok) {
      setLoraTestDownloadKey(null)
      setLoraTestDownloadError(result.error || `Failed to download ${spec.label}`)
    }
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
      loraTestDraft: draft.loraTestDraft,
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
        className="modal modal-wide settings-modal"
        role="dialog"
        aria-labelledby="settings-title"
        onMouseDown={(e) => e.stopPropagation()}
      >
        <h2 id="settings-title">Settings</h2>

        <div className="settings-tabs view-switch" role="tablist" aria-label="Settings sections">
          {TABS.map((t) => (
            <button
              key={t.id}
              type="button"
              role="tab"
              className={`view-switch-seg${tab === t.id ? ' active' : ''}`}
              aria-selected={tab === t.id}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </div>

        <div className="settings-tab-body">
          {tab === 'general' && (
            <>
              <div className="settings-section settings-section-first">
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
                    Auto: OS/driver chooses. Onboard: force integrated GPU (frees dedicated VRAM
                    when dual-GPU). Software: CPU/RAM only. Does not affect training or AI.
                    Requires restart. Windows Graphics settings for Captioer may override Onboard.
                  </p>
                </label>
              </div>

              <div className="settings-section">
                <h3>Environment</h3>
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
                  hint={
                    <>
                      Shared by Dataset Edit (WD14), LoraTrain, and ComfyUI install. Download
                      installs CUDA torch 2.9.1 (cu128), packages from{' '}
                      <code>trainer/requirements.txt</code> and{' '}
                      <code>trainer/requirements-wd14.txt</code>, and on Windows{' '}
                      <code>triton-windows</code> and <code>flash-attn</code> (FA2 wheel).
                    </>
                  }
                />
                <label className="field">
                  <span>Hugging Face token (for gated Krea models)</span>
                  <input
                    type="password"
                    value={draft.loraTrainApp.huggingfaceToken}
                    onChange={(e) =>
                      setDraft((prev) => ({
                        ...prev,
                        loraTrainApp: {
                          ...prev.loraTrainApp,
                          huggingfaceToken: e.target.value
                        }
                      }))
                    }
                    placeholder="hf_… from huggingface.co/settings/tokens"
                    autoComplete="off"
                  />
                  <p className="field-hint">
                    Also open{' '}
                    <a
                      href="https://huggingface.co/krea/Krea-2-Raw"
                      target="_blank"
                      rel="noreferrer"
                    >
                      krea/Krea-2-Raw
                    </a>{' '}
                    (and Turbo) while logged in and click Agree — token alone is not enough.
                  </p>
                </label>
                <ComfyUiBatField
                  value={draft.loraTestDraft.comfyUiBatPath}
                  onChange={(comfyUiBatPath) =>
                    setDraft((prev) => ({
                      ...prev,
                      loraTestDraft: { ...prev.loraTestDraft, comfyUiBatPath }
                    }))
                  }
                  downloadFolder={draft.loraTrainApp.downloadFolder}
                  pythonPath={draft.loraTrainApp.pythonPath}
                  enabled={open}
                />
              </div>
            </>
          )}

          {tab === 'datasetEdit' && (
            <>
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
                      <option value="">
                        {loadingModels ? 'Detecting…' : 'No models available'}
                      </option>
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

              <div className="settings-section">
                <h3>WD14 Tagging</h3>
                <p className="field-hint">
                  Used when Caption format is set to Danbooru Tags(SD/XL). Python / download
                  folder are configured on the General tab.
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
                <h3>Auto analysis</h3>
                <div className="field">
                  <div
                    className={`settings-toggle-row lora-toggle${draft.autoAnalysis ? ' is-on' : ''}`}
                  >
                    <button
                      type="button"
                      role="switch"
                      className="lora-switch"
                      aria-checked={draft.autoAnalysis}
                      aria-label="Auto analysis"
                      onClick={() =>
                        setDraft((prev) => ({
                          ...prev,
                          autoAnalysis: !prev.autoAnalysis
                        }))
                      }
                    >
                      <span className="lora-switch-knob" aria-hidden="true" />
                    </button>
                    <span className="lora-toggle-label">Background analysis</span>
                  </div>
                  <p className="field-hint">
                    On: analyze captions in the background. Off: only while the Analyze dialog is
                    open.
                  </p>
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
                        setDraft((prev) => ({
                          ...prev,
                          activeCaptionPresetId: e.target.value
                        }))
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
            </>
          )}

          {tab === 'loraTrain' && (
            <p className="field-hint">
              Job hyper-parameters (network, steps, datasets, model path, sampling) are edited on
              the LoraTrain panel. Shared environment settings live on the General tab.
            </p>
          )}

          {tab === 'loraTest' && (
            <>
              <p className="field-hint">
                Base models for ComfyUI generation. ComfyUI launch bat is on the General tab.
              </p>
              {loraTestDownloadError && <p className="test-err">{loraTestDownloadError}</p>}
              <PathBrowseField
                label={LORA_TEST_MODEL_SPECS.ditPath.label}
                value={draft.loraTestDraft.checkpointFolder}
                onChange={(checkpointFolder) =>
                  setDraft((prev) => ({
                    ...prev,
                    loraTestDraft: { ...prev.loraTestDraft, checkpointFolder }
                  }))
                }
                onBrowse={() => void browseCheckpointFolder()}
                placeholder="Checkpoint folder path"
                showDownload={!draft.loraTestDraft.checkpointFolder.trim()}
                downloading={loraTestDownloadKey === 'ditPath'}
                downloadLabel={
                  loraTestDownloadKey === 'ditPath' ? `Downloading ${loraTestDownloadPct}%` : 'Download'
                }
                onDownload={() => void downloadLoraTestModel('ditPath')}
              />
              <p className="field-hint">
                Detects .safetensors files inside this folder for the Checkpoint dropdown.
              </p>
              <PathBrowseField
                label="VAE"
                value={draft.loraTestDraft.vaePath}
                onChange={(vaePath) =>
                  setDraft((prev) => ({
                    ...prev,
                    loraTestDraft: { ...prev.loraTestDraft, vaePath }
                  }))
                }
                onBrowse={() => void browseSafetensors('vaePath', 'Select VAE safetensors')}
                showDownload={!isSafetensorsPath(draft.loraTestDraft.vaePath)}
                downloading={loraTestDownloadKey === 'vaePath'}
                downloadLabel={
                  loraTestDownloadKey === 'vaePath' ? `Downloading ${loraTestDownloadPct}%` : 'Download'
                }
                onDownload={() => void downloadLoraTestModel('vaePath')}
              />
              <PathBrowseField
                label={LORA_TEST_MODEL_SPECS.t5Path.label}
                value={draft.loraTestDraft.t5Path}
                onChange={(t5Path) =>
                  setDraft((prev) => ({
                    ...prev,
                    loraTestDraft: { ...prev.loraTestDraft, t5Path }
                  }))
                }
                onBrowse={() =>
                  void browseSafetensors('t5Path', 'Select Qwen text encoder')
                }
                showDownload={!isSafetensorsPath(draft.loraTestDraft.t5Path)}
                downloading={loraTestDownloadKey === 't5Path'}
                downloadLabel={
                  loraTestDownloadKey === 't5Path' ? `Downloading ${loraTestDownloadPct}%` : 'Download'
                }
                onDownload={() => void downloadLoraTestModel('t5Path')}
              />
            </>
          )}
        </div>

        <div className="modal-actions">
          {tab === 'datasetEdit' && (
            <>
              <button type="button" onClick={resetDefaults}>
                Reset default URLs
              </button>
              <button type="button" onClick={() => void handleTest()} disabled={testing}>
                {testing ? 'Testing…' : 'Test connection'}
              </button>
            </>
          )}
          <div className="spacer" />
          <button type="button" onClick={onClose}>
            Cancel
          </button>
          <button
            type="button"
            className="primary"
            onClick={() => void handleSave()}
            disabled={saving}
          >
            {saving ? 'Saving…' : 'Save'}
          </button>
        </div>

        {tab === 'datasetEdit' && testResult && <p className="test-ok">{testResult}</p>}
        {tab === 'datasetEdit' && testError && <p className="test-err">{testError}</p>}
      </div>
    </div>
  )
}

function PathBrowseField({
  label,
  value,
  onChange,
  onBrowse,
  placeholder,
  showDownload = false,
  downloading = false,
  downloadLabel = 'Download',
  onDownload
}: {
  label: string
  value: string
  onChange: (v: string) => void
  onBrowse: () => void
  placeholder?: string
  showDownload?: boolean
  downloading?: boolean
  downloadLabel?: string
  onDownload?: () => void
}) {
  return (
    <label className="field">
      <span>{label}</span>
      <div className="field-row">
        <input
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder ?? `${label} .safetensors path`}
          spellCheck={false}
        />
        <button type="button" onClick={onBrowse}>
          Browse
        </button>
        {showDownload && onDownload && (
          <button type="button" className="primary" onClick={onDownload} disabled={downloading}>
            {downloadLabel}
          </button>
        )}
      </div>
    </label>
  )
}
