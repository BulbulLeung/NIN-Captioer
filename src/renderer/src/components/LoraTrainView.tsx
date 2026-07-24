import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react'
import type { LoraTrainAppSettings, LoraTrainJobConfig } from '../types'
import { KREA2_RAW, KREA2_TURBO, normalizeLoraTrainJob } from '../types'
import { serializeTrainConfig } from '../services/trainConfig'
import { join } from '../utils/pathJoin'
import { GpuDeviceSelect } from './GpuDeviceSelect'
import { ResourceMonitorPane } from './ResourceMonitorPane'

type SectionId = 'basics' | 'dataset' | 'train' | 'lora' | 'sample' | 'advanced'

type ModelStatusKind = 'missing' | 'ready' | 'updateAvailable' | 'local' | 'error'

interface ModelStatusItem {
  role: string
  path: string
  repoId: string | null
  status: ModelStatusKind
  localPath?: string | null
  localRevision?: string | null
  remoteRevision?: string | null
  message?: string | null
}

function looksLikeHfRepoId(value: string): boolean {
  const s = value.trim()
  if (!s || s.includes('\\') || s.startsWith('/') || s.startsWith('.')) return false
  if (s.length >= 2 && s[1] === ':') return false
  if (s.startsWith('~')) return false
  const parts = s.split('/')
  return parts.length === 2 && parts.every(Boolean) && !parts.some((p) => p.includes(' '))
}

function resolveDownloadRepoId(st?: ModelStatusItem | null): string | null {
  if (!st) return null
  if (st.repoId?.trim()) return st.repoId.trim()
  if (looksLikeHfRepoId(st.path)) return st.path.trim()
  return null
}

const SECTIONS: { id: SectionId; label: string; hint: string }[] = [
  { id: 'basics', label: 'Basics', hint: 'Name, output, models' },
  { id: 'dataset', label: 'Dataset', hint: 'Images + captions' },
  { id: 'train', label: 'Train', hint: 'Steps, LR, optimizer' },
  { id: 'lora', label: 'LoRA', hint: 'Rank & alpha' },
  { id: 'sample', label: 'Sample', hint: 'Preview during train' },
  { id: 'advanced', label: 'Advanced', hint: 'Save, EMA, VRAM' }
]

interface Props {
  job: LoraTrainJobConfig
  appSettings: LoraTrainAppSettings
  datasetFolders: string[]
  onChange: (job: LoraTrainJobConfig) => void
  onStatus: (message: string, isError?: boolean) => void
}

function Field({
  label,
  children,
  hint
}: {
  label: string
  children: ReactNode
  hint?: string
}) {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
      {hint ? <p className="field-hint">{hint}</p> : null}
    </label>
  )
}

function CheckboxField({
  label,
  checked,
  onChange,
  hint,
  disabled
}: {
  label: string
  checked: boolean
  onChange: (v: boolean) => void
  hint?: string
  disabled?: boolean
}) {
  return (
    <label className="field checkbox-field">
      <span className="checkbox-row">
        <input
          type="checkbox"
          checked={checked}
          disabled={disabled}
          onChange={(e) => onChange(e.target.checked)}
        />
        {label}
      </span>
      {hint ? <p className="field-hint">{hint}</p> : null}
    </label>
  )
}

export function LoraTrainView({
  job,
  appSettings,
  datasetFolders,
  onChange,
  onStatus
}: Props) {
  const [draft, setDraft] = useState(() => normalizeLoraTrainJob(job))
  const [resolutionText, setResolutionText] = useState(() =>
    normalizeLoraTrainJob(job).datasets[0].resolution.join(', ')
  )
  const [activeSection, setActiveSection] = useState<SectionId>('basics')
  const [exporting, setExporting] = useState(false)
  const [training, setTraining] = useState(false)
  const [progress, setProgress] = useState<{
    step: number
    total: number
    loss: number
  } | null>(null)
  const [modelStatuses, setModelStatuses] = useState<ModelStatusItem[]>([])
  const [modelChecking, setModelChecking] = useState(false)
  const [modelCheckError, setModelCheckError] = useState<string | null>(null)
  const [dlCurrent, setDlCurrent] = useState<string | null>(null)
  const [dlPct, setDlPct] = useState(0)
  const [dlQueue, setDlQueue] = useState<string[]>([])
  const skipPersist = useRef(true)
  const draftRef = useRef(draft)
  draftRef.current = draft
  const persistTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const modelStatusesRef = useRef(modelStatuses)
  modelStatusesRef.current = modelStatuses
  const dlQueueRef = useRef<string[]>([])
  const downloadingRef = useRef(false)
  const appSettingsRef = useRef(appSettings)
  appSettingsRef.current = appSettings

  useEffect(() => {
    skipPersist.current = true
    const next = normalizeLoraTrainJob(job)
    setDraft(next)
    setResolutionText(next.datasets[0].resolution.join(', '))
  }, [job])

  const patch = useCallback((updater: (prev: LoraTrainJobConfig) => LoraTrainJobConfig) => {
    setDraft((prev) => updater(prev))
  }, [])

  const refreshModelStatus = useCallback(async () => {
    setModelChecking(true)
    setModelCheckError(null)
    try {
      const d = draftRef.current
      const settings = appSettingsRef.current
      const result = await window.api.checkModelStatus({
        pythonPath: settings.pythonPath.trim() || undefined,
        downloadPath: settings.modelDownloadPath.trim() || undefined,
        token: settings.huggingfaceToken.trim() || undefined,
        targets: [
          { role: 'train', path: d.model.train_name_or_path },
          { role: 'sample', path: d.model.sample_name_or_path }
        ]
      })
      const results = result.results || []
      setModelStatuses(results)
      if (!result.ok && result.error) {
        setModelCheckError(result.error)
      }

      // If HF id still in job but a complete snapshot exists under download path, point at local dir.
      let trainPath = d.model.train_name_or_path
      let samplePath = d.model.sample_name_or_path
      let changed = false
      for (const st of results) {
        if (
          !st.localPath ||
          !st.repoId ||
          !st.localRevision ||
          st.path !== st.repoId ||
          (st.status !== 'ready' && st.status !== 'updateAvailable')
        ) {
          continue
        }
        if (st.role === 'train' && trainPath === st.repoId) {
          trainPath = st.localPath
          changed = true
        }
        if (st.role === 'sample' && samplePath === st.repoId) {
          samplePath = st.localPath
          changed = true
        }
      }
      if (changed) {
        setDraft((prev) => ({
          ...prev,
          model: {
            ...prev.model,
            train_name_or_path: trainPath,
            name_or_path: trainPath,
            sample_name_or_path: samplePath
          }
        }))
      }
    } catch (err) {
      setModelCheckError(err instanceof Error ? err.message : String(err))
      setModelStatuses([])
    } finally {
      setModelChecking(false)
    }
  }, [])

  const applyDownloadedPath = useCallback((repoId: string, localPath: string) => {
    setDraft((prev) => {
      let trainPath = prev.model.train_name_or_path
      let samplePath = prev.model.sample_name_or_path
      for (const st of modelStatusesRef.current) {
        if (st.repoId === repoId || st.path === repoId) {
          if (st.role === 'train') trainPath = localPath
          if (st.role === 'sample') samplePath = localPath
        }
      }
      if (trainPath === repoId) trainPath = localPath
      if (samplePath === repoId) samplePath = localPath
      return {
        ...prev,
        model: {
          ...prev.model,
          train_name_or_path: trainPath,
          name_or_path: trainPath,
          sample_name_or_path: samplePath
        }
      }
    })
  }, [])

  const pumpDownloadQueue = useCallback(async () => {
    if (downloadingRef.current) return
    const next = dlQueueRef.current[0]
    if (!next) {
      setDlCurrent(null)
      setDlPct(0)
      setDlQueue([])
      return
    }
    downloadingRef.current = true
    setDlCurrent(next)
    setDlPct(0)
    setDlQueue([...dlQueueRef.current])
    const settings = appSettingsRef.current
    const result = await window.api.downloadModel({
      pythonPath: settings.pythonPath.trim() || undefined,
      downloadPath: settings.modelDownloadPath.trim() || undefined,
      token: settings.huggingfaceToken.trim() || undefined,
      repoId: next
    })
    if (!result.ok) {
      downloadingRef.current = false
      dlQueueRef.current = []
      setDlQueue([])
      setDlCurrent(null)
      onStatus(result.error || 'Failed to start model download', true)
    }
  }, [onStatus])

  const enqueueDownloads = useCallback(
    (repoIds: string[]) => {
      const unique = [...new Set(repoIds.map((r) => r.trim()).filter(Boolean))]
      if (unique.length === 0) return
      for (const id of unique) {
        if (!dlQueueRef.current.includes(id)) dlQueueRef.current.push(id)
      }
      setDlQueue([...dlQueueRef.current])
      void pumpDownloadQueue()
    },
    [pumpDownloadQueue]
  )

  useEffect(() => {
    void window.api.trainStatus().then((s) => setTraining(s.running))
    const offProg = window.api.onTrainProgress((p) => setProgress(p))
    const offDone = window.api.onTrainDone(({ path }) => {
      setTraining(false)
      onStatus(`Training done: ${path}`)
    })
    const offErr = window.api.onTrainError(({ message }) => {
      setTraining(false)
      onStatus(message, true)
    })
    const offDlProg = window.api.onModelDownloadProgress(({ repoId, pct }) => {
      setDlCurrent(repoId)
      setDlPct(pct)
    })
    const offDlDone = window.api.onModelDownloadDone(({ repoId, path }) => {
      applyDownloadedPath(repoId, path)
      onStatus(`Model ready: ${repoId}`)
      dlQueueRef.current = dlQueueRef.current.filter((id) => id !== repoId)
      setDlQueue([...dlQueueRef.current])
      downloadingRef.current = false
      if (dlQueueRef.current.length === 0) {
        setDlCurrent(null)
        setDlPct(0)
        void refreshModelStatus()
      } else {
        void pumpDownloadQueue()
      }
    })
    const offDlErr = window.api.onModelDownloadError(({ message }) => {
      downloadingRef.current = false
      dlQueueRef.current = []
      setDlQueue([])
      setDlCurrent(null)
      setDlPct(0)
      onStatus(message, true)
      void refreshModelStatus()
    })
    return () => {
      offProg()
      offDone()
      offErr()
      offDlProg()
      offDlDone()
      offDlErr()
    }
  }, [onStatus, applyDownloadedPath, pumpDownloadQueue, refreshModelStatus])

  useEffect(() => {
    const timer = setTimeout(() => {
      void refreshModelStatus()
    }, 600)
    return () => clearTimeout(timer)
  }, [
    draft.model.train_name_or_path,
    draft.model.sample_name_or_path,
    appSettings.modelDownloadPath,
    appSettings.huggingfaceToken,
    appSettings.pythonPath,
    refreshModelStatus
  ])

  useEffect(() => {
    if (skipPersist.current) {
      skipPersist.current = false
      return
    }
    if (persistTimer.current) clearTimeout(persistTimer.current)
    persistTimer.current = setTimeout(() => {
      onChange(normalizeLoraTrainJob(draftRef.current))
    }, 400)
    return () => {
      if (persistTimer.current) clearTimeout(persistTimer.current)
    }
  }, [draft, onChange])

  const ds0 = draft.datasets[0]
  const hasDataset = Boolean(ds0.folder_path?.trim())

  const statusForRole = (role: string): ModelStatusItem | undefined =>
    modelStatuses.find((s) => s.role === role)

  const statusLabel = (status?: ModelStatusKind): string => {
    if (!status) return modelChecking ? 'Checking…' : '—'
    switch (status) {
      case 'missing':
        return 'Missing'
      case 'ready':
        return 'Ready'
      case 'updateAvailable':
        return 'Update'
      case 'local':
        return 'Local'
      case 'error':
        return 'Error'
      default:
        return status
    }
  }

  const missingAny = modelStatuses.filter((s) => s.status === 'missing')
  const missingRepos = missingAny.filter((s) => Boolean(resolveDownloadRepoId(s)))
  const updateRepos = modelStatuses.filter(
    (s) => s.status === 'updateAvailable' && Boolean(resolveDownloadRepoId(s))
  )
  const isDownloading = Boolean(dlCurrent)
  // Top banner only for in-progress download / check errors (Download lives next to Missing label)
  const showBanner = Boolean(isDownloading || modelCheckError)

  const downloadForRole = (role: string) => {
    const st = statusForRole(role)
    const repo = resolveDownloadRepoId(st)
    if (!repo) return
    const hasToken = Boolean(appSettingsRef.current.huggingfaceToken.trim())
    if (!hasToken) {
      onStatus(
        'Hugging Face token required for gated Krea models. Set it in LoRA Train Settings, accept access on the model page, then retry Download.',
        true
      )
      return
    }
    enqueueDownloads([repo])
  }

  const cancelDownload = async () => {
    await window.api.cancelModelDownload()
    downloadingRef.current = false
    dlQueueRef.current = []
    setDlQueue([])
    setDlCurrent(null)
    setDlPct(0)
    onStatus('Model download cancelled')
    void refreshModelStatus()
  }

  const browseDatasetFolder = async () => {
    const dir = await window.api.openFolder()
    if (!dir) return
    patch((prev) => ({
      ...prev,
      datasets: prev.datasets.map((ds, i) => (i === 0 ? { ...ds, folder_path: dir } : ds))
    }))
  }

  const browseTrainingFolder = async () => {
    const dir = await window.api.openFolder()
    if (!dir) return
    patch((prev) => ({ ...prev, training_folder: dir }))
  }

  const exportConfig = async () => {
    setExporting(true)
    try {
      const normalized = normalizeLoraTrainJob(draftRef.current)
      const json = serializeTrainConfig(normalized, {
        huggingface_token: appSettings.huggingfaceToken || undefined
      })
      const fileName =
        appSettings.exportFileName.trim() ||
        `${normalized.name || 'captioer_krea2_train'}.json`
      const defaultPath = appSettings.exportDir
        ? join(appSettings.exportDir, fileName)
        : fileName
      const saved = await window.api.saveTextFile({
        defaultPath,
        content: json,
        filters: [
          { name: 'JSON', extensions: ['json'] },
          { name: 'All Files', extensions: ['*'] }
        ]
      })
      if (saved) onStatus(`Exported config: ${saved}`)
    } catch (err) {
      onStatus(err instanceof Error ? err.message : String(err), true)
    } finally {
      setExporting(false)
    }
  }

  const startTrain = async () => {
    const normalized = normalizeLoraTrainJob(draftRef.current)
    if (!normalized.datasets[0]?.folder_path) {
      setActiveSection('dataset')
      onStatus('Choose a dataset folder first', true)
      return
    }
    if (normalized.model.arch !== 'krea2') {
      onStatus('Only Krea 2 is supported', true)
      return
    }
    const py = appSettings.pythonPath.trim()
    setProgress(null)
    setTraining(true)
    const configJson = serializeTrainConfig(normalized, {
      huggingface_token: appSettings.huggingfaceToken || undefined
    })
    const result = await window.api.startTrain({
      pythonPath: py || undefined,
      configJson,
      device: normalized.device
    })
    if (!result.ok) {
      setTraining(false)
      onStatus(result.error || 'Failed to start training', true)
    }
  }

  const stopTrain = async () => {
    await window.api.stopTrain()
    setTraining(false)
  }

  const progressPct =
    progress && progress.total > 0
      ? Math.min(100, Math.round((100 * progress.step) / progress.total))
      : null

  const sectionTitle = SECTIONS.find((s) => s.id === activeSection)?.label ?? ''

  return (
    <div className="lora-train">
      {(showBanner) && (
        <div className="lora-model-banner" role="status">
          {isDownloading ? (
            <>
              <p>
                Downloading <code>{dlCurrent}</code>
                {dlQueue.length > 1 ? ` (+${dlQueue.length - 1} queued)` : ''}
                {` · ${dlPct}%`}
              </p>
              <div className="lora-model-banner-actions">
                <button type="button" className="danger" onClick={() => void cancelDownload()}>
                  Cancel
                </button>
              </div>
            </>
          ) : (
            <>
              <p>Model check failed: {modelCheckError}</p>
              <div className="lora-model-banner-actions">
                <button type="button" onClick={() => void refreshModelStatus()} disabled={modelChecking}>
                  {modelChecking ? 'Checking…' : 'Retry'}
                </button>
              </div>
            </>
          )}
        </div>
      )}

      <div className="lora-train-body">
        <nav className="lora-nav" aria-label="Training settings sections">
          <div className="lora-nav-items">
            {SECTIONS.map((s) => (
              <button
                key={s.id}
                type="button"
                className={`lora-nav-item${activeSection === s.id ? ' active' : ''}`}
                aria-current={activeSection === s.id ? 'page' : undefined}
                onClick={() => setActiveSection(s.id)}
              >
                <span className="lora-nav-label">{s.label}</span>
                <span className="lora-nav-hint">{s.hint}</span>
              </button>
            ))}
            <p className="lora-nav-note">
              Required: dataset folder + Train base (Raw)
              {!hasDataset ? (
                <>
                  {' '}
                  ·{' '}
                  <button
                    type="button"
                    className="linkish"
                    onClick={() => setActiveSection('dataset')}
                  >
                    Set dataset
                  </button>
                </>
              ) : null}
            </p>
          </div>
          <div className="lora-nav-actions">
            {training && progress && (
              <p className="lora-nav-progress">
                Step {progress.step}/{progress.total}
                {progressPct !== null ? ` (${progressPct}%)` : ''} · loss{' '}
                {progress.loss.toFixed(4)}
              </p>
            )}
            <button
              type="button"
              onClick={() => void exportConfig()}
              disabled={exporting || training}
            >
              {exporting ? 'Exporting…' : 'Export config'}
            </button>
            {training ? (
              <button type="button" className="danger" onClick={() => void stopTrain()}>
                Stop
              </button>
            ) : (
              <button type="button" className="primary" onClick={() => void startTrain()}>
                Start Train
              </button>
            )}
          </div>
        </nav>

        <div className={`lora-train-panel${training ? ' is-training' : ''}`}>
          <h3 className="lora-panel-title">{sectionTitle}</h3>

          {activeSection === 'basics' && (
            <div className="lora-grid">
              <Field label="Job name">
                <input
                  type="text"
                  value={draft.name}
                  disabled={training}
                  onChange={(e) => patch((p) => ({ ...p, name: e.target.value }))}
                />
              </Field>
              <Field label="Output folder" hint="Where checkpoints and LoRA weights are saved">
                <div className="model-row">
                  <input
                    type="text"
                    value={draft.training_folder}
                    disabled={training}
                    onChange={(e) =>
                      patch((p) => ({ ...p, training_folder: e.target.value }))
                    }
                  />
                  <button
                    type="button"
                    disabled={training}
                    onClick={() => void browseTrainingFolder()}
                  >
                    Browse
                  </button>
                </div>
              </Field>
              <Field label="GPU device">
                <GpuDeviceSelect
                  value={draft.device}
                  onChange={(device) => patch((p) => ({ ...p, device }))}
                />
              </Field>
              <Field
                label="Trigger word"
                hint="Optional. Prefixed onto captions when missing."
              >
                <input
                  type="text"
                  value={draft.trigger_word}
                  disabled={training}
                  onChange={(e) => patch((p) => ({ ...p, trigger_word: e.target.value }))}
                />
              </Field>
              <Field
                label="Train base (Raw)"
                hint="Official: train LoRA on Krea-2-Raw"
              >
                <div className="model-row">
                  <input
                    type="text"
                    value={draft.model.train_name_or_path}
                    disabled={training}
                    onChange={(e) =>
                      patch((p) => ({
                        ...p,
                        model: {
                          ...p.model,
                          train_name_or_path: e.target.value,
                          name_or_path: e.target.value,
                          arch: 'krea2'
                        }
                      }))
                    }
                  />
                  <button
                    type="button"
                    disabled={training}
                    onClick={() =>
                      patch((p) => ({
                        ...p,
                        model: {
                          ...p.model,
                          train_name_or_path: KREA2_RAW,
                          name_or_path: KREA2_RAW,
                          arch: 'krea2'
                        }
                      }))
                    }
                  >
                    Raw
                  </button>
                  <span
                    className={`lora-model-status status-${statusForRole('train')?.status || 'unknown'}`}
                    title={statusForRole('train')?.message || undefined}
                  >
                    {statusLabel(statusForRole('train')?.status)}
                  </span>
                  {statusForRole('train')?.status === 'missing' &&
                  resolveDownloadRepoId(statusForRole('train')) ? (
                    <button
                      type="button"
                      className="primary"
                      disabled={training || isDownloading}
                      onClick={() => downloadForRole('train')}
                    >
                      {dlCurrent &&
                      dlCurrent === resolveDownloadRepoId(statusForRole('train'))
                        ? `${dlPct}%`
                        : 'Download'}
                    </button>
                  ) : null}
                  {statusForRole('train')?.status === 'updateAvailable' &&
                  resolveDownloadRepoId(statusForRole('train')) ? (
                    <button
                      type="button"
                      className="primary"
                      disabled={training || isDownloading}
                      onClick={() => downloadForRole('train')}
                    >
                      {dlCurrent &&
                      dlCurrent === resolveDownloadRepoId(statusForRole('train'))
                        ? `${dlPct}%`
                        : 'Update'}
                    </button>
                  ) : null}
                </div>
              </Field>
              <Field
                label="Apply on (Turbo)"
                hint="Use the trained LoRA on Turbo for fast inference"
              >
                <div className="model-row">
                  <input
                    type="text"
                    value={draft.model.sample_name_or_path}
                    disabled={training}
                    onChange={(e) =>
                      patch((p) => ({
                        ...p,
                        model: { ...p.model, sample_name_or_path: e.target.value }
                      }))
                    }
                  />
                  <button
                    type="button"
                    disabled={training}
                    onClick={() =>
                      patch((p) => ({
                        ...p,
                        model: { ...p.model, sample_name_or_path: KREA2_TURBO }
                      }))
                    }
                  >
                    Turbo
                  </button>
                  <span
                    className={`lora-model-status status-${statusForRole('sample')?.status || 'unknown'}`}
                    title={statusForRole('sample')?.message || undefined}
                  >
                    {statusLabel(statusForRole('sample')?.status)}
                  </span>
                  {statusForRole('sample')?.status === 'missing' &&
                  resolveDownloadRepoId(statusForRole('sample')) ? (
                    <button
                      type="button"
                      className="primary"
                      disabled={training || isDownloading}
                      onClick={() => downloadForRole('sample')}
                    >
                      {dlCurrent &&
                      dlCurrent === resolveDownloadRepoId(statusForRole('sample'))
                        ? `${dlPct}%`
                        : 'Download'}
                    </button>
                  ) : null}
                  {statusForRole('sample')?.status === 'updateAvailable' &&
                  resolveDownloadRepoId(statusForRole('sample')) ? (
                    <button
                      type="button"
                      className="primary"
                      disabled={training || isDownloading}
                      onClick={() => downloadForRole('sample')}
                    >
                      {dlCurrent &&
                      dlCurrent === resolveDownloadRepoId(statusForRole('sample'))
                        ? `${dlPct}%`
                        : 'Update'}
                    </button>
                  ) : null}
                </div>
              </Field>
            </div>
          )}

          {activeSection === 'dataset' && (
            <div className="lora-grid">
              <Field label="Dataset folder" hint="Folder of images with matching .txt captions">
                <div className="model-row">
                  <input
                    type="text"
                    value={ds0.folder_path}
                    disabled={training}
                    onChange={(e) =>
                      patch((p) => ({
                        ...p,
                        datasets: p.datasets.map((ds, i) =>
                          i === 0 ? { ...ds, folder_path: e.target.value } : ds
                        )
                      }))
                    }
                  />
                  <button
                    type="button"
                    disabled={training}
                    onClick={() => void browseDatasetFolder()}
                  >
                    Browse
                  </button>
                </div>
                {datasetFolders.length > 0 && (
                  <select
                    className="lora-folder-pick"
                    aria-label="Pick from DatasetEdit folders"
                    disabled={training}
                    value=""
                    onChange={(e) => {
                      const v = e.target.value
                      if (!v) return
                      patch((p) => ({
                        ...p,
                        datasets: p.datasets.map((ds, i) =>
                          i === 0 ? { ...ds, folder_path: v } : ds
                        )
                      }))
                    }}
                  >
                    <option value="">Use DatasetEdit folder…</option>
                    {datasetFolders.map((dir) => (
                      <option key={dir} value={dir}>
                        {dir}
                      </option>
                    ))}
                  </select>
                )}
              </Field>
              <Field label="Caption file extension" hint="Usually txt">
                <input
                  type="text"
                  value={ds0.caption_ext}
                  disabled={training}
                  onChange={(e) =>
                    patch((p) => ({
                      ...p,
                      datasets: p.datasets.map((ds, i) =>
                        i === 0 ? { ...ds, caption_ext: e.target.value } : ds
                      )
                    }))
                  }
                />
              </Field>
              <Field
                label="Resolution"
                hint="Comma-separated sizes, e.g. 1024 or 512, 768, 1024"
              >
                <input
                  type="text"
                  value={resolutionText}
                  disabled={training}
                  onChange={(e) => setResolutionText(e.target.value)}
                  onBlur={() => {
                    const nums = resolutionText
                      .split(/[,\s]+/)
                      .map((s) => Number(s.trim()))
                      .filter((n) => Number.isFinite(n) && n > 0)
                    if (nums.length === 0) {
                      setResolutionText(ds0.resolution.join(', '))
                      return
                    }
                    setResolutionText(nums.join(', '))
                    patch((p) => ({
                      ...p,
                      datasets: p.datasets.map((ds, i) =>
                        i === 0 ? { ...ds, resolution: nums } : ds
                      )
                    }))
                  }}
                />
              </Field>
              <Field label="Caption dropout" hint="0–1; randomly blank captions during training">
                <input
                  type="number"
                  min={0}
                  max={1}
                  step={0.01}
                  value={ds0.caption_dropout_rate}
                  disabled={training}
                  onChange={(e) =>
                    patch((p) => ({
                      ...p,
                      datasets: p.datasets.map((ds, i) =>
                        i === 0
                          ? { ...ds, caption_dropout_rate: Number(e.target.value) || 0 }
                          : ds
                      )
                    }))
                  }
                />
              </Field>
              <CheckboxField
                label="Shuffle caption tokens"
                checked={ds0.shuffle_tokens}
                disabled={training}
                onChange={(v) =>
                  patch((p) => ({
                    ...p,
                    datasets: p.datasets.map((ds, i) =>
                      i === 0 ? { ...ds, shuffle_tokens: v } : ds
                    )
                  }))
                }
              />
              <CheckboxField
                label="Cache latents to disk"
                checked={ds0.cache_latents_to_disk}
                disabled={training}
                onChange={(v) =>
                  patch((p) => ({
                    ...p,
                    datasets: p.datasets.map((ds, i) =>
                      i === 0 ? { ...ds, cache_latents_to_disk: v } : ds
                    )
                  }))
                }
              />
            </div>
          )}

          {activeSection === 'train' && (
            <div className="lora-grid">
              <Field label="Training steps">
                <input
                  type="number"
                  min={1}
                  value={draft.train.steps}
                  disabled={training}
                  onChange={(e) =>
                    patch((p) => ({
                      ...p,
                      train: { ...p.train, steps: Number(e.target.value) || 1 }
                    }))
                  }
                />
              </Field>
              <Field label="Batch size">
                <input
                  type="number"
                  min={1}
                  value={draft.train.batch_size}
                  disabled={training}
                  onChange={(e) =>
                    patch((p) => ({
                      ...p,
                      train: { ...p.train, batch_size: Number(e.target.value) || 1 }
                    }))
                  }
                />
              </Field>
              <Field label="Learning rate">
                <input
                  type="number"
                  min={0}
                  step="any"
                  value={draft.train.lr}
                  disabled={training}
                  onChange={(e) =>
                    patch((p) => ({
                      ...p,
                      train: { ...p.train, lr: Number(e.target.value) || 0 }
                    }))
                  }
                />
              </Field>
              <Field label="Optimizer">
                <select
                  value={draft.train.optimizer}
                  disabled={training}
                  onChange={(e) =>
                    patch((p) => ({
                      ...p,
                      train: { ...p.train, optimizer: e.target.value }
                    }))
                  }
                >
                  <option value="adamw8bit">adamw8bit</option>
                  <option value="adamw">adamw</option>
                  <option value="prodigy">prodigy</option>
                  <option value="adafactor">adafactor</option>
                </select>
              </Field>
              <Field label="Precision (dtype)">
                <select
                  value={draft.train.dtype}
                  disabled={training}
                  onChange={(e) =>
                    patch((p) => ({
                      ...p,
                      train: { ...p.train, dtype: e.target.value }
                    }))
                  }
                >
                  <option value="bf16">bf16</option>
                  <option value="fp16">fp16</option>
                  <option value="fp32">fp32</option>
                </select>
              </Field>
              <Field label="Gradient accumulation">
                <input
                  type="number"
                  min={1}
                  value={draft.train.gradient_accumulation_steps}
                  disabled={training}
                  onChange={(e) =>
                    patch((p) => ({
                      ...p,
                      train: {
                        ...p.train,
                        gradient_accumulation_steps: Number(e.target.value) || 1
                      }
                    }))
                  }
                />
              </Field>
              <Field label="Noise scheduler" hint="Krea 2 uses flow matching">
                <select
                  value={draft.train.noise_scheduler}
                  disabled={training}
                  onChange={(e) =>
                    patch((p) => ({
                      ...p,
                      train: { ...p.train, noise_scheduler: e.target.value }
                    }))
                  }
                >
                  <option value="flowmatch">flowmatch</option>
                  <option value="ddpm">ddpm</option>
                  <option value="euler">euler</option>
                </select>
              </Field>
              <CheckboxField
                label="Gradient checkpointing"
                hint="Uses less VRAM; slightly slower"
                checked={draft.train.gradient_checkpointing}
                disabled={training}
                onChange={(v) =>
                  patch((p) => ({
                    ...p,
                    train: { ...p.train, gradient_checkpointing: v }
                  }))
                }
              />
            </div>
          )}

          {activeSection === 'lora' && (
            <div className="lora-grid">
              <Field label="Network type">
                <select
                  value={draft.network.type}
                  disabled={training}
                  onChange={(e) =>
                    patch((p) => ({
                      ...p,
                      network: { ...p.network, type: e.target.value }
                    }))
                  }
                >
                  <option value="lora">lora</option>
                  <option value="locon">locon</option>
                  <option value="lokr">lokr</option>
                </select>
              </Field>
              <Field label="Rank" hint="linear — higher = more capacity / VRAM">
                <input
                  type="number"
                  min={1}
                  value={draft.network.linear}
                  disabled={training}
                  onChange={(e) =>
                    patch((p) => ({
                      ...p,
                      network: { ...p.network, linear: Number(e.target.value) || 1 }
                    }))
                  }
                />
              </Field>
              <Field label="Alpha" hint="linear_alpha — often equal to rank">
                <input
                  type="number"
                  min={1}
                  value={draft.network.linear_alpha}
                  disabled={training}
                  onChange={(e) =>
                    patch((p) => ({
                      ...p,
                      network: {
                        ...p.network,
                        linear_alpha: Number(e.target.value) || 1
                      }
                    }))
                  }
                />
              </Field>
            </div>
          )}

          {activeSection === 'sample' && (
            <>
              <div className="lora-grid">
                <CheckboxField
                  label="Enable sampling during training"
                  hint="Off by default (faster). Samples use Apply-on (Turbo) when supported."
                  checked={!draft.train.disable_sampling}
                  disabled={training}
                  onChange={(v) =>
                    patch((p) => ({
                      ...p,
                      train: { ...p.train, disable_sampling: !v }
                    }))
                  }
                />
                <CheckboxField
                  label="Skip first sample"
                  checked={draft.train.skip_first_sample}
                  disabled={training || draft.train.disable_sampling}
                  onChange={(v) =>
                    patch((p) => ({
                      ...p,
                      train: { ...p.train, skip_first_sample: v }
                    }))
                  }
                />
                <Field label="Sample every N steps">
                  <input
                    type="number"
                    min={1}
                    value={draft.sample.sample_every}
                    disabled={training || draft.train.disable_sampling}
                    onChange={(e) =>
                      patch((p) => ({
                        ...p,
                        sample: {
                          ...p.sample,
                          sample_every: Number(e.target.value) || 1
                        }
                      }))
                    }
                  />
                </Field>
                <Field label="Width">
                  <input
                    type="number"
                    min={64}
                    step={64}
                    value={draft.sample.width}
                    disabled={training || draft.train.disable_sampling}
                    onChange={(e) =>
                      patch((p) => ({
                        ...p,
                        sample: { ...p.sample, width: Number(e.target.value) || 64 }
                      }))
                    }
                  />
                </Field>
                <Field label="Height">
                  <input
                    type="number"
                    min={64}
                    step={64}
                    value={draft.sample.height}
                    disabled={training || draft.train.disable_sampling}
                    onChange={(e) =>
                      patch((p) => ({
                        ...p,
                        sample: { ...p.sample, height: Number(e.target.value) || 64 }
                      }))
                    }
                  />
                </Field>
                <Field label="Sample steps" hint="Turbo: typically 8">
                  <input
                    type="number"
                    min={1}
                    value={draft.sample.sample_steps}
                    disabled={training || draft.train.disable_sampling}
                    onChange={(e) =>
                      patch((p) => ({
                        ...p,
                        sample: {
                          ...p.sample,
                          sample_steps: Number(e.target.value) || 1
                        }
                      }))
                    }
                  />
                </Field>
                <Field label="Guidance" hint="Turbo: often 0">
                  <input
                    type="number"
                    step="any"
                    value={draft.sample.guidance_scale}
                    disabled={training || draft.train.disable_sampling}
                    onChange={(e) =>
                      patch((p) => ({
                        ...p,
                        sample: {
                          ...p.sample,
                          guidance_scale: Number(e.target.value) || 0
                        }
                      }))
                    }
                  />
                </Field>
                <Field label="Seed">
                  <input
                    type="number"
                    value={draft.sample.seed}
                    disabled={training || draft.train.disable_sampling}
                    onChange={(e) =>
                      patch((p) => ({
                        ...p,
                        sample: { ...p.sample, seed: Number(e.target.value) || 0 }
                      }))
                    }
                  />
                </Field>
                <CheckboxField
                  label="Walk seed"
                  checked={draft.sample.walk_seed}
                  disabled={training || draft.train.disable_sampling}
                  onChange={(v) =>
                    patch((p) => ({
                      ...p,
                      sample: { ...p.sample, walk_seed: v }
                    }))
                  }
                />
              </div>
              <Field label="Sample prompts" hint="One prompt per line">
                <textarea
                  className="prompt-textarea lora-prompts"
                  value={draft.sample.prompts.join('\n')}
                  disabled={training || draft.train.disable_sampling}
                  onChange={(e) =>
                    patch((p) => ({
                      ...p,
                      sample: {
                        ...p.sample,
                        prompts: e.target.value.split('\n')
                      }
                    }))
                  }
                  spellCheck={false}
                />
              </Field>
            </>
          )}

          {activeSection === 'advanced' && (
            <div className="lora-grid">
              <Field label="Save every N steps">
                <input
                  type="number"
                  min={1}
                  value={draft.save.save_every}
                  disabled={training}
                  onChange={(e) =>
                    patch((p) => ({
                      ...p,
                      save: { ...p.save, save_every: Number(e.target.value) || 1 }
                    }))
                  }
                />
              </Field>
              <Field label="Max checkpoints to keep">
                <input
                  type="number"
                  min={1}
                  value={draft.save.max_step_saves_to_keep}
                  disabled={training}
                  onChange={(e) =>
                    patch((p) => ({
                      ...p,
                      save: {
                        ...p.save,
                        max_step_saves_to_keep: Number(e.target.value) || 1
                      }
                    }))
                  }
                />
              </Field>
              <Field label="Save weights dtype">
                <select
                  value={draft.save.dtype}
                  disabled={training}
                  onChange={(e) =>
                    patch((p) => ({
                      ...p,
                      save: { ...p.save, dtype: e.target.value }
                    }))
                  }
                >
                  <option value="float16">float16</option>
                  <option value="bf16">bf16</option>
                  <option value="float32">float32</option>
                </select>
              </Field>
              <CheckboxField
                label="Push to Hugging Face Hub"
                checked={draft.save.push_to_hub}
                disabled={training}
                onChange={(v) =>
                  patch((p) => ({ ...p, save: { ...p.save, push_to_hub: v } }))
                }
              />
              <CheckboxField
                label="Use EMA"
                checked={draft.train.ema_config.use_ema}
                disabled={training}
                onChange={(v) =>
                  patch((p) => ({
                    ...p,
                    train: {
                      ...p.train,
                      ema_config: { ...p.train.ema_config, use_ema: v }
                    }
                  }))
                }
              />
              <Field label="EMA decay">
                <input
                  type="number"
                  min={0}
                  max={1}
                  step={0.01}
                  value={draft.train.ema_config.ema_decay}
                  disabled={training}
                  onChange={(e) =>
                    patch((p) => ({
                      ...p,
                      train: {
                        ...p.train,
                        ema_config: {
                          ...p.train.ema_config,
                          ema_decay: Number(e.target.value) || 0
                        }
                      }
                    }))
                  }
                />
              </Field>
              <CheckboxField
                label="Train transformer / UNet"
                checked={draft.train.train_unet}
                disabled={training}
                onChange={(v) =>
                  patch((p) => ({ ...p, train: { ...p.train, train_unet: v } }))
                }
              />
              <CheckboxField
                label="Train text encoder"
                hint="Usually off for Krea 2"
                checked={draft.train.train_text_encoder}
                disabled={training}
                onChange={(v) =>
                  patch((p) => ({
                    ...p,
                    train: { ...p.train, train_text_encoder: v }
                  }))
                }
              />
              <CheckboxField
                label="Quantize"
                checked={draft.model.quantize}
                disabled={training}
                onChange={(v) =>
                  patch((p) => ({ ...p, model: { ...p.model, quantize: v } }))
                }
              />
              <CheckboxField
                label="Low VRAM mode"
                checked={draft.model.low_vram}
                disabled={training}
                onChange={(v) =>
                  patch((p) => ({ ...p, model: { ...p.model, low_vram: v } }))
                }
              />
            </div>
          )}
        </div>

        <ResourceMonitorPane device={draft.device} />
      </div>
    </div>
  )
}
