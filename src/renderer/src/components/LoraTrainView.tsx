import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react'
import type { LoraTrainAppSettings, LoraTrainJobConfig } from '../types'
import { KREA2_RAW, normalizeLoraTrainJob } from '../types'
import { serializeTrainConfig } from '../services/trainConfig'
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

function formatBytes(n: number): string {
  if (!Number.isFinite(n) || n < 0) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let v = n
  let i = 0
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i += 1
  }
  const digits = i === 0 ? 0 : v >= 10 ? 1 : 2
  return `${v.toFixed(digits)} ${units[i]}`
}

function downloadInputLabel(pct: number, done: number, total: number): string {
  if (total > 0) {
    return `Downloading … ${formatBytes(done)} / ${formatBytes(total)} ${pct}%`
  }
  return `Downloading … ${pct}%`
}

function folderLabel(dir: string): string {
  return dir.split(/[/\\]/).pop() ?? dir
}

const TRAIN_LOG_MAX_LINES = 2000
const LOSS_HISTORY_MAX_POINTS = 2000

/** Cumulative mean of loss values from step 0 through endIndex (inclusive). */
function cumulativeAvgLoss(points: { loss: number }[], endIndex: number): number {
  let sum = 0
  for (let i = 0; i <= endIndex; i++) sum += points[i].loss
  return sum / (endIndex + 1)
}

function formatHms(ms: number): string {
  const totalSec = Math.max(0, Math.floor(ms / 1000))
  const h = Math.floor(totalSec / 3600)
  const m = Math.floor((totalSec % 3600) / 60)
  const s = totalSec % 60
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

const SECTIONS: { id: SectionId; label: string; hint: string }[] = [
  { id: 'basics', label: 'Basics', hint: 'Name, output, models' },
  { id: 'dataset', label: 'Dataset', hint: 'Images + captions' },
  { id: 'train', label: 'Train', hint: 'Steps, LR, optimizer' },
  { id: 'lora', label: 'LoRA', hint: 'Rank & alpha' },
  { id: 'sample', label: 'Sample', hint: 'Preview during train' },
  { id: 'advanced', label: 'Advanced', hint: 'Save, EMA, VRAM' }
]

const RESOLUTION_OPTIONS = [256, 512, 768, 1024, 1280, 1328, 1536, 2048] as const
/** Rolling window for sec/step avg (number of step intervals). */
const SEC_PER_STEP_AVG_WINDOW = 10

interface Props {
  job: LoraTrainJobConfig
  appSettings: LoraTrainAppSettings
  datasetFolders: string[]
  onChange: (job: LoraTrainJobConfig) => void
  onStatus: (message: string, isError?: boolean, options?: { sticky?: boolean }) => void
  onTrainingChange?: (training: boolean) => void
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

function LossChart({
  points,
  showLoss = true,
  showAvgLoss = true
}: {
  points: { step: number; loss: number }[]
  showLoss?: boolean
  showAvgLoss?: boolean
}) {
  const wrapRef = useRef<HTMLDivElement | null>(null)
  const [size, setSize] = useState({ w: 640, h: 220 })

  useEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const update = () => {
      const r = el.getBoundingClientRect()
      setSize({
        w: Math.max(1, Math.floor(r.width)),
        h: Math.max(1, Math.floor(r.height))
      })
    }
    update()
    const ro = new ResizeObserver(update)
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const vbW = size.w
  const vbH = size.h
  const padL = 58
  const padR = 16
  const padT = 14
  const padB = 28
  const plotW = Math.max(1, vbW - padL - padR)
  const plotH = Math.max(1, vbH - padT - padB)

  if (points.length === 0) {
    return (
      <div className="lora-train-loss-chart-wrap" ref={wrapRef}>
        <svg
          className="lora-train-loss-chart"
          viewBox={`0 0 ${vbW} ${vbH}`}
          width={vbW}
          height={vbH}
          role="img"
          aria-label="Loss chart"
          preserveAspectRatio="none"
        >
          <rect
            x={padL}
            y={padT}
            width={plotW}
            height={plotH}
            className="lora-train-loss-plot-bg"
          />
        </svg>
        <div className="lora-train-loss-empty">Waiting for loss…</div>
      </div>
    )
  }

  const avgPoints: { step: number; loss: number }[] = []
  for (let i = 0; i < points.length; i++) {
    avgPoints.push({
      step: points[i].step,
      loss: cumulativeAvgLoss(points, i)
    })
  }

  let minStep = points[0].step
  let maxStep = points[0].step
  let dataMaxLoss = 0
  let hasVisible = false
  for (const p of points) {
    if (p.step < minStep) minStep = p.step
    if (p.step > maxStep) maxStep = p.step
    if (showLoss && p.loss > dataMaxLoss) {
      dataMaxLoss = p.loss
      hasVisible = true
    }
  }
  if (showAvgLoss) {
    for (const p of avgPoints) {
      if (p.loss > dataMaxLoss) {
        dataMaxLoss = p.loss
        hasVisible = true
      }
    }
  }
  if (!hasVisible) dataMaxLoss = 0.1

  const stepSpan = Math.max(1, maxStep - minStep)
  const lossMin = 0
  // Ceil to next 0.x00 (one decimal), e.g. 0.776 → 0.800, 0.258 → 0.300
  const lossMax = Math.max(0.1, Math.ceil(dataMaxLoss * 10 - 1e-12) / 10)
  const lossSpan = Math.max(lossMax - lossMin, 1e-9)

  const toX = (step: number) => padL + ((step - minStep) / stepSpan) * plotW
  const toY = (loss: number) => padT + (1 - (loss - lossMin) / lossSpan) * plotH

  const polyline = points.map((p) => `${toX(p.step).toFixed(2)},${toY(p.loss).toFixed(2)}`).join(' ')
  const avgPolyline = avgPoints
    .map((p) => `${toX(p.step).toFixed(2)},${toY(p.loss).toFixed(2)}`)
    .join(' ')
  const last = points[points.length - 1]
  const lastAvg = avgPoints[avgPoints.length - 1]
  const lastX = toX(last.step)
  const lastY = toY(last.loss)
  const lastAvgX = toX(lastAvg.step)
  const lastAvgY = toY(lastAvg.loss)

  const yTicks: number[] = []
  for (let i = 0; i <= 10; i++) {
    yTicks.push(lossMax - (i / 10) * (lossMax - lossMin))
  }
  const xTicks: number[] = []
  for (let i = 0; i <= 10; i++) {
    xTicks.push(Math.round(minStep + (i / 10) * (maxStep - minStep)))
  }

  return (
    <div className="lora-train-loss-chart-wrap" ref={wrapRef}>
      <svg
        className="lora-train-loss-chart"
        viewBox={`0 0 ${vbW} ${vbH}`}
        width={vbW}
        height={vbH}
        role="img"
        aria-label={`Loss chart, latest ${last.loss.toFixed(4)}, avg ${lastAvg.loss.toFixed(4)} at step ${last.step}`}
        preserveAspectRatio="none"
      >
        <rect x={padL} y={padT} width={plotW} height={plotH} className="lora-train-loss-plot-bg" />
        {yTicks.map((v, i) => {
          const y = toY(v)
          return (
            <g key={`y-${i}`}>
              <line
                x1={padL}
                y1={y}
                x2={padL + plotW}
                y2={y}
                className="lora-train-loss-grid"
              />
              <text x={padL - 8} y={y + 3} textAnchor="end" className="lora-train-loss-axis">
                {v.toFixed(3)}
              </text>
            </g>
          )
        })}
        {xTicks.map((v, i) => (
          <text
            key={`x-${i}`}
            x={toX(v)}
            y={padT + plotH + 18}
            textAnchor="middle"
            className="lora-train-loss-axis"
          >
            {v}
          </text>
        ))}
        {showLoss ? (
          <>
            <polyline points={polyline} fill="none" className="lora-train-loss-line" />
            <circle cx={lastX} cy={lastY} r={3.5} className="lora-train-loss-dot" />
          </>
        ) : null}
        {showAvgLoss ? (
          <>
            <polyline points={avgPolyline} fill="none" className="lora-train-loss-line-avg" />
            <circle cx={lastAvgX} cy={lastAvgY} r={3} className="lora-train-loss-dot-avg" />
          </>
        ) : null}
      </svg>
    </div>
  )
}

function ToggleField({
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
    <div className="field">
      <div
        className={`lora-toggle${checked ? ' is-on' : ''}${disabled ? ' is-disabled' : ''}`}
      >
        <button
          type="button"
          role="switch"
          className="lora-switch"
          aria-checked={checked}
          aria-label={label}
          disabled={disabled}
          onClick={() => onChange(!checked)}
        >
          <span className="lora-switch-knob" aria-hidden="true" />
        </button>
        <span className="lora-toggle-label">{label}</span>
      </div>
      {hint ? <p className="field-hint">{hint}</p> : null}
    </div>
  )
}

/** Allows clearing the field while editing; applies emptyFallback only on blur. */
function NumberField({
  value,
  emptyFallback,
  onChange,
  disabled,
  min,
  max,
  step
}: {
  value: number
  emptyFallback: number
  onChange: (n: number) => void
  disabled?: boolean
  min?: number
  max?: number
  step?: number | 'any'
}) {
  const [text, setText] = useState(() => String(value))
  const focusedRef = useRef(false)

  useEffect(() => {
    if (!focusedRef.current) {
      setText(String(value))
    }
  }, [value])

  const commit = (raw: string) => {
    const trimmed = raw.trim()
    if (trimmed === '' || !Number.isFinite(Number(trimmed))) {
      setText(String(emptyFallback))
      onChange(emptyFallback)
      return
    }
    const n = Number(trimmed)
    setText(String(n))
    onChange(n)
  }

  return (
    <input
      type="number"
      min={min}
      max={max}
      step={step}
      disabled={disabled}
      value={text}
      onFocus={() => {
        focusedRef.current = true
      }}
      onChange={(e) => {
        const next = e.target.value
        setText(next)
        if (next.trim() === '') return
        const n = Number(next)
        if (Number.isFinite(n)) onChange(n)
      }}
      onBlur={() => {
        focusedRef.current = false
        commit(text)
      }}
    />
  )
}

export function LoraTrainView({
  job,
  appSettings,
  datasetFolders,
  onChange,
  onStatus,
  onTrainingChange
}: Props) {
  const [draft, setDraft] = useState(() => normalizeLoraTrainJob(job))
  const [activeSection, setActiveSection] = useState<SectionId>('basics')
  const [training, setTraining] = useState(false)
  /** Results UI (status / loss / log) from Start until Back or Stop/error. */
  const [showTrainSession, setShowTrainSession] = useState(false)
  const [trainComplete, setTrainComplete] = useState(false)
  const [progress, setProgress] = useState<{
    step: number
    total: number
    loss: number
  } | null>(null)
  const [trainLogs, setTrainLogs] = useState<string[]>([])
  const [lossHistory, setLossHistory] = useState<{ step: number; loss: number }[]>([])
  const [showLossCurve, setShowLossCurve] = useState(true)
  const [showAvgLossCurve, setShowAvgLossCurve] = useState(true)
  const [trainStartedAt, setTrainStartedAt] = useState<number | null>(null)
  const [finalElapsedMs, setFinalElapsedMs] = useState<number | null>(null)
  const [nowTick, setNowTick] = useState(() => Date.now())
  const [secPerStepAvg, setSecPerStepAvg] = useState<number | null>(null)
  const stepTimingRef = useRef<{ step: number; at: number }[]>([])
  const trainStartedAtRef = useRef<number | null>(null)
  const logEndRef = useRef<HTMLSpanElement | null>(null)
  const [modelStatuses, setModelStatuses] = useState<ModelStatusItem[]>([])
  const [modelChecking, setModelChecking] = useState(false)
  const [modelCheckError, setModelCheckError] = useState<string | null>(null)
  const [dlCurrent, setDlCurrent] = useState<string | null>(null)
  const [dlPct, setDlPct] = useState(0)
  const [dlDone, setDlDone] = useState(0)
  const [dlTotal, setDlTotal] = useState(0)
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
  }, [job])

  useEffect(() => {
    onTrainingChange?.(training)
  }, [training, onTrainingChange])

  const resetStepTiming = useCallback(() => {
    stepTimingRef.current = []
    setSecPerStepAvg(null)
  }, [])

  const recordStepTiming = useCallback((step: number) => {
    const now = Date.now()
    const hist = stepTimingRef.current
    hist.push({ step, at: now })
    while (hist.length > SEC_PER_STEP_AVG_WINDOW + 1) hist.shift()

    if (hist.length < 2) {
      setSecPerStepAvg(null)
      return
    }

    const samples: number[] = []
    for (let i = 1; i < hist.length; i++) {
      const dStep = hist[i].step - hist[i - 1].step
      const dMs = hist[i].at - hist[i - 1].at
      if (dStep > 0 && dMs > 0) samples.push(dMs / 1000 / dStep)
    }
    setSecPerStepAvg(
      samples.length > 0 ? samples.reduce((a, b) => a + b, 0) / samples.length : null
    )
  }, [])

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
        targets: [{ role: 'train', path: d.model.train_name_or_path }]
      })
      const results = result.results || []
      setModelStatuses(results)
      if (!result.ok && result.error) {
        setModelCheckError(result.error)
      }

      // If HF id still in job but a complete snapshot exists under download path, point at local dir.
      let trainPath = d.model.train_name_or_path
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
      }
      if (changed) {
        setDraft((prev) => ({
          ...prev,
          model: {
            ...prev.model,
            train_name_or_path: trainPath,
            name_or_path: trainPath
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
      for (const st of modelStatusesRef.current) {
        if (st.repoId === repoId || st.path === repoId) {
          if (st.role === 'train') trainPath = localPath
        }
      }
      if (trainPath === repoId) trainPath = localPath
      return {
        ...prev,
        model: {
          ...prev.model,
          train_name_or_path: trainPath,
          name_or_path: trainPath
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
      setDlDone(0)
      setDlTotal(0)
      return
    }
    downloadingRef.current = true
    setDlCurrent(next)
    setDlPct(0)
    setDlDone(0)
    setDlTotal(0)
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
      setDlCurrent(null)
      setDlPct(0)
      setDlDone(0)
      setDlTotal(0)
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
      void pumpDownloadQueue()
    },
    [pumpDownloadQueue]
  )

  useEffect(() => {
    void window.api.trainStatus().then((s) => {
      setTraining(s.running)
      if (s.running) {
        setTrainStartedAt((prev) => prev ?? Date.now())
        onStatus('Training…', false, { sticky: true })
      }
    })
    const offProg = window.api.onTrainProgress((p) => {
      setProgress(p)
      recordStepTiming(p.step)
      if (typeof p.loss === 'number' && Number.isFinite(p.loss)) {
        setLossHistory((prev) => {
          const next =
            prev.length >= LOSS_HISTORY_MAX_POINTS
              ? prev.slice(-LOSS_HISTORY_MAX_POINTS + 1)
              : prev.slice()
          next.push({ step: p.step, loss: p.loss })
          return next
        })
      }
      const loss =
        typeof p.loss === 'number' && Number.isFinite(p.loss) ? p.loss.toFixed(4) : null
      onStatus(
        loss
          ? `Training step ${p.step}/${p.total} · loss=${loss}`
          : `Training step ${p.step}/${p.total}`,
        false,
        { sticky: true }
      )
    })
    const offLog = window.api.onTrainLog(({ line }) => {
      setTrainLogs((prev) => {
        const next = prev.length >= TRAIN_LOG_MAX_LINES ? prev.slice(-TRAIN_LOG_MAX_LINES + 1) : prev.slice()
        next.push(line)
        return next
      })
    })
    const offDone = window.api.onTrainDone(({ path }) => {
      const started = trainStartedAtRef.current
      const elapsed = started !== null ? Math.max(0, Date.now() - started) : 0
      setFinalElapsedMs(elapsed)
      setTrainStartedAt(null)
      trainStartedAtRef.current = null
      setTraining(false)
      setTrainComplete(true)
      onStatus(`Training done: ${path}`)
    })
    const offErr = window.api.onTrainError(({ message }) => {
      setTraining(false)
      setShowTrainSession(false)
      setTrainComplete(false)
      setFinalElapsedMs(null)
      setTrainStartedAt(null)
      trainStartedAtRef.current = null
      resetStepTiming()
      setLossHistory([])
      setTrainLogs([])
      setProgress(null)
      onStatus(message, true)
    })
    const offDlProg = window.api.onModelDownloadProgress(({ repoId, pct, done, total }) => {
      setDlCurrent(repoId)
      setDlPct(pct)
      if (typeof done === 'number') setDlDone(done)
      if (typeof total === 'number') setDlTotal(total)
      const pctLabel = Number.isFinite(pct) ? `${Math.round(pct)}%` : ''
      onStatus(
        pctLabel ? `Downloading ${repoId} ${pctLabel}` : `Downloading ${repoId}…`,
        false,
        { sticky: true }
      )
    })
    const offDlDone = window.api.onModelDownloadDone(({ repoId, path }) => {
      applyDownloadedPath(repoId, path)
      onStatus(`Model ready: ${repoId}`)
      dlQueueRef.current = dlQueueRef.current.filter((id) => id !== repoId)
      downloadingRef.current = false
      if (dlQueueRef.current.length === 0) {
        setDlCurrent(null)
        setDlPct(0)
        setDlDone(0)
        setDlTotal(0)
        void refreshModelStatus()
      } else {
        void pumpDownloadQueue()
      }
    })
    const offDlErr = window.api.onModelDownloadError(({ message }) => {
      downloadingRef.current = false
      dlQueueRef.current = []
      setDlCurrent(null)
      setDlPct(0)
      setDlDone(0)
      setDlTotal(0)
      onStatus(message, true)
      void refreshModelStatus()
    })
    return () => {
      offProg()
      offLog()
      offDone()
      offErr()
      offDlProg()
      offDlDone()
      offDlErr()
    }
  }, [onStatus, applyDownloadedPath, pumpDownloadQueue, refreshModelStatus, recordStepTiming, resetStepTiming])

  useEffect(() => {
    if (!training) return
    setNowTick(Date.now())
    const id = setInterval(() => setNowTick(Date.now()), 1000)
    return () => clearInterval(id)
  }, [training])

  useEffect(() => {
    if (!training) return
    logEndRef.current?.scrollIntoView({ block: 'end' })
  }, [trainLogs, training])

  useEffect(() => {
    const timer = setTimeout(() => {
      void refreshModelStatus()
    }, 600)
    return () => clearTimeout(timer)
  }, [
    draft.model.train_name_or_path,
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
  // Top banner only for model check errors (download progress lives in the model row)
  const showBanner = Boolean(modelCheckError)

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
    setDlCurrent(null)
    setDlPct(0)
    setDlDone(0)
    setDlTotal(0)
    onStatus('Model download cancelled')
    void refreshModelStatus()
  }

  const isRoleDownloading = (role: string): boolean => {
    const repo = resolveDownloadRepoId(statusForRole(role))
    return Boolean(dlCurrent && repo && dlCurrent === repo)
  }

  const renderModelDlButton = (role: 'train') => {
    const st = statusForRole(role)
    const repo = resolveDownloadRepoId(st)
    const downloading = isRoleDownloading(role)
    if (downloading) {
      return (
        <button
          type="button"
          className="danger lora-dl-btn"
          disabled={training}
          onClick={() => void cancelDownload()}
        >
          Cancel
        </button>
      )
    }
    if (!repo) return null
    if (st?.status === 'missing') {
      return (
        <button
          type="button"
          className="primary lora-dl-btn"
          disabled={training || isDownloading}
          onClick={() => downloadForRole(role)}
        >
          Download
        </button>
      )
    }
    if (st?.status === 'updateAvailable') {
      return (
        <button
          type="button"
          className="primary lora-dl-btn"
          disabled={training || isDownloading}
          onClick={() => downloadForRole(role)}
        >
          Update
        </button>
      )
    }
    return null
  }

  const browseTrainingFolder = async () => {
    const dir = await window.api.openFolder()
    if (!dir) return
    patch((prev) => ({ ...prev, training_folder: dir }))
  }

  const clearTrainSessionUi = () => {
    setShowTrainSession(false)
    setTrainComplete(false)
    setFinalElapsedMs(null)
    setTrainStartedAt(null)
    trainStartedAtRef.current = null
    resetStepTiming()
    setLossHistory([])
    setTrainLogs([])
    setProgress(null)
  }

  const startTrain = async () => {
    const normalized = normalizeLoraTrainJob(draftRef.current)
    if (!normalized.datasets[0]?.folder_path) {
      setActiveSection('dataset')
      onStatus('Choose a DatasetEdit preset first', true)
      return
    }
    if (normalized.model.arch !== 'krea2') {
      onStatus('Only Krea 2 is supported', true)
      return
    }
    const py = appSettings.pythonPath.trim()
    setProgress(null)
    setTrainLogs([])
    setLossHistory([])
    resetStepTiming()
    const started = Date.now()
    setTrainStartedAt(started)
    trainStartedAtRef.current = started
    setFinalElapsedMs(null)
    setTrainComplete(false)
    setNowTick(started)
    setShowTrainSession(true)
    setTraining(true)
    onStatus('Training started…', false, { sticky: true })
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
      clearTrainSessionUi()
      onStatus(result.error || 'Failed to start training', true)
    }
  }

  const stopTrain = async () => {
    await window.api.stopTrain()
    setTraining(false)
    clearTrainSessionUi()
    onStatus('Training stopped')
  }

  const backFromTrainSession = () => {
    clearTrainSessionUi()
  }

  const elapsedMs =
    finalElapsedMs !== null
      ? finalElapsedMs
      : trainStartedAt !== null
        ? Math.max(0, nowTick - trainStartedAt)
        : 0
  const etaMs = trainComplete
    ? 0
    : progress && progress.step > 0 && progress.total > progress.step && elapsedMs > 0
      ? ((progress.total - progress.step) * elapsedMs) / progress.step
      : null
  const stepsLabel = progress
    ? `${progress.step}/${progress.total}`
    : `0/${draft.train.steps || 0}`

  const sectionTitle = SECTIONS.find((s) => s.id === activeSection)?.label ?? ''

  return (
    <div className="lora-train">
      {(showBanner) && (
        <div className="lora-model-banner" role="status">
          <p>Model check failed: {modelCheckError}</p>
          <div className="lora-model-banner-actions">
            <button type="button" onClick={() => void refreshModelStatus()} disabled={modelChecking}>
              {modelChecking ? 'Checking…' : 'Retry'}
            </button>
          </div>
        </div>
      )}

      <div className="lora-train-body">
        <nav className="lora-nav" aria-label="Training settings sections">
          {showTrainSession ? (
            <div className="lora-train-status">
              <p className="lora-train-status-title">
                {trainComplete ? 'Training done' : 'Training...'}
              </p>
              <div className="lora-train-status-steps-block">
                <p className="lora-train-status-steps">{stepsLabel} Steps</p>
                <p className="lora-train-status-avg">
                  {secPerStepAvg !== null
                    ? `${secPerStepAvg.toFixed(1)} sec/step avg.`
                    : '--.- sec/step avg.'}
                </p>
              </div>
              <p className="lora-train-status-line">Elapsed: {formatHms(elapsedMs)}</p>
              <p className="lora-train-status-line">
                ETA: {etaMs !== null ? formatHms(etaMs) : '--:--:--'}
              </p>
            </div>
          ) : (
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
                Required: DatasetEdit preset + Train base (Raw)
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
          )}
          <div className="lora-nav-actions">
            {training ? (
              <button type="button" className="danger" onClick={() => void stopTrain()}>
                Stop
              </button>
            ) : showTrainSession && trainComplete ? (
              <button type="button" className="primary" onClick={backFromTrainSession}>
                Back
              </button>
            ) : (
              <button type="button" className="primary" onClick={() => void startTrain()}>
                Start Train
              </button>
            )}
          </div>
        </nav>

        {showTrainSession ? (
          <div className="lora-train-mid">
            <div className="lora-train-loss-panel">
              <div className="lora-train-loss-header">
                <h3 className="lora-panel-title">Loss</h3>
                <div className="lora-train-loss-legend">
                  <label
                    className={`lora-train-loss-legend-item${showLossCurve ? '' : ' is-off'}`}
                  >
                    <input
                      type="checkbox"
                      className="lora-train-loss-check"
                      checked={showLossCurve}
                      onChange={(e) => setShowLossCurve(e.target.checked)}
                      aria-label="Show loss curve"
                    />
                    <span className="lora-train-loss-swatch is-raw" />
                    loss
                    {lossHistory.length > 0
                      ? `=${lossHistory[lossHistory.length - 1].loss.toFixed(4)}`
                      : ''}
                  </label>
                  <label
                    className={`lora-train-loss-legend-item${showAvgLossCurve ? '' : ' is-off'}`}
                  >
                    <input
                      type="checkbox"
                      className="lora-train-loss-check"
                      checked={showAvgLossCurve}
                      onChange={(e) => setShowAvgLossCurve(e.target.checked)}
                      aria-label="Show avg. loss curve"
                    />
                    <span className="lora-train-loss-swatch is-avg" />
                    avg. loss
                    {lossHistory.length > 0
                      ? `=${cumulativeAvgLoss(lossHistory, lossHistory.length - 1).toFixed(4)}`
                      : ''}
                  </label>
                </div>
              </div>
              <LossChart
                points={lossHistory}
                showLoss={showLossCurve}
                showAvgLoss={showAvgLossCurve}
              />
            </div>
            <div className="lora-train-log-panel">
              <h3 className="lora-panel-title">Log</h3>
              <pre className="lora-train-log">
                {trainLogs.length > 0 ? trainLogs.join('\n') : 'Waiting for trainer output…'}
                <span ref={logEndRef} />
              </pre>
            </div>
          </div>
        ) : (
        <div className="lora-train-panel">
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
                    value={
                      isRoleDownloading('train')
                        ? downloadInputLabel(dlPct, dlDone, dlTotal)
                        : draft.model.train_name_or_path
                    }
                    disabled={training || isRoleDownloading('train')}
                    readOnly={isRoleDownloading('train')}
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
                    disabled={training || isRoleDownloading('train')}
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
                  {renderModelDlButton('train')}
                </div>
              </Field>
            </div>
          )}

          {activeSection === 'dataset' && (
            <div className="lora-grid">
              <Field
                label="Dataset"
                hint="Select a dataset preset from DatasetEdit (images with matching .txt captions)"
              >
                <select
                  className="lora-folder-pick"
                  aria-label="DatasetEdit preset"
                  disabled={training || (datasetFolders.length === 0 && !ds0.folder_path)}
                  value={ds0.folder_path || ''}
                  onChange={(e) => {
                    const v = e.target.value
                    patch((p) => ({
                      ...p,
                      datasets: p.datasets.map((ds, i) =>
                        i === 0 ? { ...ds, folder_path: v } : ds
                      )
                    }))
                  }}
                  title={ds0.folder_path || undefined}
                >
                  <option value="">
                    {datasetFolders.length === 0
                      ? 'No DatasetEdit presets — add one in DatasetEdit'
                      : 'Select DatasetEdit preset…'}
                  </option>
                  {ds0.folder_path && !datasetFolders.includes(ds0.folder_path) ? (
                    <option value={ds0.folder_path}>
                      {folderLabel(ds0.folder_path)} (not in DatasetEdit)
                    </option>
                  ) : null}
                  {datasetFolders.map((dir) => (
                    <option key={dir} value={dir} title={dir}>
                      {folderLabel(dir)}
                    </option>
                  ))}
                </select>
              </Field>
              <Field
                label="Resolutions"
                hint="Toggle one or more training sizes"
              >
                <div className="lora-resolution-grid" role="group" aria-label="Resolutions">
                  {RESOLUTION_OPTIONS.map((size) => {
                    const on = ds0.resolution.includes(size)
                    return (
                      <div
                        key={size}
                        className={`lora-toggle lora-resolution-toggle${on ? ' is-on' : ''}${
                          training ? ' is-disabled' : ''
                        }`}
                      >
                        <button
                          type="button"
                          role="switch"
                          className="lora-switch"
                          aria-checked={on}
                          aria-label={`${size}`}
                          disabled={training}
                          onClick={() => {
                            patch((p) => ({
                              ...p,
                              datasets: p.datasets.map((ds, i) => {
                                if (i !== 0) return ds
                                const has = ds.resolution.includes(size)
                                if (has) {
                                  const next = ds.resolution.filter((r) => r !== size)
                                  return {
                                    ...ds,
                                    resolution: next.length > 0 ? next : [size]
                                  }
                                }
                                return {
                                  ...ds,
                                  resolution: [...ds.resolution, size].sort((a, b) => a - b)
                                }
                              })
                            }))
                          }}
                        >
                          <span className="lora-switch-knob" aria-hidden="true" />
                        </button>
                        <span className="lora-resolution-value">{size}</span>
                      </div>
                    )
                  })}
                </div>
              </Field>
              <Field label="Caption dropout" hint="0–1; randomly blank captions during training">
                <NumberField
                  min={0}
                  max={1}
                  step={0.01}
                  value={ds0.caption_dropout_rate}
                  emptyFallback={0}
                  disabled={training}
                  onChange={(n) =>
                    patch((p) => ({
                      ...p,
                      datasets: p.datasets.map((ds, i) =>
                        i === 0 ? { ...ds, caption_dropout_rate: n } : ds
                      )
                    }))
                  }
                />
              </Field>
              <ToggleField
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
              <ToggleField
                label="Cache latents to disk"
                hint="Encode images once with VAE; required on ~24GB GPUs"
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
              <ToggleField
                label="Cache text embeddings to disk"
                hint="Encode captions once with text encoder; required on ~24GB GPUs"
                checked={draft.train.cache_text_embeddings}
                disabled={training}
                onChange={(v) =>
                  patch((p) => ({
                    ...p,
                    train: { ...p.train, cache_text_embeddings: v }
                  }))
                }
              />
            </div>
          )}

          {activeSection === 'train' && (
            <div className="lora-grid">
              <Field label="Training steps">
                <NumberField
                  min={1}
                  value={draft.train.steps}
                  emptyFallback={1}
                  disabled={training}
                  onChange={(n) =>
                    patch((p) => ({
                      ...p,
                      train: { ...p.train, steps: n }
                    }))
                  }
                />
              </Field>
              <Field label="Batch size">
                <NumberField
                  min={1}
                  value={draft.train.batch_size}
                  emptyFallback={1}
                  disabled={training}
                  onChange={(n) =>
                    patch((p) => ({
                      ...p,
                      train: { ...p.train, batch_size: n }
                    }))
                  }
                />
              </Field>
              <Field label="Learning rate">
                <NumberField
                  min={0}
                  step="any"
                  value={draft.train.lr}
                  emptyFallback={0}
                  disabled={training}
                  onChange={(n) =>
                    patch((p) => ({
                      ...p,
                      train: { ...p.train, lr: n }
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
                <NumberField
                  min={1}
                  value={draft.train.gradient_accumulation_steps}
                  emptyFallback={1}
                  disabled={training}
                  onChange={(n) =>
                    patch((p) => ({
                      ...p,
                      train: {
                        ...p.train,
                        gradient_accumulation_steps: n
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
              <ToggleField
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
                <NumberField
                  min={1}
                  value={draft.network.linear}
                  emptyFallback={1}
                  disabled={training}
                  onChange={(n) =>
                    patch((p) => ({
                      ...p,
                      network: { ...p.network, linear: n }
                    }))
                  }
                />
              </Field>
              <Field label="Alpha" hint="linear_alpha — often equal to rank">
                <NumberField
                  min={1}
                  value={draft.network.linear_alpha}
                  emptyFallback={1}
                  disabled={training}
                  onChange={(n) =>
                    patch((p) => ({
                      ...p,
                      network: {
                        ...p.network,
                        linear_alpha: n
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
                <div className="lora-field-row">
                  <ToggleField
                    label="Enable sampling during training"
                    hint="Off by default (faster). Mid-train samples use Train-on (Raw) + LoRA."
                    checked={!draft.train.disable_sampling}
                    disabled={training}
                    onChange={(v) =>
                      patch((p) => ({
                        ...p,
                        train: { ...p.train, disable_sampling: !v }
                      }))
                    }
                  />
                  <ToggleField
                    label="Skip first sample"
                    hint="When off, sample once at step 0 before training starts."
                    checked={draft.train.skip_first_sample}
                    disabled={training || draft.train.disable_sampling}
                    onChange={(v) =>
                      patch((p) => ({
                        ...p,
                        train: { ...p.train, skip_first_sample: v }
                      }))
                    }
                  />
                </div>
                <Field label="Sample every N steps">
                  <NumberField
                    min={1}
                    value={draft.sample.sample_every}
                    emptyFallback={1}
                    disabled={training || draft.train.disable_sampling}
                    onChange={(n) =>
                      patch((p) => ({
                        ...p,
                        sample: {
                          ...p.sample,
                          sample_every: n
                        }
                      }))
                    }
                  />
                </Field>
                <Field label="Sample steps" hint="Denoise steps for mid-train preview (Raw)">
                  <NumberField
                    min={1}
                    value={draft.sample.sample_steps}
                    emptyFallback={1}
                    disabled={training || draft.train.disable_sampling}
                    onChange={(n) =>
                      patch((p) => ({
                        ...p,
                        sample: {
                          ...p.sample,
                          sample_steps: n
                        }
                      }))
                    }
                  />
                </Field>
                <Field label="Guidance" hint="CFG scale; 0 disables guidance">
                  <NumberField
                    step="any"
                    value={draft.sample.guidance_scale}
                    emptyFallback={0}
                    disabled={training || draft.train.disable_sampling}
                    onChange={(n) =>
                      patch((p) => ({
                        ...p,
                        sample: {
                          ...p.sample,
                          guidance_scale: n
                        }
                      }))
                    }
                  />
                </Field>
              </div>

              <div className="lora-prompt-groups">
                <div className="lora-prompt-groups-header">
                  <span>Sample prompts</span>
                  <p className="field-hint">
                    Each group has its own prompt, size, and seed
                  </p>
                </div>
                {draft.sample.prompts.map((item, index) => (
                  <div key={index} className="lora-prompt-group">
                    <label className="field">
                      <span>Prompt {index + 1}</span>
                      <input
                        type="text"
                        value={item.prompt}
                        disabled={training || draft.train.disable_sampling}
                        onChange={(e) => {
                          const value = e.target.value
                          patch((p) => ({
                            ...p,
                            sample: {
                              ...p.sample,
                              prompts: p.sample.prompts.map((pr, i) =>
                                i === index ? { ...pr, prompt: value } : pr
                              )
                            }
                          }))
                        }}
                        spellCheck={false}
                      />
                    </label>
                    <div className="lora-prompt-meta">
                      <label className="field">
                        <span>Width</span>
                        <NumberField
                          min={64}
                          step={64}
                          value={item.width}
                          emptyFallback={64}
                          disabled={training || draft.train.disable_sampling}
                          onChange={(n) => {
                            patch((p) => ({
                              ...p,
                              sample: {
                                ...p.sample,
                                prompts: p.sample.prompts.map((pr, i) =>
                                  i === index ? { ...pr, width: n } : pr
                                )
                              }
                            }))
                          }}
                        />
                      </label>
                      <label className="field">
                        <span>Height</span>
                        <NumberField
                          min={64}
                          step={64}
                          value={item.height}
                          emptyFallback={64}
                          disabled={training || draft.train.disable_sampling}
                          onChange={(n) => {
                            patch((p) => ({
                              ...p,
                              sample: {
                                ...p.sample,
                                prompts: p.sample.prompts.map((pr, i) =>
                                  i === index ? { ...pr, height: n } : pr
                                )
                              }
                            }))
                          }}
                        />
                      </label>
                      <label className="field">
                        <span>Seed</span>
                        <NumberField
                          value={item.seed}
                          emptyFallback={0}
                          disabled={training || draft.train.disable_sampling}
                          onChange={(n) => {
                            patch((p) => ({
                              ...p,
                              sample: {
                                ...p.sample,
                                prompts: p.sample.prompts.map((pr, i) =>
                                  i === index ? { ...pr, seed: n } : pr
                                )
                              }
                            }))
                          }}
                        />
                      </label>
                      <button
                        type="button"
                        className="lora-prompt-remove"
                        aria-label={`Delete prompt ${index + 1}`}
                        title="Delete group"
                        disabled={training || draft.train.disable_sampling}
                        onClick={() =>
                          patch((p) => ({
                            ...p,
                            sample: {
                              ...p.sample,
                              prompts: p.sample.prompts.filter((_, i) => i !== index)
                            }
                          }))
                        }
                      >
                        −
                      </button>
                    </div>
                  </div>
                ))}
                <div className="lora-prompt-add">
                  <button
                    type="button"
                    aria-label="Add prompt group"
                    title="Add prompt group"
                    disabled={training || draft.train.disable_sampling}
                    onClick={() =>
                      patch((p) => {
                        const last = p.sample.prompts[p.sample.prompts.length - 1]
                        return {
                          ...p,
                          sample: {
                            ...p.sample,
                            prompts: [
                              ...p.sample.prompts,
                              {
                                prompt: '',
                                width: last?.width ?? p.sample.width,
                                height: last?.height ?? p.sample.height,
                                seed: (last?.seed ?? p.sample.seed) + 1
                              }
                            ]
                          }
                        }
                      })
                    }
                  >
                    +
                  </button>
                </div>
              </div>
            </>
          )}

          {activeSection === 'advanced' && (
            <div className="lora-grid">
              <Field label="Save every N steps">
                <NumberField
                  min={1}
                  value={draft.save.save_every}
                  emptyFallback={1}
                  disabled={training}
                  onChange={(n) =>
                    patch((p) => ({
                      ...p,
                      save: { ...p.save, save_every: n }
                    }))
                  }
                />
              </Field>
              <Field label="Max checkpoints to keep">
                <NumberField
                  min={1}
                  value={draft.save.max_step_saves_to_keep}
                  emptyFallback={1}
                  disabled={training}
                  onChange={(n) =>
                    patch((p) => ({
                      ...p,
                      save: {
                        ...p.save,
                        max_step_saves_to_keep: n
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
              <ToggleField
                label="Push to Hugging Face Hub"
                checked={draft.save.push_to_hub}
                disabled={training}
                onChange={(v) =>
                  patch((p) => ({ ...p, save: { ...p.save, push_to_hub: v } }))
                }
              />
              <ToggleField
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
                <NumberField
                  min={0}
                  max={1}
                  step={0.01}
                  value={draft.train.ema_config.ema_decay}
                  emptyFallback={0}
                  disabled={training}
                  onChange={(n) =>
                    patch((p) => ({
                      ...p,
                      train: {
                        ...p.train,
                        ema_config: {
                          ...p.train.ema_config,
                          ema_decay: n
                        }
                      }
                    }))
                  }
                />
              </Field>
              <ToggleField
                label="Train transformer / UNet"
                checked={draft.train.train_unet}
                disabled={training}
                onChange={(v) =>
                  patch((p) => ({ ...p, train: { ...p.train, train_unet: v } }))
                }
              />
              <ToggleField
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
              <ToggleField
                label="Quantize"
                checked={draft.model.quantize}
                disabled={training}
                onChange={(v) =>
                  patch((p) => ({ ...p, model: { ...p.model, quantize: v } }))
                }
              />
              <ToggleField
                label="Low VRAM mode"
                hint="On: stage TE/VAE/DiT offload + force gradient checkpointing (saves VRAM, slower samples). Off: keep DiT on GPU during sample (needs more VRAM)."
                checked={draft.model.low_vram}
                disabled={training}
                onChange={(v) =>
                  patch((p) => ({ ...p, model: { ...p.model, low_vram: v } }))
                }
              />
            </div>
          )}
        </div>
        )}

        <ResourceMonitorPane device={draft.device} />
      </div>
    </div>
  )
}
