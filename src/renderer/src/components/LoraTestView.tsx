import { useCallback, useEffect, useRef, useState } from 'react'
import type { LoraTestDraft, LoraTrainAppSettings, LoraTrainJobConfig } from '../types'
import { modelDownloadPathFromDownloadFolder, normalizeLoraTestDraft } from '../types'
import {
  generateWithComfy,
  loraNameFromPath,
  parentDir
} from '../services/comfyGenerate'
import { ResourceMonitorPane } from './ResourceMonitorPane'

const SAMPLERS = [
  'euler',
  'euler_ancestral',
  'heun',
  'dpmpp_2m',
  'dpmpp_2m_sde',
  'dpmpp_sde',
  'uni_pc',
  'lms',
  'ddim'
]

const SCHEDULERS = [
  'simple',
  'normal',
  'karras',
  'exponential',
  'sgm_uniform',
  'ddim_uniform',
  'beta'
]

interface CheckpointItem {
  step: number
  path: string
}

interface HistoryItem {
  url: string
  filePath: string
  loraStep: number | null
  loraStrength: number | null
  seed: number | null
}

interface Props {
  job: LoraTrainJobConfig
  jobId: string
  draft: LoraTestDraft
  appSettings: LoraTrainAppSettings
  onDraftChange: (draft: LoraTestDraft) => void
  onStatus: (message: string, isError?: boolean, options?: { sticky?: boolean }) => void
}

function basename(fullPath: string): string {
  return loraNameFromPath(fullPath)
}

function checkBasenameConflicts(paths: { label: string; path: string }[]): string | null {
  const map = new Map<string, string>()
  for (const { label, path } of paths) {
    const name = basename(path).toLowerCase()
    if (!name) continue
    const prev = map.get(name)
    if (prev && prev !== path) {
      return `Duplicate filename "${basename(path)}" used by ${label} and another model path`
    }
    map.set(name, path)
  }
  return null
}

/** Parse Captioer loratest filenames: stepNNNNNN_w1.00_seed123 or legacy stepNNNNNN_seed123. */
function parseLoratestMeta(fileName: string): {
  loraStep: number | null
  loraStrength: number | null
  seed: number | null
} {
  const stem = fileName.replace(/\.[^.]+$/, '')
  const withWeight = /^step(\d+)_w([0-9]+(?:\.[0-9]+)?)_seed(\d+)(?:_\d+)?$/i.exec(stem)
  if (withWeight) {
    return {
      loraStep: Number(withWeight[1]),
      loraStrength: Number(withWeight[2]),
      seed: Number(withWeight[3])
    }
  }
  const legacy = /^step(\d+)_seed(\d+)(?:_\d+)?$/i.exec(stem)
  if (legacy) {
    return {
      loraStep: Number(legacy[1]),
      loraStrength: null,
      seed: Number(legacy[2])
    }
  }
  return { loraStep: null, loraStrength: null, seed: null }
}

function formatMetaValue(v: number | null, digits?: number): string {
  if (v == null || !Number.isFinite(v)) return '—'
  if (digits != null) return v.toFixed(digits)
  return String(v)
}

export function LoraTestView({
  job,
  jobId,
  draft,
  appSettings,
  onDraftChange,
  onStatus
}: Props) {
  const [local, setLocal] = useState(() => normalizeLoraTestDraft(draft))
  const [checkpoints, setCheckpoints] = useState<CheckpointItem[]>([])
  const [comfyOnline, setComfyOnline] = useState(false)
  const [generating, setGenerating] = useState(false)
  const [genProgress, setGenProgress] = useState<string | null>(null)
  const [genError, setGenError] = useState<string | null>(null)
  const [history, setHistory] = useState<HistoryItem[]>([])
  const [activeIndex, setActiveIndex] = useState(0)
  const [startingComfy, setStartingComfy] = useState(false)
  const persistTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const skipPersist = useRef(true)
  const abortRef = useRef<AbortController | null>(null)
  const seededJobRef = useRef<string | null>(null)

  const patch = useCallback((partial: Partial<LoraTestDraft>) => {
    setLocal((prev) => normalizeLoraTestDraft({ ...prev, ...partial }))
  }, [])

  useEffect(() => {
    skipPersist.current = true
    setLocal(normalizeLoraTestDraft(draft))
  }, [draft])

  useEffect(() => {
    if (skipPersist.current) {
      skipPersist.current = false
      return
    }
    if (persistTimer.current) clearTimeout(persistTimer.current)
    persistTimer.current = setTimeout(() => {
      onDraftChange(normalizeLoraTestDraft(local))
    }, 400)
    return () => {
      if (persistTimer.current) clearTimeout(persistTimer.current)
    }
  }, [local, onDraftChange])

  useEffect(() => {
    if (seededJobRef.current === jobId) return
    seededJobRef.current = jobId
    const sample = job.sample
    const first = sample.prompts[0]
    const trigger = (job.trigger_word || '').trim()
    setLocal((prev) => {
      const next = { ...prev }
      let changed = false
      if (!prev.prompt.trim()) {
        const base = first?.prompt || ''
        next.prompt = trigger ? `${trigger}, ${base}` : base
        changed = true
      }
      if (!prev.negative.trim() && sample.neg) {
        next.negative = sample.neg
        changed = true
      }
      if (prev.steps === 20 && sample.sample_steps) {
        next.steps = sample.sample_steps
        changed = true
      }
      if (prev.cfg === 3.5 && sample.guidance_scale) {
        next.cfg = sample.guidance_scale
        changed = true
      }
      if (first?.width) {
        next.width = first.width
        changed = true
      }
      if (first?.height) {
        next.height = first.height
        changed = true
      }
      if (first?.seed != null) {
        next.seed = first.seed
        changed = true
      }
      if (changed) skipPersist.current = true
      return normalizeLoraTestDraft(next)
    })
  }, [jobId, job])

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return
      if (e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return
      const el = document.activeElement as HTMLElement | null
      if (el) {
        const tag = el.tagName
        if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return
        if (el.isContentEditable) return
      }
      if (history.length < 2) return
      const cur = Math.min(Math.max(0, activeIndex), history.length - 1)
      // History is newest-first (left → right); arrows follow the strip.
      const next =
        e.key === 'ArrowLeft'
          ? Math.max(0, cur - 1)
          : Math.min(history.length - 1, cur + 1)
      if (next === cur) return
      e.preventDefault()
      setActiveIndex(next)
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [history, activeIndex])

  const activeItem = history[activeIndex] ?? null
  const imageUrl = activeItem?.url ?? null

  const refreshGallery = useCallback(async () => {
    const folder = (job.training_folder || '').trim()
    const name = (job.name || '').trim()
    if (!folder || !name) {
      setHistory([])
      setActiveIndex(0)
      return
    }
    try {
      const result = await window.api.loraTestListGallery({
        trainingFolder: folder,
        jobName: name
      })
      if (!result.ok) {
        setHistory([])
        setActiveIndex(0)
        return
      }
      const items: HistoryItem[] = result.images.map((img) => {
        const meta = parseLoratestMeta(img.name)
        return {
          url: window.api.toLocalUrl(img.path),
          filePath: img.path,
          loraStep: meta.loraStep,
          loraStrength: meta.loraStrength,
          seed: meta.seed
        }
      })
      setHistory(items)
      setActiveIndex(0)
    } catch {
      setHistory([])
      setActiveIndex(0)
    }
  }, [job.training_folder, job.name])

  const refreshCheckpoints = useCallback(async () => {
    const folder = (job.training_folder || '').trim()
    const name = (job.name || '').trim()
    if (!folder || !name) {
      setCheckpoints([])
      return
    }
    try {
      const result = await window.api.listTrainCheckpoints({
        trainingFolder: folder,
        jobName: name
      })
      if (result.ok) {
        setCheckpoints(result.checkpoints)
        setLocal((prev) => {
          const available = new Set(result.checkpoints.map((c) => c.step))
          let selected = prev.selectedLoraSteps.filter((s) => available.has(s))
          if (selected.length === 0 && result.checkpoints.length > 0) {
            selected = [result.checkpoints[result.checkpoints.length - 1].step]
          }
          if (
            selected.length === prev.selectedLoraSteps.length &&
            selected.every((s, i) => s === prev.selectedLoraSteps[i])
          ) {
            return prev
          }
          return normalizeLoraTestDraft({ ...prev, selectedLoraSteps: selected })
        })
      } else {
        setCheckpoints([])
      }
    } catch {
      setCheckpoints([])
    }
  }, [job.training_folder, job.name])

  useEffect(() => {
    void refreshCheckpoints()
  }, [refreshCheckpoints, jobId])

  useEffect(() => {
    void refreshGallery()
  }, [refreshGallery, jobId])

  const refreshComfyStatus = useCallback(async () => {
    try {
      const st = await window.api.comfyStatus()
      setComfyOnline(st.online)
      return st.online
    } catch {
      setComfyOnline(false)
      return false
    }
  }, [])

  useEffect(() => {
    void refreshComfyStatus()
    const t = setInterval(() => void refreshComfyStatus(), 5000)
    return () => clearInterval(t)
  }, [refreshComfyStatus])

  const toggleLoraStep = (step: number) => {
    setLocal((prev) => {
      const set = new Set(prev.selectedLoraSteps)
      if (set.has(step)) set.delete(step)
      else set.add(step)
      return normalizeLoraTestDraft({
        ...prev,
        selectedLoraSteps: [...set].sort((a, b) => a - b)
      })
    })
  }

  const modelFolders = () => {
    const dit = parentDir(local.ditPath.trim())
    const vae = parentDir(local.vaePath.trim())
    const t5 = parentDir(local.t5Path.trim())
    return {
      ditFolders: dit ? [dit] : [],
      vaeFolders: vae ? [vae] : [],
      clipFolders: t5 ? [t5] : []
    }
  }

  const ensureComfy = async (): Promise<boolean> => {
    const bat = local.comfyUiBatPath.trim()
    if (!bat) {
      setGenError('Set ComfyUI launch bat in LoraTest Settings (or Download ComfyUI)')
      return false
    }
    setStartingComfy(true)
    onStatus('Starting ComfyUI…')
    try {
      const modelsRoot = modelDownloadPathFromDownloadFolder(appSettings.downloadFolder)
      const loraFolder = joinPath(job.training_folder, job.name)
      const folders = modelFolders()
      const result = await window.api.startComfyUi({
        batPath: bat,
        pythonPath: appSettings.pythonPath.trim() || undefined,
        modelsRoot,
        loraFolders: loraFolder ? [loraFolder] : [],
        ...folders
      })
      if (!result.ok) {
        setGenError(result.error || 'Failed to start ComfyUI')
        onStatus(result.error || 'Failed to start ComfyUI', true)
        return false
      }
      setComfyOnline(true)
      onStatus(result.alreadyRunning ? 'ComfyUI ready' : 'ComfyUI ready')
      return true
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err)
      setGenError(msg)
      onStatus(msg, true)
      return false
    } finally {
      setStartingComfy(false)
    }
  }

  const onGenerate = async () => {
    if (generating) return
    setGenError(null)

    const ditPath = local.ditPath.trim()
    const vaePath = local.vaePath.trim()
    const t5Path = local.t5Path.trim()
    // Krea2 needs DiT + VAE + Qwen TE (Text Encoder slot).
    if (!ditPath || !vaePath || !t5Path) {
      const msg = 'Set DiT, VAE, and Text Encoder (Qwen) in LoraTest Settings'
      setGenError(msg)
      onStatus(msg, true)
      return
    }
    const conflict = checkBasenameConflicts(
      [
        { label: 'DiT', path: ditPath },
        { label: 'VAE', path: vaePath },
        { label: 'Text Encoder', path: t5Path }
      ].filter((x) => x.path)
    )
    if (conflict) {
      setGenError(conflict)
      onStatus(conflict, true)
      return
    }
    if (local.selectedLoraSteps.length < 1) {
      const msg = 'Select at least one trained LoRA checkpoint'
      setGenError(msg)
      onStatus(msg, true)
      return
    }
    if (checkpoints.length === 0) {
      const msg = 'No trained LoRA checkpoints for this job'
      setGenError(msg)
      onStatus(msg, true)
      return
    }

    const selected = local.selectedLoraSteps
      .map((step) => checkpoints.find((c) => c.step === step))
      .filter((c): c is CheckpointItem => Boolean(c))
    if (selected.length < 1) {
      const msg = 'Selected LoRA checkpoints not found on disk'
      setGenError(msg)
      onStatus(msg, true)
      return
    }

    const trainingFolder = (job.training_folder || '').trim()
    const jobName = (job.name || '').trim()
    if (!trainingFolder || !jobName) {
      const msg = 'Set Job name and Output Folder in LoraTrain settings'
      setGenError(msg)
      onStatus(msg, true)
      return
    }

    const ok = await ensureComfy()
    if (!ok) return

    setGenerating(true)
    abortRef.current?.abort()
    const ac = new AbortController()
    abortRef.current = ac
    const runSeed =
      local.seed === -1
        ? Math.floor(Math.random() * 0xffffffff)
        : Math.max(0, Math.round(local.seed))
    const shared = {
      prompt: local.prompt,
      negative: local.negative,
      steps: local.steps,
      cfg: local.cfg,
      seed: runSeed,
      width: local.width,
      height: local.height,
      sampler: local.sampler,
      scheduler: local.scheduler,
      ditName: basename(ditPath),
      vaeName: basename(vaePath),
      t5Name: basename(t5Path),
      loraStrength: local.loraStrength
    }

    try {
      let doneCount = 0
      for (let i = 0; i < selected.length; i++) {
        if (ac.signal.aborted) throw new Error('Cancelled')
        const ckpt = selected[i]
        setGenProgress(`Generating ${i + 1}/${selected.length} (step ${ckpt.step})…`)
        onStatus(`Generating ${i + 1}/${selected.length}…`)
        const result = await generateWithComfy(
          {
            ...shared,
            loraName: loraNameFromPath(ckpt.path)
          },
          { signal: ac.signal }
        )
        if (!result.filePath) {
          throw new Error('ComfyUI finished but local image path was not resolved')
        }
        const weightTag = Number(local.loraStrength).toFixed(2)
        const saved = await window.api.loraTestSaveGeneratedImage({
          sourcePath: result.filePath,
          trainingFolder,
          jobName,
          fileName: `step${String(ckpt.step).padStart(6, '0')}_w${weightTag}_seed${runSeed}`
        })
        if (!saved.ok || !saved.path) {
          throw new Error(saved.error || 'Failed to save image into loratest folder')
        }
        doneCount += 1
        await refreshGallery()
      }
      setGenProgress(null)
      onStatus(`Generate done (${doneCount} image${doneCount === 1 ? '' : 's'})`)
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err)
      setGenError(msg)
      onStatus(msg, true)
      setGenProgress(null)
    } finally {
      setGenerating(false)
    }
  }

  const stopComfy = async () => {
    await window.api.stopComfyUi()
    setComfyOnline(false)
    onStatus('ComfyUI stopped')
  }

  return (
    <div className="lora-test">
      <div className="lora-test-body">
        <aside className="lora-test-settings">
          <div className="lora-test-settings-scroll">
            <div className="lora-test-comfy-row">
              <span
                className={`lora-test-comfy-dot${comfyOnline ? ' online' : ''}`}
                title={comfyOnline ? 'ComfyUI online' : 'ComfyUI offline'}
              />
              <span className="lora-test-comfy-label">
                {comfyOnline ? 'ComfyUI online' : startingComfy ? 'Starting…' : 'ComfyUI offline'}
              </span>
              {comfyOnline ? (
                <button type="button" className="toolbar-icon-btn" onClick={() => void stopComfy()}>
                  Stop
                </button>
              ) : (
                <button
                  type="button"
                  className="toolbar-icon-btn"
                  disabled={startingComfy || !local.comfyUiBatPath.trim()}
                  onClick={() => void ensureComfy()}
                >
                  Start
                </button>
              )}
            </div>

            <label className="field">
              <span>Prompt</span>
              <textarea
                rows={6}
                value={local.prompt}
                onChange={(e) => patch({ prompt: e.target.value })}
              />
            </label>

            <label className="field">
              <span>Negative</span>
              <textarea
                rows={2}
                value={local.negative}
                onChange={(e) => patch({ negative: e.target.value })}
              />
            </label>

            <div className="field-row-grid">
              <label className="field">
                <span>Steps</span>
                <input
                  type="number"
                  min={1}
                  max={150}
                  value={local.steps}
                  onChange={(e) => patch({ steps: Number(e.target.value) })}
                />
              </label>
              <label className="field">
                <span>CFG / Guidance</span>
                <input
                  type="number"
                  min={0}
                  max={20}
                  step={0.1}
                  value={local.cfg}
                  onChange={(e) => patch({ cfg: Number(e.target.value) })}
                />
              </label>
            </div>

            <div className="field-row-grid">
              <label className="field">
                <span>Width</span>
                <input
                  type="number"
                  min={64}
                  step={64}
                  value={local.width}
                  onChange={(e) => patch({ width: Number(e.target.value) })}
                />
              </label>
              <label className="field">
                <span>Height</span>
                <input
                  type="number"
                  min={64}
                  step={64}
                  value={local.height}
                  onChange={(e) => patch({ height: Number(e.target.value) })}
                />
              </label>
            </div>

            <label className="field">
              <span>Seed</span>
              <div className="field-row">
                <input
                  type="number"
                  min={-1}
                  value={local.seed}
                  onChange={(e) => patch({ seed: Number(e.target.value) })}
                />
                <button type="button" onClick={() => patch({ seed: -1 })}>
                  Random
                </button>
              </div>
            </label>

            <div className="field-row-grid">
              <label className="field">
                <span>Sampler</span>
                <select
                  value={local.sampler}
                  onChange={(e) => patch({ sampler: e.target.value })}
                >
                  {SAMPLERS.map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span>Scheduler</span>
                <select
                  value={local.scheduler}
                  onChange={(e) => patch({ scheduler: e.target.value })}
                >
                  {SCHEDULERS.map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </select>
              </label>
            </div>

            <div className="lora-test-lora-block">
              <span className="lora-test-lora-heading">Trained LoRAs</span>
              <p className="field-hint">Select at least one. Each selected LoRA generates one image.</p>
              {checkpoints.length === 0 ? (
                <p className="field-hint">No checkpoints for this job yet.</p>
              ) : (
                <ul className="lora-test-lora-list">
                  {checkpoints.map((c) => (
                    <li key={c.step}>
                      <label className="checkbox-field">
                        <input
                          type="checkbox"
                          checked={local.selectedLoraSteps.includes(c.step)}
                          onChange={() => toggleLoraStep(c.step)}
                        />
                        <span title={c.path}>
                          step {c.step}
                          <span className="lora-test-lora-file"> {basename(c.path)}</span>
                        </span>
                      </label>
                    </li>
                  ))}
                </ul>
              )}
              <label className="field">
                <span>LoRA strength (shared)</span>
                <input
                  type="number"
                  min={0}
                  max={2}
                  step={0.05}
                  value={local.loraStrength}
                  onChange={(e) => patch({ loraStrength: Number(e.target.value) })}
                />
              </label>
            </div>

            {genProgress && <p className="field-hint">{genProgress}</p>}
            {genError && <p className="lora-test-error">{genError}</p>}
          </div>

          <div className="lora-test-generate-bar">
            <button
              type="button"
              className="primary lora-test-generate-btn"
              disabled={generating || startingComfy}
              onClick={() => void onGenerate()}
            >
              {generating ? genProgress || 'Generating…' : 'Generate'}
            </button>
          </div>
        </aside>

        <section className="lora-test-viewer">
          {activeItem && (
            <div className="lora-test-viewer-meta">
              <span>LoRA step {formatMetaValue(activeItem.loraStep)}</span>
              <span>Weight {formatMetaValue(activeItem.loraStrength, 2)}</span>
              <span>Seed {formatMetaValue(activeItem.seed)}</span>
            </div>
          )}
          <div className="lora-test-viewer-main">
            {imageUrl ? (
              <img src={imageUrl} alt="Generated" className="lora-test-viewer-img" />
            ) : (
              <p className="lora-test-viewer-empty">Generated images appear here</p>
            )}
          </div>
          {history.length > 1 && (
            <div className="lora-test-history">
              {history.map((item, idx) => (
                <button
                  key={item.url}
                  type="button"
                  className={`lora-test-history-thumb${idx === activeIndex ? ' active' : ''}`}
                  onClick={() => setActiveIndex(idx)}
                >
                  <img src={item.url} alt="" />
                </button>
              ))}
            </div>
          )}
        </section>

        <ResourceMonitorPane device={job.device} />
      </div>
    </div>
  )
}

function joinPath(a: string, b: string): string {
  const left = (a || '').trim().replace(/[/\\]+$/, '')
  const right = (b || '').trim().replace(/^[/\\]+/, '')
  if (!left || !right) return ''
  return `${left}\\${right}`
}
