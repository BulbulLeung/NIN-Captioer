import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { LoraTestDraft, LoraTrainAppSettings, LoraTrainJobConfig } from '../types'
import { modelDownloadPathFromDownloadFolder, normalizeLoraTestDraft } from '../types'
import {
  generateWithComfy,
  interruptComfyGeneration,
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

interface DitCheckpointItem {
  name: string
  path: string
}

interface HistoryItem {
  url: string
  filePath: string
  promptIndex: number | null
  checkpointName: string | null
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

function pathKey(p: string): string {
  return p.replace(/\\/g, '/').toLowerCase()
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

/** Parse Captioer loratest filenames: [pNN_]stepNNNNNN_w1.00_seed123[_ckpt_NAME] or legacy. */
function parseLoratestMeta(fileName: string): {
  promptIndex: number | null
  checkpointName: string | null
  loraStep: number | null
  loraStrength: number | null
  seed: number | null
} {
  const stem = fileName.replace(/\.[^.]+$/, '')
  const withPrompt =
    /^p(\d+)_step(\d+)_w([0-9]+(?:\.[0-9]+)?)_seed(\d+)(?:_ckpt_(.+?))?(?:_\d{10,})?$/i.exec(
      stem
    )
  if (withPrompt) {
    const rawCkpt = (withPrompt[5] || '').trim()
    return {
      promptIndex: Number(withPrompt[1]),
      loraStep: Number(withPrompt[2]),
      loraStrength: Number(withPrompt[3]),
      seed: Number(withPrompt[4]),
      checkpointName: rawCkpt || null
    }
  }
  const withCkpt =
    /^step(\d+)_w([0-9]+(?:\.[0-9]+)?)_seed(\d+)(?:_ckpt_(.+?))?(?:_\d{10,})?$/i.exec(stem)
  if (withCkpt) {
    const rawCkpt = (withCkpt[4] || '').trim()
    return {
      promptIndex: null,
      loraStep: Number(withCkpt[1]),
      loraStrength: Number(withCkpt[2]),
      seed: Number(withCkpt[3]),
      checkpointName: rawCkpt || null
    }
  }
  const withWeight = /^step(\d+)_w([0-9]+(?:\.[0-9]+)?)_seed(\d+)(?:_\d+)?$/i.exec(stem)
  if (withWeight) {
    return {
      promptIndex: null,
      loraStep: Number(withWeight[1]),
      loraStrength: Number(withWeight[2]),
      seed: Number(withWeight[3]),
      checkpointName: null
    }
  }
  const legacy = /^step(\d+)_seed(\d+)(?:_\d+)?$/i.exec(stem)
  if (legacy) {
    return {
      promptIndex: null,
      loraStep: Number(legacy[1]),
      loraStrength: null,
      seed: Number(legacy[2]),
      checkpointName: null
    }
  }
  return {
    promptIndex: null,
    checkpointName: null,
    loraStep: null,
    loraStrength: null,
    seed: null
  }
}

function sanitizeCheckpointTag(name: string): string {
  return name
    .replace(/\.safetensors$/i, '')
    .replace(/[<>:"/\\|?*\x00-\x1f]+/g, '_')
    .replace(/\s+/g, '_')
    .trim()
}

function formatMetaValue(v: number | null, digits?: number): string {
  if (v == null || !Number.isFinite(v)) return '—'
  if (digits != null) return v.toFixed(digits)
  return String(v)
}

const FILTER_NONE = '__none__'

type FilterCategory = 'checkpoint' | 'prompt' | 'step' | 'weight'

type FilterMaps = Record<FilterCategory, Record<string, boolean>>

type FilterOption = { key: string; label: string }

const EMPTY_FILTER_MAPS: FilterMaps = {
  checkpoint: {},
  prompt: {},
  step: {},
  weight: {}
}

function checkpointFilterKey(item: HistoryItem): string {
  const name = (item.checkpointName || '').trim()
  return name || FILTER_NONE
}

function promptFilterKey(item: HistoryItem): string {
  return item.promptIndex != null && Number.isFinite(item.promptIndex)
    ? String(item.promptIndex)
    : FILTER_NONE
}

function stepFilterKey(item: HistoryItem): string {
  return item.loraStep != null && Number.isFinite(item.loraStep)
    ? String(item.loraStep)
    : FILTER_NONE
}

function weightFilterKey(item: HistoryItem): string {
  return item.loraStrength != null && Number.isFinite(item.loraStrength)
    ? item.loraStrength.toFixed(2)
    : FILTER_NONE
}

function filterKeyLabel(category: FilterCategory, key: string): string {
  if (key === FILTER_NONE) return '—'
  if (category === 'prompt') return `Prompt ${key}`
  return key
}

function compareFilterKeys(category: FilterCategory, a: string, b: string): number {
  if (a === FILTER_NONE) return 1
  if (b === FILTER_NONE) return -1
  if (category === 'checkpoint') return a.localeCompare(b, undefined, { sensitivity: 'base' })
  const na = Number(a)
  const nb = Number(b)
  if (Number.isFinite(na) && Number.isFinite(nb) && na !== nb) return na - nb
  return a.localeCompare(b, undefined, { numeric: true })
}

function collectFilterOptions(items: HistoryItem[]): Record<FilterCategory, FilterOption[]> {
  const buckets: Record<FilterCategory, Set<string>> = {
    checkpoint: new Set(),
    prompt: new Set(),
    step: new Set(),
    weight: new Set()
  }
  for (const item of items) {
    buckets.checkpoint.add(checkpointFilterKey(item))
    buckets.prompt.add(promptFilterKey(item))
    buckets.step.add(stepFilterKey(item))
    buckets.weight.add(weightFilterKey(item))
  }
  const toOptions = (category: FilterCategory): FilterOption[] =>
    [...buckets[category]]
      .sort((a, b) => compareFilterKeys(category, a, b))
      .map((key) => ({ key, label: filterKeyLabel(category, key) }))
  return {
    checkpoint: toOptions('checkpoint'),
    prompt: toOptions('prompt'),
    step: toOptions('step'),
    weight: toOptions('weight')
  }
}

function syncFilterMap(prev: Record<string, boolean>, keys: string[]): Record<string, boolean> {
  const next: Record<string, boolean> = {}
  for (const key of keys) {
    next[key] = key in prev ? Boolean(prev[key]) : true
  }
  return next
}

function itemMatchesFilter(item: HistoryItem, filterOn: FilterMaps): boolean {
  return (
    filterOn.checkpoint[checkpointFilterKey(item)] === true &&
    filterOn.prompt[promptFilterKey(item)] === true &&
    filterOn.step[stepFilterKey(item)] === true &&
    filterOn.weight[weightFilterKey(item)] === true
  )
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
  const [ditFiles, setDitFiles] = useState<DitCheckpointItem[]>([])
  const [comfyOnline, setComfyOnline] = useState(false)
  const [generating, setGenerating] = useState(false)
  const [genProgress, setGenProgress] = useState<string | null>(null)
  const [genError, setGenError] = useState<string | null>(null)
  const [history, setHistory] = useState<HistoryItem[]>([])
  const [activeIndex, setActiveIndex] = useState(0)
  const [filterOpen, setFilterOpen] = useState(false)
  const [filterOn, setFilterOn] = useState<FilterMaps>(EMPTY_FILTER_MAPS)
  const [startingComfy, setStartingComfy] = useState(false)
  const [editingPromptIndex, setEditingPromptIndex] = useState<number | null>(null)
  const [editingPromptDraft, setEditingPromptDraft] = useState('')
  const persistTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const skipPersist = useRef(true)
  const abortRef = useRef<AbortController | null>(null)
  const seededJobRef = useRef<string | null>(null)
  const activePathRef = useRef<string | null>(null)

  const filterOptions = useMemo(() => collectFilterOptions(history), [history])

  const filteredHistory = useMemo(
    () => history.filter((item) => itemMatchesFilter(item, filterOn)),
    [history, filterOn]
  )

  useEffect(() => {
    setFilterOn((prev) => ({
      checkpoint: syncFilterMap(
        prev.checkpoint,
        filterOptions.checkpoint.map((o) => o.key)
      ),
      prompt: syncFilterMap(
        prev.prompt,
        filterOptions.prompt.map((o) => o.key)
      ),
      step: syncFilterMap(
        prev.step,
        filterOptions.step.map((o) => o.key)
      ),
      weight: syncFilterMap(
        prev.weight,
        filterOptions.weight.map((o) => o.key)
      )
    }))
  }, [filterOptions])

  useEffect(() => {
    const path = activePathRef.current
    if (filteredHistory.length === 0) {
      setActiveIndex(0)
      return
    }
    const idx = path ? filteredHistory.findIndex((item) => item.filePath === path) : -1
    setActiveIndex(idx >= 0 ? idx : 0)
  }, [filteredHistory])

  const patch = useCallback((partial: Partial<LoraTestDraft>) => {
    setLocal((prev) => normalizeLoraTestDraft({ ...prev, ...partial }))
  }, [])

  const reportStatus = useCallback(
    (message: string, isError?: boolean) => {
      onStatus(message, isError, isError ? undefined : { sticky: true })
    },
    [onStatus]
  )

  const toggleFilterValue = useCallback((category: FilterCategory, key: string) => {
    setFilterOn((prev) => ({
      ...prev,
      [category]: {
        ...prev[category],
        [key]: !prev[category][key]
      }
    }))
  }, [])

  const setAllFilters = useCallback(
    (on: boolean) => {
      setFilterOn({
        checkpoint: Object.fromEntries(filterOptions.checkpoint.map((o) => [o.key, on])),
        prompt: Object.fromEntries(filterOptions.prompt.map((o) => [o.key, on])),
        step: Object.fromEntries(filterOptions.step.map((o) => [o.key, on])),
        weight: Object.fromEntries(filterOptions.weight.map((o) => [o.key, on]))
      })
    },
    [filterOptions]
  )

  useEffect(() => {
    if (!generating) return
    const off = window.api.onComfyLog(({ line }) => {
      const text = (line || '').trim()
      if (!text) return
      reportStatus(text.length > 240 ? `${text.slice(0, 237)}…` : text)
    })
    return off
  }, [generating, reportStatus])

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
      if (prev.prompts.every((p) => !p.text.trim())) {
        const seeded = sample.prompts
          .map((sp) => {
            const base = (sp.prompt || '').trim()
            if (!base) return null
            return {
              text: trigger ? `${trigger}, ${base}` : base,
              enabled: true
            }
          })
          .filter((p): p is { text: string; enabled: boolean } => p != null)
        next.prompts = seeded.length > 0 ? seeded : [{ text: '', enabled: true }]
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
      if (filteredHistory.length < 2) return
      const cur = Math.min(Math.max(0, activeIndex), filteredHistory.length - 1)
      // History is newest-first (left → right); arrows follow the strip.
      const next =
        e.key === 'ArrowLeft'
          ? Math.max(0, cur - 1)
          : Math.min(filteredHistory.length - 1, cur + 1)
      if (next === cur) return
      e.preventDefault()
      setActiveIndex(next)
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [filteredHistory, activeIndex])

  const activeItem = filteredHistory[activeIndex] ?? null
  const imageUrl = activeItem?.url ?? null

  useEffect(() => {
    activePathRef.current = activeItem?.filePath ?? null
  }, [activeItem])

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
          promptIndex: meta.promptIndex,
          checkpointName: meta.checkpointName,
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

  const refreshDitCheckpoints = useCallback(async () => {
    const folder = local.checkpointFolder.trim()
    if (!folder) {
      setDitFiles([])
      setLocal((prev) => (prev.ditPath ? normalizeLoraTestDraft({ ...prev, ditPath: '' }) : prev))
      return
    }
    try {
      const result = await window.api.loraTestListDitCheckpoints(folder)
      if (!result.ok) {
        setDitFiles([])
        return
      }
      const files = result.files
      setDitFiles(files)
      setLocal((prev) => {
        if (files.length === 0) {
          return prev.ditPath ? normalizeLoraTestDraft({ ...prev, ditPath: '' }) : prev
        }
        const match = files.find((f) => pathKey(f.path) === pathKey(prev.ditPath))
        if (match) {
          return match.path === prev.ditPath
            ? prev
            : normalizeLoraTestDraft({ ...prev, ditPath: match.path })
        }
        return normalizeLoraTestDraft({ ...prev, ditPath: files[0].path })
      })
    } catch {
      setDitFiles([])
    }
  }, [local.checkpointFolder])

  useEffect(() => {
    void refreshDitCheckpoints()
  }, [refreshDitCheckpoints])

  useEffect(() => {
    void refreshCheckpoints()
  }, [refreshCheckpoints, jobId])

  useEffect(() => {
    void refreshGallery()
  }, [refreshGallery, jobId])

  useEffect(() => {
    setFilterOpen(false)
  }, [jobId])

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
      if (set.has(step)) {
        if (set.size <= 1) return prev
        set.delete(step)
      } else {
        set.add(step)
      }
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
    reportStatus('Starting ComfyUI…')
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
        reportStatus(result.error || 'Failed to start ComfyUI', true)
        return false
      }
      setComfyOnline(true)
      reportStatus('ComfyUI ready')
      return true
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err)
      setGenError(msg)
      reportStatus(msg, true)
      return false
    } finally {
      setStartingComfy(false)
    }
  }

  const onCancelGenerate = () => {
    abortRef.current?.abort()
    void interruptComfyGeneration()
    setGenProgress(null)
    reportStatus('Cancelling…')
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
      reportStatus(msg, true)
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
      reportStatus(conflict, true)
      return
    }
    if (local.selectedLoraSteps.length < 1) {
      const msg = 'Select at least one trained LoRA checkpoint'
      setGenError(msg)
      reportStatus(msg, true)
      return
    }
    if (checkpoints.length === 0) {
      const msg = 'No trained LoRA checkpoints for this job'
      setGenError(msg)
      reportStatus(msg, true)
      return
    }

    const selected = local.selectedLoraSteps
      .map((step) => checkpoints.find((c) => c.step === step))
      .filter((c): c is CheckpointItem => Boolean(c))
    if (selected.length < 1) {
      const msg = 'Selected LoRA checkpoints not found on disk'
      setGenError(msg)
      reportStatus(msg, true)
      return
    }

    const promptEntries = local.prompts
      .map((entry, index) => ({
        text: entry.text.trim(),
        enabled: entry.enabled !== false,
        index
      }))
      .filter((p) => p.enabled && p.text)
    if (promptEntries.length < 1) {
      const msg = 'Enable at least one non-empty prompt'
      setGenError(msg)
      reportStatus(msg, true)
      return
    }

    const trainingFolder = (job.training_folder || '').trim()
    const jobName = (job.name || '').trim()
    if (!trainingFolder || !jobName) {
      const msg = 'Set Job name and Output Folder in LoraTrain settings'
      setGenError(msg)
      reportStatus(msg, true)
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
    const sharedBase = {
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
    const total = promptEntries.length * selected.length

    try {
      let doneCount = 0
      for (const { text: promptText, index: promptIndex } of promptEntries) {
        for (let i = 0; i < selected.length; i++) {
          if (ac.signal.aborted) throw new Error('Cancelled')
          const ckpt = selected[i]
          const n = doneCount + 1
          setGenProgress(
            `Generating ${n}/${total} (prompt ${promptIndex + 1}, step ${ckpt.step})…`
          )
          const result = await generateWithComfy(
            {
              ...sharedBase,
              prompt: promptText,
              loraName: loraNameFromPath(ckpt.path)
            },
            { signal: ac.signal }
          )
          if (!result.filePath) {
            throw new Error('ComfyUI finished but local image path was not resolved')
          }
          const weightTag = Number(local.loraStrength).toFixed(2)
          const ckptTag = sanitizeCheckpointTag(basename(ditPath))
          const promptTag = `p${String(promptIndex + 1).padStart(2, '0')}`
          const saved = await window.api.loraTestSaveGeneratedImage({
            sourcePath: result.filePath,
            trainingFolder,
            jobName,
            fileName: `${promptTag}_step${String(ckpt.step).padStart(6, '0')}_w${weightTag}_seed${runSeed}${
              ckptTag ? `_ckpt_${ckptTag}` : ''
            }`
          })
          if (!saved.ok || !saved.path) {
            throw new Error(saved.error || 'Failed to save image into loratest folder')
          }
          doneCount += 1
          await refreshGallery()
        }
      }
      setGenProgress(null)
      reportStatus('Done')
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err)
      if (ac.signal.aborted || msg === 'Cancelled') {
        setGenError(null)
        reportStatus('Cancelled')
      } else {
        setGenError(msg)
        reportStatus(msg, true)
      }
      setGenProgress(null)
    } finally {
      if (abortRef.current === ac) abortRef.current = null
      setGenerating(false)
    }
  }

  const stopComfy = async () => {
    await window.api.stopComfyUi()
    setComfyOnline(false)
    reportStatus('ComfyUI stopped')
  }

  const openPromptEditor = (index: number) => {
    setEditingPromptIndex(index)
    setEditingPromptDraft(local.prompts[index]?.text ?? '')
  }

  const closePromptEditor = (save: boolean) => {
    if (save && editingPromptIndex != null) {
      const next = local.prompts.map((p, i) =>
        i === editingPromptIndex ? { ...p, text: editingPromptDraft } : p
      )
      patch({ prompts: next })
    }
    setEditingPromptIndex(null)
    setEditingPromptDraft('')
  }

  const addPrompt = () => {
    patch({ prompts: [...local.prompts, { text: '', enabled: true }] })
  }

  const removePrompt = (index: number) => {
    if (local.prompts.length <= 1) {
      patch({ prompts: [{ text: '', enabled: true }] })
      return
    }
    const next = local.prompts.filter((_, i) => i !== index)
    if (!next.some((p) => p.enabled)) {
      next[0] = { ...next[0], enabled: true }
    }
    patch({ prompts: next })
  }

  const togglePromptEnabled = (index: number) => {
    const entry = local.prompts[index]
    if (!entry) return
    if (entry.enabled) {
      const enabledCount = local.prompts.filter((p) => p.enabled).length
      if (enabledCount <= 1) return
    }
    patch({
      prompts: local.prompts.map((p, i) =>
        i === index ? { ...p, enabled: !p.enabled } : p
      )
    })
  }

  useEffect(() => {
    if (editingPromptIndex == null) return
    const idx = editingPromptIndex
    const draftText = editingPromptDraft
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return
      e.preventDefault()
      setLocal((prev) => {
        const next = prev.prompts.map((p, i) =>
          i === idx ? { ...p, text: draftText } : p
        )
        return normalizeLoraTestDraft({ ...prev, prompts: next })
      })
      setEditingPromptIndex(null)
      setEditingPromptDraft('')
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [editingPromptIndex, editingPromptDraft])

  return (
    <>
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
              <span>Checkpoint</span>
              <select
                value={local.ditPath}
                disabled={ditFiles.length === 0}
                onChange={(e) => patch({ ditPath: e.target.value })}
              >
                {ditFiles.length === 0 ? (
                  <option value="">No safetensors in folder</option>
                ) : (
                  ditFiles.map((f) => (
                    <option key={f.path} value={f.path}>
                      {f.name}
                    </option>
                  ))
                )}
              </select>
            </label>

            <div className="lora-test-prompt-groups">
              <div className="lora-test-prompt-groups-header">
                <span>Prompts</span>
              </div>
              {local.prompts.map((entry, index) => {
                const enabledCount = local.prompts.filter((p) => p.enabled).length
                const lockOn = entry.enabled && enabledCount <= 1
                return (
                <div key={index} className="lora-test-prompt-row">
                  <label className="field">
                    <span>Prompt {index + 1}</span>
                    <div className="lora-test-prompt-input-row">
                      <div
                        className={`lora-toggle lora-test-prompt-toggle${
                          entry.enabled ? ' is-on' : ''
                        }${lockOn ? ' is-disabled' : ''}`}
                      >
                        <button
                          type="button"
                          role="switch"
                          className="lora-switch"
                          aria-checked={entry.enabled}
                          aria-label={`Enable prompt ${index + 1}`}
                          title={
                            lockOn
                              ? 'At least one prompt must stay on'
                              : entry.enabled
                                ? 'Enabled for generate'
                                : 'Skipped when generating'
                          }
                          disabled={lockOn}
                          onClick={() => togglePromptEnabled(index)}
                        >
                          <span className="lora-switch-knob" aria-hidden="true" />
                        </button>
                      </div>
                      <input
                        type="text"
                        className="lora-test-prompt-input"
                        value={entry.text}
                        readOnly
                        placeholder="Click to edit…"
                        title={entry.text || undefined}
                        onClick={() => openPromptEditor(index)}
                        onFocus={(e) => {
                          e.target.blur()
                          openPromptEditor(index)
                        }}
                      />
                      <button
                        type="button"
                        className="lora-test-prompt-remove"
                        title="Remove prompt"
                        disabled={local.prompts.length <= 1}
                        onClick={() => removePrompt(index)}
                      >
                        −
                      </button>
                    </div>
                  </label>
                </div>
                )
              })}
              <div className="lora-test-prompt-add">
                <button type="button" onClick={addPrompt}>
                  +
                </button>
              </div>
            </div>

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
              {checkpoints.length > 0 && (
                <ul className="lora-test-lora-list">
                  {checkpoints.map((c) => {
                    const on = local.selectedLoraSteps.includes(c.step)
                    const lockOn = on && local.selectedLoraSteps.length <= 1
                    return (
                      <li key={c.step}>
                        <div
                          className={`lora-toggle lora-test-lora-toggle${on ? ' is-on' : ''}${
                            lockOn ? ' is-disabled' : ''
                          }`}
                        >
                          <button
                            type="button"
                            role="switch"
                            className="lora-switch"
                            aria-checked={on}
                            aria-label={`Select LoRA step ${c.step}`}
                            title={
                              lockOn ? 'At least one LoRA step must stay on' : c.path
                            }
                            disabled={lockOn}
                            onClick={() => toggleLoraStep(c.step)}
                          >
                            <span className="lora-switch-knob" aria-hidden="true" />
                          </button>
                          <span className="lora-toggle-label" title={c.path}>
                            step {c.step}
                          </span>
                        </div>
                      </li>
                    )
                  })}
                </ul>
              )}
              <label className="field">
                <span>LoRA strength</span>
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
              className={`${generating ? 'danger' : 'primary'} lora-test-generate-btn`}
              disabled={startingComfy}
              onClick={() => {
                if (generating) onCancelGenerate()
                else void onGenerate()
              }}
            >
              {generating ? 'Cancel' : 'Generate'}
            </button>
          </div>
        </aside>

        <section className="lora-test-viewer">
          <div className="lora-test-viewer-meta">
            <div className="lora-test-viewer-meta-left">
              {activeItem && (
                <>
                  {activeItem.promptIndex != null && (
                    <span>Prompt {activeItem.promptIndex}</span>
                  )}
                  <span title={activeItem.checkpointName || undefined}>
                    {activeItem.checkpointName || '—'}
                  </span>
                  <span>LoRA step {formatMetaValue(activeItem.loraStep)}</span>
                  <span>Weight {formatMetaValue(activeItem.loraStrength, 2)}</span>
                  <span>Seed {formatMetaValue(activeItem.seed)}</span>
                </>
              )}
            </div>
            <button
              type="button"
              className="lora-test-filter-btn"
              title="Filter"
              aria-label="Filter"
              aria-haspopup="dialog"
              aria-expanded={filterOpen}
              onClick={() => setFilterOpen(true)}
            >
              Filter
            </button>
          </div>
          <div className="lora-test-viewer-main">
            {imageUrl ? (
              <img src={imageUrl} alt="Generated" className="lora-test-viewer-img" />
            ) : (
              <p className="lora-test-viewer-empty">
                {history.length > 0
                  ? 'No images match the current filter'
                  : 'Generated images appear here'}
              </p>
            )}
          </div>
          {filteredHistory.length > 1 && (
            <div className="lora-test-history">
              {filteredHistory.map((item, idx) => (
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
    {editingPromptIndex != null && (
      <div
        className="modal-backdrop"
        onMouseDown={(e) => {
          if (e.target === e.currentTarget) closePromptEditor(true)
        }}
      >
        <div
          className="modal modal-wide"
          role="dialog"
          aria-labelledby="lora-test-prompt-edit-title"
          onMouseDown={(e) => e.stopPropagation()}
        >
          <h2 id="lora-test-prompt-edit-title">Prompt {editingPromptIndex + 1}</h2>
          <textarea
            className="prompt-textarea"
            value={editingPromptDraft}
            onChange={(e) => setEditingPromptDraft(e.target.value)}
            autoFocus
            spellCheck={false}
          />
          <div className="modal-actions">
            <button type="button" onClick={() => closePromptEditor(false)}>
              Cancel
            </button>
            <div className="spacer" />
            <button type="button" className="primary" onClick={() => closePromptEditor(true)}>
              Done
            </button>
          </div>
        </div>
      </div>
    )}
    {filterOpen && (
      <div
        className="modal-backdrop"
        onMouseDown={(e) => {
          if (e.target === e.currentTarget) setFilterOpen(false)
        }}
      >
        <div
          className="modal lora-test-filter-modal"
          role="dialog"
          aria-labelledby="lora-test-filter-title"
          onMouseDown={(e) => e.stopPropagation()}
        >
          <div className="lora-test-filter-header">
            <h2 id="lora-test-filter-title">Filter</h2>
            <div className="lora-test-filter-header-actions">
              <button type="button" onClick={() => setAllFilters(true)}>
                All On
              </button>
              <button type="button" onClick={() => setAllFilters(false)}>
                All Off
              </button>
            </div>
          </div>
          <div className="lora-test-filter-body">
            {(
              [
                { id: 'checkpoint', title: 'Checkpoint' },
                { id: 'prompt', title: 'Prompt' },
                { id: 'step', title: 'Step' },
                { id: 'weight', title: 'Weight' }
              ] as const
            ).map((section) => (
              <div key={section.id} className="lora-test-filter-section">
                <h3>{section.title}</h3>
                {filterOptions[section.id].length === 0 ? (
                  <p className="lora-test-filter-empty">No values in loratest folder</p>
                ) : (
                  <ul className="lora-test-filter-list">
                    {filterOptions[section.id].map((opt) => {
                      const on = filterOn[section.id][opt.key] === true
                      return (
                        <li key={opt.key}>
                          <div className={`lora-toggle${on ? ' is-on' : ''}`}>
                            <button
                              type="button"
                              role="switch"
                              className="lora-switch"
                              aria-checked={on}
                              aria-label={`Filter ${section.title} ${opt.label}`}
                              title={opt.label}
                              onClick={() => toggleFilterValue(section.id, opt.key)}
                            >
                              <span className="lora-switch-knob" aria-hidden="true" />
                            </button>
                            <span className="lora-toggle-label" title={opt.label}>
                              {opt.label}
                            </span>
                          </div>
                        </li>
                      )
                    })}
                  </ul>
                )}
              </div>
            ))}
          </div>
          <div className="modal-actions">
            <div className="spacer" />
            <button type="button" className="primary" onClick={() => setFilterOpen(false)}>
              Close
            </button>
          </div>
        </div>
      </div>
    )}
  </>
  )
}

function joinPath(a: string, b: string): string {
  const left = (a || '').trim().replace(/[/\\]+$/, '')
  const right = (b || '').trim().replace(/^[/\\]+/, '')
  if (!left || !right) return ''
  return `${left}\\${right}`
}
