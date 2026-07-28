/** Defaults for Captioer native Krea 2 LoRA trainer (train on Raw, sample on Turbo). */

import { join } from '../utils/pathJoin'

export type ActiveView = 'datasetEdit' | 'loraTrain'

export type LoraTrainArch = 'krea2'
export type LoraTrainQuantizeMode = 'none' | 'qfloat8' | 'float8' | 'int8'

export interface LoraTrainEmaConfig {
  use_ema: boolean
  ema_decay: number
}

export interface LoraTrainDatasetConfig {
  folder_path: string
  caption_ext: string
  caption_dropout_rate: number
  shuffle_tokens: boolean
  cache_latents_to_disk: boolean
  /** Enabled resolution tiers; AR bucketing uses min as floor and long-side closest tier. */
  resolution: number[]
}

export interface LoraTrainNetworkConfig {
  type: string
  linear: number
  linear_alpha: number
}

export interface LoraTrainSaveConfig {
  dtype: string
  save_every: number
  max_step_saves_to_keep: number
  push_to_hub: boolean
}

export interface LoraTrainTrainConfig {
  batch_size: number
  steps: number
  gradient_accumulation_steps: number
  train_unet: boolean
  train_text_encoder: boolean
  gradient_checkpointing: boolean
  /** Cache text encoder outputs to disk (AI-Toolkit-style; required on ~24GB). */
  cache_text_embeddings: boolean
  noise_scheduler: string
  optimizer: string
  lr: number
  dtype: string
  skip_first_sample: boolean
  disable_sampling: boolean
  ema_config: LoraTrainEmaConfig
}

export interface LoraTrainModelConfig {
  /** @deprecated prefer train_name_or_path */
  name_or_path: string
  train_name_or_path: string
  arch: LoraTrainArch
  quantize: LoraTrainQuantizeMode
  low_vram: boolean
  /**
   * Stream DiT transformer blocks CPU↔GPU (independent of low_vram).
   * Saves VRAM; slower than full GPU residency.
   */
  layer_offload: boolean
  /**
   * 0 = auto (trainer picks % from free VRAM);
   * 1–100 = manual fraction of transformer blocks streamed off GPU.
   */
  layer_offload_percent: number
}

export interface LoraTrainSamplePrompt {
  prompt: string
  width: number
  height: number
  seed: number
}

export interface LoraTrainSampleConfig {
  sampler: string
  sample_every: number
  sample_start_step: number
  /** Fallback size/seed for legacy migration and empty prompt lists */
  width: number
  height: number
  prompts: LoraTrainSamplePrompt[]
  neg: string
  seed: number
  guidance_scale: number
  sample_steps: number
}

export interface LoraTrainJobConfig {
  name: string
  training_folder: string
  device: string
  trigger_word: string
  network: LoraTrainNetworkConfig
  save: LoraTrainSaveConfig
  datasets: LoraTrainDatasetConfig[]
  train: LoraTrainTrainConfig
  model: LoraTrainModelConfig
  sample: LoraTrainSampleConfig
}

export interface LoraTrainJobPreset {
  id: string
  job: LoraTrainJobConfig
}

/** Environment / preference settings for the LoraTrain Settings dialog. */
export interface LoraTrainAppSettings {
  pythonPath: string
  /**
   * Shared root for Python install + model downloads.
   * Empty = AppData/Roaming/Captioer (userData); uses `{root}/python` and `{root}/models`.
   */
  downloadFolder: string
  huggingfaceToken: string
}

export const KREA2_RAW = 'krea/Krea-2-Raw'

export const DEFAULT_SAMPLE_PROMPTS: string[] = [
  `A striking, high-contrast waist-up front-facing portrait of an otherworldly, ethereal girl looking directly at the viewer. She has translucent, glowing alabaster skin and long, weightless hair of pure white starlight floating around her. Her large, serene eyes gaze straight forward, staring directly into the viewer's eyes like pools of liquid silver, shining brightly against a deep, pitch-black cosmic abyss.
The background is a stark, midnight-black void filled with thousands of sharp, highly-reflective crystal particles and shattered gemstone shards hovering frozen in space. These crystalline particles act as tiny prisms, catching a hidden light source and exploding with brilliant, high-contrast flares of electric blue, magenta, and sharp white.
Cutting sharply across the entire composition from the top-right corner to the bottom-left corner is a rigid band of five taut, parallel straight lines of humming, pure white light. These five luminous geometric lines slash through the space, casting intense, bright highlights onto her symmetrical face, collarbones, and shoulder.
Her slender, pale hands are held in a gentle, graceful gesture in front of her chest; her long fingers hover millimeters away from the five rigid straight light lines, guiding her hands along the diagonal path with absolute focus.
At the chords' vibration, nearby floating crystal particles ripple and spin, scattering a dazzling storm of sharp, starry lens flares and microscopic glowing dust around her cheek and shoulders.
She wears a high-collared gown crafted from layers of woven starlight and transparent diamond-veil fabric, which reflects the intense contrasting light and projects intricate, glittering caustic patterns across her serene front-facing neck and face.`
]

const DEFAULT_SAMPLE_WIDTH = 1024
const DEFAULT_SAMPLE_HEIGHT = 1024
const DEFAULT_SAMPLE_SEED = 42

function buildDefaultSamplePrompts(): LoraTrainSamplePrompt[] {
  return DEFAULT_SAMPLE_PROMPTS.map((prompt, i) => ({
    prompt,
    width: DEFAULT_SAMPLE_WIDTH,
    height: DEFAULT_SAMPLE_HEIGHT,
    seed: DEFAULT_SAMPLE_SEED + i
  }))
}

export const DEFAULT_LORA_TRAIN_JOB: LoraTrainJobConfig = {
  name: 'my_first_krea2_lora_v1',
  training_folder: 'output',
  device: 'cuda:0',
  trigger_word: '',
  network: {
    type: 'lora',
    linear: 16,
    linear_alpha: 16
  },
  save: {
    dtype: 'fp16',
    save_every: 250,
    max_step_saves_to_keep: 4,
    push_to_hub: false
  },
  datasets: [
    {
      folder_path: '',
      caption_ext: 'txt',
      caption_dropout_rate: 0.05,
      shuffle_tokens: false,
      cache_latents_to_disk: true,
      resolution: [1024]
    }
  ],
  train: {
    batch_size: 1,
    steps: 2000,
    gradient_accumulation_steps: 1,
    train_unet: true,
    train_text_encoder: false,
    gradient_checkpointing: true,
    cache_text_embeddings: true,
    noise_scheduler: 'flowmatch',
    optimizer: 'adamw8bit',
    lr: 1e-4,
    dtype: 'bf16',
    skip_first_sample: false,
    disable_sampling: true,
    ema_config: {
      use_ema: false,
      ema_decay: 0.99
    }
  },
  model: {
    name_or_path: KREA2_RAW,
    train_name_or_path: KREA2_RAW,
    arch: 'krea2',
    quantize: 'none',
    low_vram: false,
    layer_offload: false,
    layer_offload_percent: 0
  },
  sample: {
    sampler: 'flowmatch',
    sample_every: 250,
    sample_start_step: 0,
    width: DEFAULT_SAMPLE_WIDTH,
    height: DEFAULT_SAMPLE_HEIGHT,
    prompts: buildDefaultSamplePrompts(),
    neg: '',
    seed: DEFAULT_SAMPLE_SEED,
    guidance_scale: 3.5,
    sample_steps: 20
  }
}

export const DEFAULT_LORA_TRAIN_APP: LoraTrainAppSettings = {
  pythonPath: '',
  downloadFolder: '',
  huggingfaceToken: ''
}

export const DEFAULT_LORA_TRAIN_JOB_PRESET_ID = 'job-default'

export function createLoraTrainJobId(): string {
  return `job-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`
}

function asRecord(value: unknown): Record<string, unknown> | null {
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    return value as Record<string, unknown>
  }
  return null
}

function asNumber(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback
}

function asBool(value: unknown, fallback: boolean): boolean {
  return typeof value === 'boolean' ? value : fallback
}

function asString(value: unknown, fallback: string): string {
  return typeof value === 'string' ? value : fallback
}

function asNumberArray(value: unknown, fallback: number[]): number[] {
  if (!Array.isArray(value)) return [...fallback]
  const nums = value.filter((v): v is number => typeof v === 'number' && Number.isFinite(v))
  return nums.length > 0 ? nums : [...fallback]
}

function normalizeDataset(raw: unknown, fallback: LoraTrainDatasetConfig): LoraTrainDatasetConfig {
  const o = asRecord(raw)
  if (!o) {
    return {
      ...fallback,
      resolution: [...fallback.resolution]
    }
  }
  return {
    folder_path: asString(o.folder_path, fallback.folder_path),
    caption_ext: 'txt',
    caption_dropout_rate: asNumber(o.caption_dropout_rate, fallback.caption_dropout_rate),
    shuffle_tokens: asBool(o.shuffle_tokens, fallback.shuffle_tokens),
    cache_latents_to_disk: asBool(o.cache_latents_to_disk, fallback.cache_latents_to_disk),
    resolution: asNumberArray(o.resolution, fallback.resolution)
  }
}

function normalizeSaveDtype(value: unknown, fallback: string): string {
  const raw = asString(value, fallback).toLowerCase()
  if (raw === 'float16' || raw === 'fp16') return 'fp16'
  if (raw === 'bfloat16' || raw === 'bf16') return 'bf16'
  if (raw === 'float32' || raw === 'fp32') return 'fp32'
  return fallback === 'float16' ? 'fp16' : fallback
}

function normalizeNetworkType(value: unknown, fallback: string): string {
  const raw = asString(value, fallback).toLowerCase()
  if (raw === 'lora' || raw === 'locon' || raw === 'lokr') return raw
  return 'lora'
}

function normalizeArch(value: unknown): LoraTrainArch {
  return value === 'krea2' || value === 'flux' ? 'krea2' : 'krea2'
}

function normalizeQuantizeMode(
  value: unknown,
  fallback: LoraTrainQuantizeMode
): LoraTrainQuantizeMode {
  if (typeof value === 'boolean') return value ? 'int8' : 'none'
  const raw = asString(value, fallback).toLowerCase()
  if (raw === 'none' || raw === '-none-' || raw === 'off' || raw === 'false') return 'none'
  if (raw === 'qfloat8') return 'qfloat8'
  if (raw === 'float8') return 'float8'
  if (raw === 'int8' || raw === 'true') return 'int8'
  return fallback
}

function normalizeSamplePrompts(
  raw: unknown,
  defaults: {
    width: number
    height: number
    seed: number
    prompts: LoraTrainSamplePrompt[]
  }
): LoraTrainSamplePrompt[] {
  if (!Array.isArray(raw)) {
    return defaults.prompts.map((p) => ({ ...p }))
  }
  if (raw.length === 0) return []

  return raw.map((item, i) => {
    const seedFallback = defaults.seed + i
    if (typeof item === 'string') {
      return {
        prompt: item,
        width: defaults.width,
        height: defaults.height,
        seed: seedFallback
      }
    }
    const o = asRecord(item)
    if (!o) {
      return {
        prompt: '',
        width: defaults.width,
        height: defaults.height,
        seed: seedFallback
      }
    }
    return {
      prompt: asString(o.prompt ?? o.text, ''),
      width: asNumber(o.width, defaults.width),
      height: asNumber(o.height, defaults.height),
      seed: asNumber(o.seed, seedFallback)
    }
  })
}

export function normalizeLoraTrainJob(
  raw: Partial<LoraTrainJobConfig> | null | undefined
): LoraTrainJobConfig {
  const d = DEFAULT_LORA_TRAIN_JOB
  const o = asRecord(raw) ?? {}
  const network = asRecord(o.network) ?? {}
  const save = asRecord(o.save) ?? {}
  const train = asRecord(o.train) ?? {}
  const ema = asRecord(train.ema_config) ?? {}
  const model = asRecord(o.model) ?? {}
  const sample = asRecord(o.sample) ?? {}

  let datasets: LoraTrainDatasetConfig[]
  if (Array.isArray(o.datasets) && o.datasets.length > 0) {
    datasets = o.datasets.map((item, i) =>
      normalizeDataset(item, d.datasets[Math.min(i, d.datasets.length - 1)])
    )
  } else {
    datasets = d.datasets.map((ds) => ({ ...ds, resolution: [...ds.resolution] }))
  }

  const legacyPath = asString(model.name_or_path, d.model.train_name_or_path)
  const trainPath = asString(model.train_name_or_path, legacyPath || d.model.train_name_or_path)

  return {
    name: asString(o.name, d.name),
    training_folder: asString(o.training_folder, d.training_folder),
    device: asString(o.device, d.device),
    trigger_word: asString(o.trigger_word, d.trigger_word),
    network: {
      type: normalizeNetworkType(network.type, d.network.type),
      linear: asNumber(network.linear, d.network.linear),
      linear_alpha: asNumber(network.linear_alpha, d.network.linear_alpha)
    },
    save: {
      dtype: normalizeSaveDtype(save.dtype, d.save.dtype),
      save_every: asNumber(save.save_every, d.save.save_every),
      max_step_saves_to_keep: asNumber(
        save.max_step_saves_to_keep,
        d.save.max_step_saves_to_keep
      ),
      push_to_hub: false
    },
    datasets,
    train: {
      batch_size: asNumber(train.batch_size, d.train.batch_size),
      steps: asNumber(train.steps, d.train.steps),
      gradient_accumulation_steps: asNumber(
        train.gradient_accumulation_steps,
        d.train.gradient_accumulation_steps
      ),
      train_unet: true,
      train_text_encoder: false,
      gradient_checkpointing: asBool(
        train.gradient_checkpointing,
        d.train.gradient_checkpointing
      ),
      cache_text_embeddings: asBool(
        train.cache_text_embeddings,
        d.train.cache_text_embeddings
      ),
      noise_scheduler: asString(train.noise_scheduler, d.train.noise_scheduler),
      optimizer: asString(train.optimizer, d.train.optimizer),
      lr: asNumber(train.lr, d.train.lr),
      dtype: asString(train.dtype, d.train.dtype),
      skip_first_sample: asBool(train.skip_first_sample, d.train.skip_first_sample),
      disable_sampling: asBool(train.disable_sampling, d.train.disable_sampling),
      ema_config: {
        use_ema: asBool(ema.use_ema, d.train.ema_config.use_ema),
        ema_decay: asNumber(ema.ema_decay, d.train.ema_config.ema_decay)
      }
    },
    model: {
      name_or_path: trainPath,
      train_name_or_path: trainPath,
      arch: normalizeArch(model.arch),
      quantize: normalizeQuantizeMode(model.quantize, d.model.quantize),
      low_vram: asBool(model.low_vram, d.model.low_vram),
      layer_offload: asBool(model.layer_offload, d.model.layer_offload),
      layer_offload_percent: (() => {
        // Legacy layer_offload_mode=auto → 0 (Auto). Otherwise clamp percent.
        if (asString(model.layer_offload_mode, '') === 'auto') return 0
        const raw = asNumber(model.layer_offload_percent, d.model.layer_offload_percent)
        return Math.min(100, Math.max(0, Math.round(raw)))
      })()
    },
    sample: {
      sampler: asString(sample.sampler, d.sample.sampler),
      sample_every: asNumber(sample.sample_every, d.sample.sample_every),
      sample_start_step: asNumber(sample.sample_start_step, d.sample.sample_start_step),
      width: asNumber(sample.width, d.sample.width),
      height: asNumber(sample.height, d.sample.height),
      prompts: normalizeSamplePrompts(sample.prompts, {
        width: asNumber(sample.width, d.sample.width),
        height: asNumber(sample.height, d.sample.height),
        seed: asNumber(sample.seed, d.sample.seed),
        prompts: d.sample.prompts
      }),
      neg: asString(sample.neg, d.sample.neg),
      seed: asNumber(sample.seed, d.sample.seed),
      guidance_scale: asNumber(sample.guidance_scale, d.sample.guidance_scale),
      sample_steps: asNumber(sample.sample_steps, d.sample.sample_steps)
    }
  }
}

export function createDefaultLoraTrainJobPreset(
  job?: LoraTrainJobConfig,
  id = DEFAULT_LORA_TRAIN_JOB_PRESET_ID
): LoraTrainJobPreset {
  return {
    id,
    job: normalizeLoraTrainJob(job ?? DEFAULT_LORA_TRAIN_JOB)
  }
}

export function normalizeLoraTrainJobPresets(
  rawJobs: unknown,
  legacyJob: unknown,
  activeIdRaw: unknown
): { jobs: LoraTrainJobPreset[]; activeId: string } {
  const jobs: LoraTrainJobPreset[] = []
  if (Array.isArray(rawJobs)) {
    for (const item of rawJobs) {
      const o = asRecord(item)
      if (!o) continue
      const id = asString(o.id, '')
      if (!id) continue
      jobs.push({
        id,
        job: normalizeLoraTrainJob(o.job ?? o)
      })
    }
  }
  if (jobs.length === 0) {
    jobs.push(createDefaultLoraTrainJobPreset(legacyJob as LoraTrainJobConfig | undefined))
  }
  const activeId =
    typeof activeIdRaw === 'string' && jobs.some((j) => j.id === activeIdRaw)
      ? activeIdRaw
      : jobs[0].id
  return { jobs, activeId }
}

export function normalizeLoraTrainApp(
  raw: Partial<LoraTrainAppSettings> | null | undefined
): LoraTrainAppSettings {
  const d = DEFAULT_LORA_TRAIN_APP
  const o = asRecord(raw) ?? {}
  return {
    pythonPath: asString(o.pythonPath, d.pythonPath),
    downloadFolder: asString(o.downloadFolder, d.downloadFolder),
    huggingfaceToken: asString(o.huggingfaceToken, d.huggingfaceToken)
  }
}

/** Resolved Python install path from downloadFolder; undefined = main default (userData/python). */
export function pythonInstallPathFromDownloadFolder(
  downloadFolder: string
): string | undefined {
  const trimmed = downloadFolder.trim()
  return trimmed ? join(trimmed, 'python') : undefined
}

/** Resolved model download path from downloadFolder; undefined = main default (userData/models). */
export function modelDownloadPathFromDownloadFolder(
  downloadFolder: string
): string | undefined {
  const trimmed = downloadFolder.trim()
  return trimmed ? join(trimmed, 'models') : undefined
}

export function normalizeActiveView(value: unknown): ActiveView {
  return value === 'loraTrain' ? 'loraTrain' : 'datasetEdit'
}
