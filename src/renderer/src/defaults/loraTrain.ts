/** Defaults for Captioer native Krea 2 LoRA trainer (train on Raw, sample on Turbo). */

export type ActiveView = 'datasetEdit' | 'loraTrain'

export type LoraTrainArch = 'krea2'

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
  sample_name_or_path: string
  arch: LoraTrainArch
  quantize: boolean
  low_vram: boolean
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

/** Environment / preference settings for the LoraTrain Settings dialog. */
export interface LoraTrainAppSettings {
  pythonPath: string
  huggingfaceToken: string
  /** Local folder for HF model downloads; empty = app userData/models */
  modelDownloadPath: string
  defaultTrainingFolder: string
  defaultDevice: string
  exportDir: string
  exportFileName: string
  /** @deprecated migrated to pythonPath */
  aiToolkitPath?: string
  yamlExportDir?: string
  yamlExportFileName?: string
}

export const KREA2_RAW = 'krea/Krea-2-Raw'
export const KREA2_TURBO = 'krea/Krea-2-Turbo'

export const DEFAULT_SAMPLE_PROMPTS: string[] = [
  'woman with red hair, playing chess at the park, bomb going off in the background',
  'a woman holding a coffee cup, in a beanie, sitting at a cafe',
  'a horse is a DJ at a night club, fish eye lens, smoke machine, lazer lights, holding a martini',
  'a man showing off his cool new t shirt at the beach, a shark is jumping out of the water in the background',
  'a bear building a log cabin in the snow covered mountains',
  'woman playing the guitar, on stage, singing a song, laser lights, punk rocker',
  'hipster man with a beard, building a chair, in a wood shop',
  'photo of a man, white background, medium shot, modeling clothing, studio lighting, white backdrop',
  "a man holding a sign that says, 'this is a sign'",
  'a bulldog, in a post apocalyptic world, with a shotgun, in a leather jacket, in a desert, with a motorcycle'
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
    dtype: 'float16',
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
    sample_name_or_path: KREA2_TURBO,
    arch: 'krea2',
    quantize: false,
    low_vram: false
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
    guidance_scale: 0,
    sample_steps: 8
  }
}

export const DEFAULT_LORA_TRAIN_APP: LoraTrainAppSettings = {
  pythonPath: '',
  huggingfaceToken: '',
  modelDownloadPath: '',
  defaultTrainingFolder: 'output',
  defaultDevice: 'cuda:0',
  exportDir: '',
  exportFileName: 'captioer_krea2_train.json'
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
  if (!o) return { ...fallback, resolution: [...fallback.resolution] }
  return {
    folder_path: asString(o.folder_path, fallback.folder_path),
    caption_ext: 'txt',
    caption_dropout_rate: asNumber(o.caption_dropout_rate, fallback.caption_dropout_rate),
    shuffle_tokens: asBool(o.shuffle_tokens, fallback.shuffle_tokens),
    cache_latents_to_disk: asBool(o.cache_latents_to_disk, fallback.cache_latents_to_disk),
    resolution: asNumberArray(o.resolution, fallback.resolution)
  }
}

function normalizeArch(value: unknown): LoraTrainArch {
  return value === 'krea2' || value === 'flux' ? 'krea2' : 'krea2'
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
  const samplePath = asString(model.sample_name_or_path, d.model.sample_name_or_path)

  return {
    name: asString(o.name, d.name),
    training_folder: asString(o.training_folder, d.training_folder),
    device: asString(o.device, d.device),
    trigger_word: asString(o.trigger_word, d.trigger_word),
    network: {
      type: asString(network.type, d.network.type),
      linear: asNumber(network.linear, d.network.linear),
      linear_alpha: asNumber(network.linear_alpha, d.network.linear_alpha)
    },
    save: {
      dtype: asString(save.dtype, d.save.dtype),
      save_every: asNumber(save.save_every, d.save.save_every),
      max_step_saves_to_keep: asNumber(
        save.max_step_saves_to_keep,
        d.save.max_step_saves_to_keep
      ),
      push_to_hub: asBool(save.push_to_hub, d.save.push_to_hub)
    },
    datasets,
    train: {
      batch_size: asNumber(train.batch_size, d.train.batch_size),
      steps: asNumber(train.steps, d.train.steps),
      gradient_accumulation_steps: asNumber(
        train.gradient_accumulation_steps,
        d.train.gradient_accumulation_steps
      ),
      train_unet: asBool(train.train_unet, d.train.train_unet),
      train_text_encoder: asBool(train.train_text_encoder, d.train.train_text_encoder),
      gradient_checkpointing: asBool(
        train.gradient_checkpointing,
        d.train.gradient_checkpointing
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
      sample_name_or_path: samplePath,
      arch: normalizeArch(model.arch),
      quantize: asBool(model.quantize, d.model.quantize),
      low_vram: asBool(model.low_vram, d.model.low_vram)
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

export function normalizeLoraTrainApp(
  raw: Partial<LoraTrainAppSettings> | null | undefined
): LoraTrainAppSettings {
  const d = DEFAULT_LORA_TRAIN_APP
  const o = asRecord(raw) ?? {}
  const pythonFromLegacy =
    typeof o.aiToolkitPath === 'string' && o.aiToolkitPath && !o.pythonPath
      ? ''
      : asString(o.pythonPath, d.pythonPath)
  return {
    pythonPath: pythonFromLegacy,
    huggingfaceToken: asString(o.huggingfaceToken, d.huggingfaceToken),
    modelDownloadPath: asString(o.modelDownloadPath, d.modelDownloadPath),
    defaultTrainingFolder: asString(o.defaultTrainingFolder, d.defaultTrainingFolder),
    defaultDevice: asString(o.defaultDevice, d.defaultDevice),
    exportDir: asString(
      o.exportDir ?? o.yamlExportDir,
      d.exportDir
    ),
    exportFileName: asString(
      o.exportFileName ?? o.yamlExportFileName,
      d.exportFileName
    )
  }
}

export function normalizeActiveView(value: unknown): ActiveView {
  return value === 'loraTrain' ? 'loraTrain' : 'datasetEdit'
}
