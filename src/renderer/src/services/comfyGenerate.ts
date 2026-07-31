/** ComfyUI client via main-process HTTP proxy (renderer fetch is blocked cross-origin). */

export const COMFY_BASE_URL = 'http://127.0.0.1:8188'

export interface ComfyGenerateParams {
  prompt: string
  negative: string
  steps: number
  cfg: number
  seed: number
  width: number
  height: number
  sampler: string
  scheduler: string
  /** Filename under diffusion_models / unet. */
  ditName: string
  /** Filename under vae. */
  vaeName: string
  /** Qwen / text-encoder filename under clip / text_encoders. */
  t5Name: string
  /** LoRA filename under loras/. */
  loraName: string
  loraStrength: number
  /** Unique SaveImage prefix so Comfy does not return a cached (already-moved) file. */
  savePrefix?: string
}

export interface ComfyImageRef {
  filename: string
  subfolder: string
  type: string
}

function clientId(): string {
  return `captioer-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`
}

async function comfyHttp(
  url: string,
  init?: { method?: string; body?: string; timeoutMs?: number }
): Promise<{ ok: boolean; status: number; text: string; error?: string }> {
  return window.api.comfyHttpRequest({
    url,
    method: init?.method || 'GET',
    headers: init?.body ? { 'Content-Type': 'application/json' } : undefined,
    body: init?.body,
    timeoutMs: init?.timeoutMs
  })
}

/** Build API-format prompt graph for native Krea2 (CLIPLoader type=krea2 + Qwen TE). */
export function buildKrea2TestWorkflow(p: ComfyGenerateParams): Record<string, unknown> {
  const strength = p.loraStrength ?? 1
  const clipName = (p.t5Name || '').trim()

  return {
    '1': {
      class_type: 'UNETLoader',
      inputs: {
        unet_name: p.ditName,
        weight_dtype: 'default'
      }
    },
    '10': {
      class_type: 'CLIPLoader',
      inputs: {
        clip_name: clipName,
        type: 'krea2',
        device: 'default'
      }
    },
    '11': {
      class_type: 'VAELoader',
      inputs: {
        vae_name: p.vaeName
      }
    },
    '2': {
      class_type: 'LoraLoader',
      inputs: {
        model: ['1', 0],
        clip: ['10', 0],
        lora_name: p.loraName,
        strength_model: strength,
        strength_clip: strength
      }
    },
    '3': {
      class_type: 'CLIPTextEncode',
      inputs: {
        text: p.prompt,
        clip: ['2', 1]
      }
    },
    '4': {
      class_type: 'CLIPTextEncode',
      inputs: {
        text: p.negative || '',
        clip: ['2', 1]
      }
    },
    '6': {
      class_type: 'EmptySD3LatentImage',
      inputs: {
        width: p.width,
        height: p.height,
        batch_size: 1
      }
    },
    '7': {
      class_type: 'KSampler',
      inputs: {
        seed: p.seed,
        steps: p.steps,
        cfg: p.cfg,
        sampler_name: p.sampler,
        scheduler: p.scheduler,
        denoise: 1,
        model: ['2', 0],
        positive: ['3', 0],
        negative: ['4', 0],
        latent_image: ['6', 0]
      }
    },
    '8': {
      class_type: 'VAEDecode',
      inputs: {
        samples: ['7', 0],
        vae: ['11', 0]
      }
    },
    '9': {
      class_type: 'SaveImage',
      inputs: {
        filename_prefix: (p.savePrefix || 'Captioer_LoraTest').trim() || 'Captioer_LoraTest',
        images: ['8', 0]
      }
    }
  }
}

async function queuePrompt(
  workflow: Record<string, unknown>,
  baseUrl: string,
  cid: string
): Promise<string> {
  const url = `${baseUrl}/prompt`
  const res = await comfyHttp(url, {
    method: 'POST',
    body: JSON.stringify({ prompt: workflow, client_id: cid }),
    timeoutMs: 60_000
  })
  if (res.error && !res.status) {
    throw new Error(`ComfyUI /prompt failed: ${res.error}`)
  }
  if (!res.ok) {
    throw new Error(`ComfyUI /prompt failed (${res.status}): ${res.text.slice(0, 500)}`)
  }
  let data: { prompt_id?: string }
  try {
    data = JSON.parse(res.text) as { prompt_id?: string }
  } catch {
    throw new Error(`ComfyUI /prompt invalid JSON: ${res.text.slice(0, 300)}`)
  }
  if (!data.prompt_id) {
    throw new Error(`ComfyUI rejected prompt: ${res.text.slice(0, 500)}`)
  }
  return data.prompt_id
}

type HistoryStatus = {
  completed?: boolean
  status_str?: string
  messages?: Array<[string, Record<string, unknown>?]>
}

async function historyEntry(
  promptId: string,
  baseUrl: string
): Promise<{
  outputs?: Record<string, { images?: ComfyImageRef[] }>
  status?: HistoryStatus
} | null> {
  const res = await comfyHttp(`${baseUrl}/history/${promptId}`, { timeoutMs: 30_000 })
  if (!res.ok) return null
  try {
    const data = JSON.parse(res.text) as Record<
      string,
      {
        outputs?: Record<string, { images?: ComfyImageRef[] }>
        status?: HistoryStatus
      }
    >
    return data[promptId] ?? null
  } catch {
    return null
  }
}

function formatComfyExecutionError(entry: {
  status?: HistoryStatus
}): string {
  const messages = entry.status?.messages || []
  for (const msg of messages) {
    if (!Array.isArray(msg) || msg[0] !== 'execution_error') continue
    const detail = (msg[1] || {}) as {
      exception_type?: string
      exception_message?: string
      node_id?: string
      node_type?: string
    }
    const type = detail.exception_type || 'Error'
    const text = (detail.exception_message || '').trim().replace(/\s+/g, ' ')
    const node =
      detail.node_type && detail.node_id
        ? ` @ ${detail.node_type}#${detail.node_id}`
        : detail.node_type
          ? ` @ ${detail.node_type}`
          : ''
    if (text) return `ComfyUI execution error${node}: ${type}: ${text.slice(0, 400)}`
  }
  return 'ComfyUI execution error'
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  if (signal?.aborted) return Promise.reject(new Error('Cancelled'))
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort)
      resolve()
    }, ms)
    const onAbort = () => {
      clearTimeout(timer)
      reject(new Error('Cancelled'))
    }
    signal?.addEventListener('abort', onAbort, { once: true })
  })
}

/** Stop the running prompt and clear any queued prompts in ComfyUI. */
export async function interruptComfyGeneration(baseUrl = COMFY_BASE_URL): Promise<void> {
  await Promise.allSettled([
    comfyHttp(`${baseUrl}/interrupt`, { method: 'POST', timeoutMs: 5000 }),
    comfyHttp(`${baseUrl}/queue`, {
      method: 'POST',
      body: JSON.stringify({ clear: true }),
      timeoutMs: 5000
    })
  ])
}

/** Poll /history instead of renderer WebSocket (same CORS issue). */
async function waitForPromptDone(
  promptId: string,
  baseUrl: string,
  signal?: AbortSignal
): Promise<void> {
  const started = Date.now()
  while (Date.now() - started < 600_000) {
    if (signal?.aborted) throw new Error('Cancelled')
    const entry = await historyEntry(promptId, baseUrl)
    if (signal?.aborted) throw new Error('Cancelled')
    if (entry) {
      const statusStr = entry.status?.status_str || ''
      if (statusStr === 'error') {
        throw new Error(formatComfyExecutionError(entry))
      }
      if (entry.outputs && Object.keys(entry.outputs).length > 0) {
        return
      }
      if (entry.status?.completed) {
        return
      }
    }
    await sleep(750, signal)
  }
  throw new Error('Timed out waiting for ComfyUI generation')
}

async function historyImages(
  promptId: string,
  baseUrl: string
): Promise<ComfyImageRef[]> {
  const entry = await historyEntry(promptId, baseUrl)
  if (!entry?.outputs) return []
  const images: ComfyImageRef[] = []
  for (const node of Object.values(entry.outputs)) {
    if (node.images?.length) images.push(...node.images)
  }
  return images
}

export function comfyViewUrl(img: ComfyImageRef, baseUrl = COMFY_BASE_URL): string {
  const q = new URLSearchParams({
    filename: img.filename,
    subfolder: img.subfolder || '',
    type: img.type || 'output'
  })
  return `${baseUrl}/view?${q.toString()}`
}

/** Prefer local-file:// (Electron-safe) over http://127.0.0.1:8188/view (often broken in img). */
async function comfyDisplayUrl(img: ComfyImageRef, baseUrl = COMFY_BASE_URL): Promise<string> {
  const resolved = await window.api.comfyResolveImagePath(img)
  if (resolved.ok && resolved.path) {
    return window.api.toLocalUrl(resolved.path)
  }
  return comfyViewUrl(img, baseUrl)
}

export async function generateWithComfy(
  params: ComfyGenerateParams,
  opts?: { baseUrl?: string; signal?: AbortSignal }
): Promise<{ imageUrl: string; images: ComfyImageRef[]; filePath?: string }> {
  const baseUrl = opts?.baseUrl || COMFY_BASE_URL
  const probe = await comfyHttp(`${baseUrl}/system_stats`, { timeoutMs: 5000 })
  if (!probe.ok) {
    throw new Error(
      `ComfyUI not reachable (${probe.status || 0}): ${probe.error || probe.text || 'offline'}`
    )
  }
  const savePrefix = `Captioer_LoraTest_${Date.now().toString(36)}_${Math.random()
    .toString(36)
    .slice(2, 8)}`
  const workflow = buildKrea2TestWorkflow({ ...params, savePrefix })
  const cid = clientId()
  if (opts?.signal?.aborted) throw new Error('Cancelled')
  const promptId = await queuePrompt(workflow, baseUrl, cid)
  if (opts?.signal?.aborted) throw new Error('Cancelled')
  await waitForPromptDone(promptId, baseUrl, opts?.signal)
  const images = await historyImages(promptId, baseUrl)
  if (!images.length) throw new Error('ComfyUI finished but returned no images')
  const imageUrl = await comfyDisplayUrl(images[0], baseUrl)
  const pathRes = await window.api.comfyResolveImagePath(images[0])
  return { imageUrl, images, filePath: pathRes.path }
}

export function loraNameFromPath(fullPath: string): string {
  const parts = fullPath.replace(/\\/g, '/').split('/')
  return parts[parts.length - 1] || fullPath
}

export function parentDir(fullPath: string): string {
  const normalized = fullPath.replace(/\\/g, '/')
  const idx = normalized.lastIndexOf('/')
  return idx >= 0 ? normalized.slice(0, idx) : ''
}
