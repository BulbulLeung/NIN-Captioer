import type { AppSettings } from '../types'
import { formatLocalAiError } from './localAiError'

const CAPTION_TIMEOUT_MS = 300_000

async function fetchWithTimeout(
  url: string,
  init: RequestInit,
  userSignal: AbortSignal | undefined,
  timeoutMs: number
): Promise<Response> {
  const ctrl = new AbortController()
  const timer = setTimeout(() => ctrl.abort('timeout'), timeoutMs)
  const onUserAbort = () => ctrl.abort('cancel')
  if (userSignal) {
    if (userSignal.aborted) ctrl.abort('cancel')
    else userSignal.addEventListener('abort', onUserAbort, { once: true })
  }
  try {
    return await fetch(url, { ...init, signal: ctrl.signal })
  } catch (err) {
    const reason = ctrl.signal.reason
    if (reason === 'timeout') {
      throw new Error(
        `Caption timed out (${Math.round(timeoutMs / 1000)}s). Check that the vision model is loaded.`
      )
    }
    if (reason === 'cancel') throw new Error('Caption cancelled')
    throw err instanceof Error ? err : new Error(String(err))
  } finally {
    clearTimeout(timer)
    userSignal?.removeEventListener('abort', onUserAbort)
  }
}

/** Pull the clean training caption from a multi-section model reply. */
export function extractFinalCaption(raw: string): string {
  let text = raw.trim()
  text = text.replace(/<think>[\s\S]*?<\/think>/gi, '').trim()

  const section3 = text.match(
    /(?:###\s*3[\s\S]*?100%\s*Clean\s*Text\s*Copy|###\s*3\.?[^\n]*|Final Flux Style Caption\s*:?)\s*([\s\S]*)/i
  )
  if (section3?.[1]) {
    text = section3[1].trim()
  }

  // Drop leading labels / section 2 leftovers
  text = text.replace(/^[\s\S]*?(?:Final Flux Style Caption\s*:?\s*)/i, '').trim()

  // Prefer content after the last horizontal rule or section header noise
  const parts = text.split(/\n#{1,3}\s+/)
  if (parts.length > 1) {
    text = parts[parts.length - 1].trim()
  }

  text = text
    .replace(/^```(?:text|markdown)?\s*/i, '')
    .replace(/```$/i, '')
    .trim()

  // If still multi-paragraph analysis, take the longest paragraph
  const paragraphs = text
    .split(/\n\s*\n/)
    .map((p) => p.replace(/\n/g, ' ').trim())
    .filter((p) => p.length > 40 && !/^[-*#]/.test(p) && !/^(shot type|camera)/i.test(p))

  if (paragraphs.length > 0) {
    paragraphs.sort((a, b) => b.length - a.length)
    text = paragraphs[0]
  }

  return text.trim()
}

function activePresetPrompt(settings: AppSettings): string {
  const preset =
    settings.captionPresets.find((p) => p.id === settings.activeCaptionPresetId) ??
    settings.captionPresets[0]
  return preset?.prompt?.trim() ?? ''
}

function captionModelName(settings: AppSettings): string {
  return settings.model || ''
}

async function captionWithLmStudio(
  settings: AppSettings,
  fullPrompt: string,
  mimeType: string,
  base64: string,
  signal?: AbortSignal
): Promise<string> {
  const model = captionModelName(settings)
  if (!model) throw new Error('Model is required in Settings')

  const url = `${settings.lmStudioBaseUrl.replace(/\/$/, '')}/chat/completions`
  const res = await fetchWithTimeout(
    url,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model,
        temperature: 0.2,
        messages: [
          {
            role: 'user',
            content: [
              { type: 'text', text: fullPrompt },
              {
                type: 'image_url',
                image_url: { url: `data:${mimeType};base64,${base64}` }
              }
            ]
          }
        ]
      })
    },
    signal,
    CAPTION_TIMEOUT_MS
  )

  if (!res.ok) {
    const body = await res.text().catch(() => '')
    throw new Error(`LM Studio caption error ${res.status}: ${body || res.statusText}`)
  }

  const data = (await res.json()) as {
    choices?: { message?: { content?: string } }[]
  }
  const content = data.choices?.[0]?.message?.content?.trim()
  if (!content) throw new Error('LM Studio returned empty caption')
  return extractFinalCaption(content)
}

async function captionWithOllama(
  settings: AppSettings,
  fullPrompt: string,
  base64: string,
  signal?: AbortSignal
): Promise<string> {
  const model = captionModelName(settings)
  if (!model) throw new Error('Model is required in Settings')

  const url = `${settings.ollamaBaseUrl.replace(/\/$/, '')}/api/chat`
  const res = await fetchWithTimeout(
    url,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model,
        stream: false,
        think: false,
        keep_alive: '60m',
        messages: [
          {
            role: 'user',
            content: fullPrompt,
            images: [base64]
          }
        ],
        options: {
          temperature: 0.2,
          num_ctx: 8192,
          num_predict: 2048
        }
      })
    },
    signal,
    CAPTION_TIMEOUT_MS
  )

  if (!res.ok) {
    const body = await res.text().catch(() => '')
    throw new Error(`Ollama caption error ${res.status}: ${body || res.statusText}`)
  }

  const data = (await res.json()) as { message?: { content?: string } }
  const content = data.message?.content?.trim()
  if (!content) throw new Error('Ollama returned empty caption')
  return extractFinalCaption(content)
}

export async function generateCaptionForImage(
  settings: AppSettings,
  imagePath: string,
  signal?: AbortSignal
): Promise<string> {
  const presetPrompt = activePresetPrompt(settings)
  if (!presetPrompt) throw new Error('No caption prompt preset selected')

  try {
    const [{ positivePrompt }, { mimeType, base64 }] = await Promise.all([
      window.api.readImageMeta(imagePath),
      window.api.readImageBase64(imagePath)
    ])

    const pngInfo =
      positivePrompt.trim() || '(no PNG Info / positive prompt found in image metadata)'
    const fullPrompt = `${presetPrompt}\n${pngInfo}`

    if (settings.provider === 'lmstudio') {
      return await captionWithLmStudio(settings, fullPrompt, mimeType, base64, signal)
    }
    return await captionWithOllama(settings, fullPrompt, base64, signal)
  } catch (err) {
    throw new Error(formatLocalAiError(err, settings))
  }
}

function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) throw new Error('Caption cancelled')
}

/** Ensure WD14 ONNX assets exist under the configured download path. */
export async function ensureWd14ModelReady(
  settings: AppSettings,
  signal?: AbortSignal
): Promise<string> {
  throwIfAborted(signal)
  const onAbort = () => {
    void window.api.cancelWd14()
  }
  signal?.addEventListener('abort', onAbort, { once: true })
  try {
    const res = await window.api.ensureWd14Model({
      pythonPath: settings.loraTrainApp.pythonPath || undefined,
      downloadPath: settings.loraTrainApp.modelDownloadPath || undefined,
      token: settings.loraTrainApp.huggingfaceToken || undefined,
      repoId: settings.wd14.modelRepoId
    })
    throwIfAborted(signal)
    if (!res.ok || !res.modelDir) {
      throw new Error(res.error || 'Failed to prepare WD14 model')
    }
    return res.modelDir
  } finally {
    signal?.removeEventListener('abort', onAbort)
  }
}

/**
 * Tag one or more images with WD14 ONNX (model loaded once for the batch).
 * Returns a map of absolute image path → comma-separated tags.
 */
export async function generateWd14TagsForImages(
  settings: AppSettings,
  imagePaths: string[],
  signal?: AbortSignal
): Promise<Map<string, string>> {
  if (imagePaths.length === 0) return new Map()
  throwIfAborted(signal)

  const onAbort = () => {
    void window.api.cancelWd14()
  }
  signal?.addEventListener('abort', onAbort, { once: true })

  try {
    const modelDir = await ensureWd14ModelReady(settings, signal)
    throwIfAborted(signal)

    const res = await window.api.tagWd14({
      pythonPath: settings.loraTrainApp.pythonPath || undefined,
      modelDir,
      threshold: settings.wd14.threshold,
      characterThreshold: settings.wd14.characterThreshold,
      imagePaths,
      ensure: true,
      downloadPath: settings.loraTrainApp.modelDownloadPath || undefined,
      token: settings.loraTrainApp.huggingfaceToken || undefined,
      repoId: settings.wd14.modelRepoId
    })

    throwIfAborted(signal)

    if (!res.ok && res.results.length === 0) {
      throw new Error(res.error || 'WD14 tagging failed')
    }

    const out = new Map<string, string>()
    const byLower = new Map<string, string>()
    for (const item of res.results) {
      if (item.tags != null && item.tags !== '' && !item.error) {
        out.set(item.path, item.tags)
        byLower.set(item.path.replace(/\\/g, '/').toLowerCase(), item.tags)
      }
    }

    // Normalize lookup for Windows path casing / separators
    for (const asked of imagePaths) {
      if (out.has(asked)) continue
      const hit = byLower.get(asked.replace(/\\/g, '/').toLowerCase())
      if (hit) out.set(asked, hit)
    }

    if (imagePaths.length === 1) {
      const only = imagePaths[0]
      const tags = out.get(only)
      if (!tags) {
        const itemErr = res.results.find(
          (r) =>
            r.path.replace(/\\/g, '/').toLowerCase() ===
            only.replace(/\\/g, '/').toLowerCase()
        )
        throw new Error(itemErr?.error || res.error || 'WD14 returned empty tags')
      }
    }

    return out
  } finally {
    signal?.removeEventListener('abort', onAbort)
  }
}

export async function generateWd14TagsForImage(
  settings: AppSettings,
  imagePath: string,
  signal?: AbortSignal
): Promise<string> {
  const map = await generateWd14TagsForImages(settings, [imagePath], signal)
  const tags = map.get(imagePath)
  if (!tags) {
    // Fallback: any single result
    const first = map.values().next().value
    if (first) return first
    throw new Error('WD14 returned empty tags')
  }
  return tags
}

/** Dispatch Auto Caption / reCaption by selected caption format. */
export async function generateCaptionByFormat(
  settings: AppSettings,
  imagePath: string,
  signal?: AbortSignal
): Promise<string> {
  if (settings.captionFormat === 'wd14') {
    return generateWd14TagsForImage(settings, imagePath, signal)
  }
  return generateCaptionForImage(settings, imagePath, signal)
}
