import type { AppSettings } from '../types'
import {
  buildDetailLookup,
  CAPTION_CATEGORIES,
  emptyClassification,
  type CaptionCategoryId,
  type CaptionClassification
} from './captionAnalysis'

const CLASSIFY_TIMEOUT_MS = 120_000

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
        `Classification timed out (${Math.round(timeoutMs / 1000)}s). Check that Ollama / LM Studio is running and the model is loaded.`
      )
    }
    if (reason === 'cancel') {
      throw new Error('Classification cancelled')
    }
    throw err instanceof Error ? err : new Error(String(err))
  } finally {
    clearTimeout(timer)
    userSignal?.removeEventListener('abort', onUserAbort)
  }
}

function taxonomyPromptBlock(): string {
  return CAPTION_CATEGORIES.map((cat) => {
    const ids = cat.details.map((d) => d.id).join(', ')
    return `- ${cat.id}: [${ids}]`
  }).join('\n')
}

function buildClassifyPrompt(caption: string): string {
  return `Classify this image caption into the given categories.
Only use detail ids from the allowed lists below. Include a detail only if the caption clearly implies it (synonyms OK).
Return ONLY valid JSON with exactly these keys, each an array of detail id strings (empty array if none):
{"subject":[],"camera":[],"clothing":[],"pose":[],"expression":[],"scene":[]}

Allowed detail ids per category:
${taxonomyPromptBlock()}

Caption:
${caption}`
}

function stripThinkTags(text: string): string {
  return text.replace(/<think>[\s\S]*?<\/think>/gi, '').trim()
}

function extractJsonObject(raw: string): string | null {
  const cleaned = stripThinkTags(raw)
    .replace(/^```(?:json)?\s*/i, '')
    .replace(/```$/i, '')
    .trim()

  const start = cleaned.indexOf('{')
  const end = cleaned.lastIndexOf('}')
  if (start < 0 || end <= start) return null
  return cleaned.slice(start, end + 1)
}

const CATEGORY_IDS: CaptionCategoryId[] = [
  'subject',
  'camera',
  'clothing',
  'pose',
  'expression',
  'scene'
]

export function parseClassificationResponse(raw: string): CaptionClassification {
  const lookup = buildDetailLookup()
  const empty = emptyClassification()
  const jsonStr = extractJsonObject(raw)
  if (!jsonStr) return empty

  let parsed: unknown
  try {
    parsed = JSON.parse(jsonStr)
  } catch {
    return empty
  }

  if (!parsed || typeof parsed !== 'object') return empty

  const obj = parsed as Record<string, unknown>
  const result = emptyClassification()

  for (const catId of CATEGORY_IDS) {
    const value = obj[catId]
    if (!Array.isArray(value)) continue
    const allowed = lookup[catId]
    const ids: string[] = []
    for (const item of value) {
      if (typeof item !== 'string') continue
      const key = item.trim().toLowerCase()
      const canonical = allowed.get(key)
      if (canonical && !ids.includes(canonical)) ids.push(canonical)
    }
    result[catId] = ids
  }

  return result
}

async function classifyWithLmStudio(
  baseUrl: string,
  model: string,
  caption: string,
  signal?: AbortSignal
): Promise<string> {
  if (!model) throw new Error('Model is required in Settings')

  const url = `${baseUrl.replace(/\/$/, '')}/chat/completions`
  const res = await fetchWithTimeout(
    url,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model,
        messages: [
          {
            role: 'system',
            content: 'You classify captions. Reply with JSON only. No markdown, no explanations.'
          },
          {
            role: 'user',
            content: buildClassifyPrompt(caption)
          }
        ],
        temperature: 0.1,
        max_tokens: 512
      })
    },
    signal,
    CLASSIFY_TIMEOUT_MS
  )

  if (!res.ok) {
    const body = await res.text().catch(() => '')
    throw new Error(`LM Studio error ${res.status}: ${body || res.statusText}`)
  }

  const data = (await res.json()) as {
    choices?: { message?: { content?: string } }[]
  }
  const content = data.choices?.[0]?.message?.content?.trim()
  if (!content) throw new Error('LM Studio returned empty classification')
  return content
}

async function classifyWithOllama(
  baseUrl: string,
  model: string,
  caption: string,
  signal?: AbortSignal
): Promise<string> {
  if (!model) throw new Error('Ollama model name is required in Settings')

  const url = `${baseUrl.replace(/\/$/, '')}/api/chat`
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
            role: 'system',
            content: 'You classify captions. Reply with JSON only. No markdown, no explanations. No thinking tags.'
          },
          {
            role: 'user',
            content: buildClassifyPrompt(caption)
          }
        ],
        options: {
          temperature: 0.1,
          num_ctx: 4096,
          num_predict: 512
        }
      })
    },
    signal,
    CLASSIFY_TIMEOUT_MS
  )

  if (!res.ok) {
    const body = await res.text().catch(() => '')
    throw new Error(`Ollama error ${res.status}: ${body || res.statusText}`)
  }

  const data = (await res.json()) as { message?: { content?: string } }
  let content = data.message?.content?.trim() ?? ''
  content = stripThinkTags(content)
  if (!content) throw new Error('Ollama returned empty classification')
  return content
}

async function classifyOnce(
  settings: AppSettings,
  caption: string,
  signal?: AbortSignal
): Promise<CaptionClassification> {
  const raw =
    settings.provider === 'lmstudio'
      ? await classifyWithLmStudio(
          settings.lmStudioBaseUrl,
          settings.model,
          caption,
          signal
        )
      : await classifyWithOllama(settings.ollamaBaseUrl, settings.model, caption, signal)

  return parseClassificationResponse(raw)
}

/** Classify one caption via local AI. Retries once if the first parse yields all-empty. */
export async function classifyCaption(
  settings: AppSettings,
  caption: string,
  signal?: AbortSignal
): Promise<CaptionClassification> {
  const trimmed = caption.trim()
  if (!trimmed) return emptyClassification()

  const first = await classifyOnce(settings, trimmed, signal)
  const anyHit = CAPTION_CATEGORIES.some((cat) => first[cat.id].length > 0)
  if (anyHit) return first

  // Retry once in case the model returned non-JSON on the first try
  const second = await classifyOnce(settings, trimmed, signal)
  return second
}
