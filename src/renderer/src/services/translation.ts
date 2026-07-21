import type { AppSettings } from '../types'
import { LANGUAGES } from '../types'

const TRANSLATE_TIMEOUT_MS = 120_000
const CACHE_MAX = 80

/** In-memory cache: avoid re-calling the model for the same caption. */
const translationCache = new Map<string, string>()

function languageName(code: string): string {
  return LANGUAGES.find((l) => l.code === code)?.label ?? code
}

function cacheKey(
  settings: AppSettings,
  text: string,
  direction: 'en-to-target' | 'target-to-en'
): string {
  const base =
    settings.provider === 'lmstudio' ? settings.lmStudioBaseUrl : settings.ollamaBaseUrl
  return [
    settings.provider,
    settings.model,
    settings.targetLanguage,
    direction,
    base,
    text
  ].join('\0')
}

function cacheGet(key: string): string | undefined {
  const hit = translationCache.get(key)
  if (hit === undefined) return undefined
  translationCache.delete(key)
  translationCache.set(key, hit)
  return hit
}

function cacheSet(key: string, value: string): void {
  if (translationCache.has(key)) translationCache.delete(key)
  translationCache.set(key, value)
  while (translationCache.size > CACHE_MAX) {
    const oldest = translationCache.keys().next().value
    if (oldest === undefined) break
    translationCache.delete(oldest)
  }
}

function buildPrompt(text: string, sourceLang: string, targetLang: string): string {
  return `Translate from ${sourceLang} to ${targetLang}. Output only the translation.\n\n${text}`
}

function maxPredictTokens(textLen: number): number {
  return Math.min(2048, Math.max(256, Math.ceil(textLen / 2) + 128))
}

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
        `Translation timed out (${Math.round(timeoutMs / 1000)}s). Check that Ollama / LM Studio is running and the model is loaded.`
      )
    }
    if (reason === 'cancel') {
      throw new Error('Translation cancelled')
    }
    throw err instanceof Error ? err : new Error(String(err))
  } finally {
    clearTimeout(timer)
    userSignal?.removeEventListener('abort', onUserAbort)
  }
}

async function translateWithLmStudio(
  baseUrl: string,
  model: string,
  text: string,
  sourceLang: string,
  targetLang: string,
  signal?: AbortSignal
): Promise<string> {
  const url = `${baseUrl.replace(/\/$/, '')}/chat/completions`
  const res = await fetchWithTimeout(
    url,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: model || 'local-model',
        messages: [
          {
            role: 'system',
            content: 'Translate only. No explanations.'
          },
          {
            role: 'user',
            content: buildPrompt(text, sourceLang, targetLang)
          }
        ],
        temperature: 0.1,
        max_tokens: maxPredictTokens(text.length)
      })
    },
    signal,
    TRANSLATE_TIMEOUT_MS
  )

  if (!res.ok) {
    const body = await res.text().catch(() => '')
    throw new Error(`LM Studio error ${res.status}: ${body || res.statusText}`)
  }

  const data = (await res.json()) as {
    choices?: { message?: { content?: string } }[]
  }
  const content = data.choices?.[0]?.message?.content?.trim()
  if (!content) throw new Error('LM Studio returned empty translation')
  return content
}

async function translateWithOllama(
  baseUrl: string,
  model: string,
  text: string,
  sourceLang: string,
  targetLang: string,
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
            content: 'Translate only. No explanations. No thinking tags.'
          },
          {
            role: 'user',
            content: buildPrompt(text, sourceLang, targetLang)
          }
        ],
        options: {
          temperature: 0.1,
          num_ctx: 2048,
          num_predict: maxPredictTokens(text.length)
        }
      })
    },
    signal,
    TRANSLATE_TIMEOUT_MS
  )

  if (!res.ok) {
    const body = await res.text().catch(() => '')
    throw new Error(`Ollama error ${res.status}: ${body || res.statusText}`)
  }

  const data = (await res.json()) as { message?: { content?: string } }
  let content = data.message?.content?.trim() ?? ''
  content = content.replace(/<think>[\s\S]*?<\/think>/gi, '').trim()
  if (!content) throw new Error('Ollama returned empty translation')
  return content
}

export async function translateText(
  settings: AppSettings,
  text: string,
  direction: 'en-to-target' | 'target-to-en',
  signal?: AbortSignal
): Promise<string> {
  const trimmed = text.trim()
  if (!trimmed) return ''

  const key = cacheKey(settings, trimmed, direction)
  const cached = cacheGet(key)
  if (cached !== undefined) return cached

  const targetName = languageName(settings.targetLanguage)
  const sourceLang = direction === 'en-to-target' ? 'English' : targetName
  const targetLang = direction === 'en-to-target' ? targetName : 'English'

  const result =
    settings.provider === 'lmstudio'
      ? await translateWithLmStudio(
          settings.lmStudioBaseUrl,
          settings.model,
          trimmed,
          sourceLang,
          targetLang,
          signal
        )
      : await translateWithOllama(
          settings.ollamaBaseUrl,
          settings.model,
          trimmed,
          sourceLang,
          targetLang,
          signal
        )

  cacheSet(key, result)
  return result
}

export async function listModels(settings: AppSettings): Promise<string[]> {
  if (settings.provider === 'lmstudio') {
    const base = settings.lmStudioBaseUrl.replace(/\/$/, '')
    const res = await fetch(`${base}/models`)
    if (!res.ok) throw new Error(`Cannot reach LM Studio (${res.status})`)
    const data = (await res.json()) as { data?: { id?: string }[] }
    return (data.data?.map((m) => m.id).filter((id): id is string => Boolean(id)) ?? []).sort(
      (a, b) => a.localeCompare(b)
    )
  }

  const base = settings.ollamaBaseUrl.replace(/\/$/, '')
  const res = await fetch(`${base}/api/tags`)
  if (!res.ok) throw new Error(`Cannot reach Ollama (${res.status})`)
  const data = (await res.json()) as { models?: { name?: string }[] }
  return (data.models?.map((m) => m.name).filter((name): name is string => Boolean(name)) ?? []).sort(
    (a, b) => a.localeCompare(b)
  )
}

export async function testConnection(settings: AppSettings): Promise<string> {
  const models = await listModels(settings)
  const providerLabel = settings.provider === 'lmstudio' ? 'LM Studio' : 'Ollama'
  if (models.length === 0) {
    return `Connected to ${providerLabel} (no models detected)`
  }
  const preview = models.slice(0, 5).join(', ')
  const more = models.length > 5 ? '…' : ''
  return `Connected to ${providerLabel}, ${models.length} model(s): ${preview}${more}`
}
