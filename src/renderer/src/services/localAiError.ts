import type { AppSettings } from '../types'

const CANCEL_MESSAGES = new Set(['Translation cancelled', 'Caption cancelled'])

const NETWORK_RE =
  /failed to fetch|networkerror|network error|load failed|econnrefused|err_connection|enotfound|econnreset/i

export function hostPortFromBaseUrl(baseUrl: string): string {
  try {
    const u = new URL(baseUrl)
    if (u.port) return `${u.hostname}:${u.port}`
    if (u.protocol === 'https:') return `${u.hostname}:443`
    if (u.protocol === 'http:') return `${u.hostname}:80`
    return u.hostname
  } catch {
    return baseUrl.replace(/^https?:\/\//, '').replace(/\/.*$/, '') || baseUrl
  }
}

export function providerLabel(settings: AppSettings): string {
  return settings.provider === 'lmstudio' ? 'LM Studio' : 'Ollama'
}

export function localAiEndpoint(settings: AppSettings): string {
  const base =
    settings.provider === 'lmstudio' ? settings.lmStudioBaseUrl : settings.ollamaBaseUrl
  return hostPortFromBaseUrl(base)
}

export function formatLocalAiError(err: unknown, settings: AppSettings): string {
  const raw = err instanceof Error ? err.message : String(err)
  if (CANCEL_MESSAGES.has(raw)) return raw

  const host = localAiEndpoint(settings)
  const provider = providerLabel(settings)
  const model = settings.model.trim() || '(none)'

  if (NETWORK_RE.test(raw)) {
    return `Failed to connect to [${host}] ${provider} model [${model}]`
  }

  return `${raw} (${provider} [${host}] model [${model}])`
}
