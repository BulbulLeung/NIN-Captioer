import { DEFAULT_WD14_SETTINGS } from '../defaults/captionFormats'

const HF_MODELS_URL =
  'https://huggingface.co/api/models?author=SmilingWolf&search=tagger&limit=100&full=true'

const FETCH_TIMEOUT_MS = 20_000

/** Curated fallback when Hugging Face is unreachable. */
export const FALLBACK_WD14_MODEL_REPOS: string[] = [
  'SmilingWolf/wd-swinv2-tagger-v3',
  'SmilingWolf/wd-vit-tagger-v3',
  'SmilingWolf/wd-convnext-tagger-v3',
  'SmilingWolf/wd-vit-large-tagger-v3',
  'SmilingWolf/wd-eva02-large-tagger-v3',
  'SmilingWolf/wd-v1-4-moat-tagger-v2',
  'SmilingWolf/wd-v1-4-swinv2-tagger-v2',
  'SmilingWolf/wd-v1-4-convnext-tagger-v2',
  'SmilingWolf/wd-v1-4-convnextv2-tagger-v2',
  'SmilingWolf/wd-v1-4-vit-tagger-v2'
]

interface HfModelSibling {
  rfilename?: string
}

interface HfModelEntry {
  id?: string
  modelId?: string
  downloads?: number
  lastModified?: string
  tags?: string[]
  siblings?: HfModelSibling[]
}

function hasOnnxTaggerFiles(entry: HfModelEntry): boolean {
  const files = new Set(
    (entry.siblings ?? [])
      .map((s) => (s.rfilename || '').replace(/\\/g, '/').toLowerCase())
      .filter(Boolean)
  )
  if (files.size === 0) {
    // Without siblings, accept onnx-tagged tagger repos by name
    const id = (entry.id || entry.modelId || '').toLowerCase()
    const tags = (entry.tags || []).map((t) => t.toLowerCase())
    return id.includes('tagger') && tags.includes('onnx')
  }
  return files.has('model.onnx') && files.has('selected_tags.csv')
}

function sortRepoIds(ids: string[]): string[] {
  const score = (id: string): number => {
    const lower = id.toLowerCase()
    let s = 0
    if (lower.includes('-v3')) s += 300
    if (lower.includes('-v2')) s += 200
    if (lower.includes('swinv2')) s += 40
    if (lower.includes('eva02')) s += 30
    if (lower.includes('vit-large')) s += 25
    if (lower.includes('vit')) s += 15
    if (lower.includes('convnext')) s += 10
    if (lower.includes('moat')) s += 5
    return s
  }
  return [...ids].sort((a, b) => score(b) - score(a) || a.localeCompare(b))
}

/**
 * List SmilingWolf WD tagger repos that ship ONNX + selected_tags.csv.
 * Fetches from the Hugging Face Hub API; falls back to a curated list on failure.
 */
export async function listWd14ModelRepos(signal?: AbortSignal): Promise<string[]> {
  const ctrl = new AbortController()
  const timer = setTimeout(() => ctrl.abort('timeout'), FETCH_TIMEOUT_MS)
  const onUserAbort = () => ctrl.abort('cancel')
  if (signal) {
    if (signal.aborted) ctrl.abort('cancel')
    else signal.addEventListener('abort', onUserAbort, { once: true })
  }

  try {
    const res = await fetch(HF_MODELS_URL, { signal: ctrl.signal })
    if (!res.ok) {
      throw new Error(`Hugging Face API error ${res.status}: ${res.statusText}`)
    }
    const data = (await res.json()) as HfModelEntry[]
    if (!Array.isArray(data)) {
      throw new Error('Unexpected Hugging Face API response')
    }

    const ids = data
      .filter(hasOnnxTaggerFiles)
      .map((e) => (e.id || e.modelId || '').trim())
      .filter((id) => /^SmilingWolf\/wd-.+-tagger/i.test(id))

    const unique = [...new Set(ids)]
    if (unique.length === 0) {
      throw new Error('No ONNX WD14 tagger models found')
    }
    return sortRepoIds(unique)
  } catch (err) {
    const reason = ctrl.signal.reason
    if (reason === 'cancel') throw new Error('Cancelled')
    if (reason === 'timeout') {
      throw new Error('Timed out while listing WD14 models from Hugging Face')
    }
    throw err instanceof Error ? err : new Error(String(err))
  } finally {
    clearTimeout(timer)
    signal?.removeEventListener('abort', onUserAbort)
  }
}

/** Prefer live list; on failure return fallback (never throws). */
export async function listWd14ModelReposOrFallback(
  signal?: AbortSignal
): Promise<{ repos: string[]; fromNetwork: boolean; error?: string }> {
  try {
    const repos = await listWd14ModelRepos(signal)
    const withDefault = repos.includes(DEFAULT_WD14_SETTINGS.modelRepoId)
      ? repos
      : sortRepoIds([DEFAULT_WD14_SETTINGS.modelRepoId, ...repos])
    return { repos: withDefault, fromNetwork: true }
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err)
    if (message === 'Cancelled') {
      return { repos: [...FALLBACK_WD14_MODEL_REPOS], fromNetwork: false, error: message }
    }
    return {
      repos: [...FALLBACK_WD14_MODEL_REPOS],
      fromNetwork: false,
      error: message
    }
  }
}
