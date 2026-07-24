/** Minimal path join for renderer (avoid Node path dependency). */
export function join(...parts: string[]): string {
  const filtered = parts.filter((p) => p != null && p !== '')
  if (filtered.length === 0) return ''
  const sep = filtered.some((p) => p.includes('\\')) ? '\\' : '/'
  return filtered
    .map((part, i) => {
      let s = part.replace(/[/\\]+/g, sep)
      if (i > 0) {
        while (s.startsWith(sep)) s = s.slice(sep.length)
      }
      if (i < filtered.length - 1) {
        while (s.endsWith(sep)) s = s.slice(0, -sep.length)
      }
      return s
    })
    .join(sep)
}
