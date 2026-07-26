import { useEffect, useRef, useState, type ReactNode } from 'react'

interface Props {
  value: string
  onChange: (value: string) => void
  installPath: string
  onInstallPathChange: (value: string) => void
  /** When false, skip auto-probe (dialog closed). */
  enabled?: boolean
  /** Optional hint under the field; default explains shared WD14 + LoRA use. */
  hint?: ReactNode
}

const DEFAULT_HINT: ReactNode = (
  <>
    Shared by Dataset Edit (WD14 tagging) and LoRA Train. Leave empty to use{' '}
    <code>python</code> from PATH.
  </>
)

type ProbeStatus = 'ready' | 'missingPython' | 'missingPackages' | 'error' | 'checking'

export function PythonExecutableField({
  value,
  onChange,
  installPath,
  onInstallPathChange,
  enabled = true,
  hint = DEFAULT_HINT
}: Props) {
  const [probeStatus, setProbeStatus] = useState<ProbeStatus>('checking')
  const [probeMsg, setProbeMsg] = useState<string | null>(null)
  const [installing, setInstalling] = useState(false)
  const [installMsg, setInstallMsg] = useState<string | null>(null)
  const [installPct, setInstallPct] = useState(0)
  const probeTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const offProgress = useRef<(() => void) | null>(null)

  const runProbe = async (pythonPath: string) => {
    setProbeStatus('checking')
    setProbeMsg(null)
    try {
      const result = await window.api.probePython(pythonPath.trim() || undefined)
      setProbeStatus(result.status)
      setProbeMsg(result.message)
    } catch (err) {
      setProbeStatus('error')
      setProbeMsg(err instanceof Error ? err.message : String(err))
    }
  }

  useEffect(() => {
    if (!enabled) return
    if (probeTimer.current) clearTimeout(probeTimer.current)
    probeTimer.current = setTimeout(() => {
      void runProbe(value)
    }, 400)
    return () => {
      if (probeTimer.current) clearTimeout(probeTimer.current)
    }
  }, [value, enabled])

  useEffect(() => {
    return () => {
      offProgress.current?.()
      offProgress.current = null
    }
  }, [])

  const browsePython = async () => {
    const file = await window.api.openFile({
      title: 'Select Python executable',
      filters: [
        { name: 'Python', extensions: ['exe'] },
        { name: 'All Files', extensions: ['*'] }
      ]
    })
    if (!file) return
    onChange(file)
  }

  const browseInstallPath = async () => {
    const dir = await window.api.openFolder()
    if (!dir) return
    onInstallPathChange(dir)
  }

  const useDefaultInstallPath = async () => {
    const path = await window.api.defaultPythonInstallPath()
    onInstallPathChange(path)
  }

  const startInstall = async () => {
    if (installing) return
    setInstalling(true)
    setInstallMsg('Starting…')
    setInstallPct(0)
    offProgress.current?.()
    offProgress.current = window.api.onPythonInstallProgress((p) => {
      setInstallMsg(p.message)
      setInstallPct(p.pct)
    })
    try {
      const result = await window.api.installPython({
        installPath: installPath.trim() || undefined
      })
      if (result.ok && result.pythonPath) {
        onChange(result.pythonPath)
        setInstallMsg(result.message)
        setProbeStatus('ready')
        setProbeMsg(result.message)
      } else {
        setInstallMsg(result.message || 'Install failed')
        setProbeStatus('error')
        setProbeMsg(result.message || 'Install failed')
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err)
      setInstallMsg(msg)
      setProbeStatus('error')
      setProbeMsg(msg)
    } finally {
      offProgress.current?.()
      offProgress.current = null
      setInstalling(false)
    }
  }

  const cancelInstall = async () => {
    await window.api.cancelPythonInstall()
    setInstalling(false)
    setInstallMsg('Cancelled')
  }

  const showDownload =
    !installing &&
    (probeStatus === 'missingPython' ||
      probeStatus === 'missingPackages' ||
      probeStatus === 'error')

  return (
    <div className="python-executable-field">
      <label className="field">
        <span>Python executable</span>
        <div className="model-row">
          <input
            type="text"
            value={value}
            onChange={(e) => onChange(e.target.value)}
            placeholder="e.g. C:\Python311\python.exe or python"
            spellCheck={false}
            disabled={installing}
          />
          <button type="button" onClick={() => void browsePython()} disabled={installing}>
            Browse
          </button>
        </div>
      </label>

      <label className="field">
        <span>Python install path</span>
        <div className="model-row">
          <input
            type="text"
            value={installPath}
            onChange={(e) => onInstallPathChange(e.target.value)}
            placeholder="Empty = app userData/python"
            spellCheck={false}
            disabled={installing}
          />
          <button type="button" onClick={() => void browseInstallPath()} disabled={installing}>
            Browse
          </button>
          <button type="button" onClick={() => void useDefaultInstallPath()} disabled={installing}>
            Default
          </button>
        </div>
        <p className="field-hint">
          Used by Download to install Python, create a venv, and install requirements (CUDA torch
          preferred; on Windows also triton-windows).
        </p>
      </label>

      <div className="field">
        <div className="model-row python-probe-row">
          {probeStatus === 'checking' && <p className="field-hint">Checking Python…</p>}
          {probeStatus === 'ready' && probeMsg && <p className="test-ok">{probeMsg}</p>}
          {(probeStatus === 'missingPython' ||
            probeStatus === 'missingPackages' ||
            probeStatus === 'error') &&
            probeMsg && <p className="test-err">{probeMsg}</p>}
          {showDownload && (
            <button type="button" className="primary" onClick={() => void startInstall()}>
              Download
            </button>
          )}
          {installing && (
            <>
              <p className="field-hint">
                {installMsg || 'Installing…'}
                {installPct > 0 ? ` (${installPct}%)` : ''}
              </p>
              <button type="button" className="danger" onClick={() => void cancelInstall()}>
                Cancel
              </button>
            </>
          )}
          {!installing && installMsg && probeStatus === 'ready' && (
            <p className="test-ok">{installMsg}</p>
          )}
        </div>
      </div>

      {hint != null && <p className="field-hint">{hint}</p>}
    </div>
  )
}
