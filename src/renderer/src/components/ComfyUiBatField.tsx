import { useEffect, useRef, useState } from 'react'

interface Props {
  value: string
  onChange: (value: string) => void
  downloadFolder: string
  pythonPath: string
  enabled?: boolean
}

export function ComfyUiBatField({
  value,
  onChange,
  downloadFolder,
  pythonPath,
  enabled = true
}: Props) {
  const [probeOk, setProbeOk] = useState<boolean | null>(null)
  const [probeMsg, setProbeMsg] = useState<string | null>(null)
  const [installing, setInstalling] = useState(false)
  const [installMsg, setInstallMsg] = useState<string | null>(null)
  const [installPct, setInstallPct] = useState(0)
  const probeTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const offProgress = useRef<(() => void) | null>(null)

  const runProbe = async (batPath: string) => {
    if (!batPath.trim()) {
      setProbeOk(false)
      setProbeMsg('Not set — Download installs ComfyUI and writes run_captioer.bat')
      return
    }
    try {
      const result = await window.api.probeComfyBat(batPath.trim())
      setProbeOk(result.ok)
      setProbeMsg(result.message)
    } catch (err) {
      setProbeOk(false)
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

  const browseBat = async () => {
    const file = await window.api.openFile({
      title: 'Select ComfyUI launch bat',
      filters: [
        { name: 'Batch', extensions: ['bat', 'cmd'] },
        { name: 'All Files', extensions: ['*'] }
      ]
    })
    if (!file) return
    onChange(file)
  }

  const startInstall = async () => {
    if (installing) return
    setInstalling(true)
    setInstallMsg('Starting…')
    setInstallPct(0)
    offProgress.current?.()
    offProgress.current = window.api.onComfyInstallProgress((p) => {
      setInstallMsg(p.message)
      setInstallPct(p.pct)
    })
    try {
      const result = await window.api.installComfyUi({
        downloadFolder: downloadFolder.trim() || undefined,
        pythonPath: pythonPath.trim() || undefined
      })
      if (result.ok && result.batPath) {
        setInstallMsg(null)
        onChange(result.batPath)
      } else {
        setInstallMsg(null)
        setProbeOk(false)
        setProbeMsg(result.message || 'Install failed')
      }
    } catch (err) {
      setInstallMsg(null)
      setProbeOk(false)
      setProbeMsg(err instanceof Error ? err.message : String(err))
    } finally {
      offProgress.current?.()
      offProgress.current = null
      setInstalling(false)
    }
  }

  const cancelInstall = async () => {
    await window.api.cancelComfyInstall()
    setInstalling(false)
    setInstallMsg(null)
    setProbeMsg('Cancelled')
    setProbeOk(false)
  }

  const showDownload = !installing && probeOk === false

  return (
    <label className="field">
      <span>ComfyUI launch bat</span>
      <div className="field-row">
        <input
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder="Path to run_captioer.bat (or your ComfyUI .bat)"
          spellCheck={false}
        />
        <button type="button" onClick={() => void browseBat()} disabled={installing}>
          Browse
        </button>
        {showDownload && (
          <button type="button" className="primary" onClick={() => void startInstall()}>
            Download
          </button>
        )}
        {installing && (
          <button type="button" onClick={() => void cancelInstall()}>
            Cancel
          </button>
        )}
      </div>
      {installing && (
        <p className="field-hint">
          Installing… {installPct}% — {installMsg || '…'}
        </p>
      )}
      {!installing && probeMsg && (
        <p className={`field-hint${probeOk === false ? ' field-hint-warn' : ''}`}>{probeMsg}</p>
      )}
      <p className="field-hint">
        Download clones ComfyUI via git into{' '}
        <code>{'{downloadFolder}/ComfyUI'}</code>, installs requirements with the Python above,
        and writes <code>run_captioer.bat</code>. Requires Git on PATH.
      </p>
    </label>
  )
}
