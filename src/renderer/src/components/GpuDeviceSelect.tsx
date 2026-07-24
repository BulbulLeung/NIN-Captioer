import { useCallback, useEffect, useMemo, useState } from 'react'

export interface GpuDeviceOption {
  id: string
  label: string
}

interface Props {
  value: string
  onChange: (deviceId: string) => void
  /** When false, skip auto-fetch (e.g. settings dialog closed). Default true. */
  enabled?: boolean
}

export function GpuDeviceSelect({ value, onChange, enabled = true }: Props) {
  const [devices, setDevices] = useState<GpuDeviceOption[]>([])
  const [loading, setLoading] = useState(false)

  const refresh = useCallback(async () => {
    setLoading(true)
    try {
      const list = await window.api.listGpuDevices()
      setDevices(list)
    } catch {
      setDevices([{ id: 'cuda:0', label: 'cuda:0 (not detected)' }])
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    if (!enabled) return
    void refresh()
  }, [enabled, refresh])

  const options = useMemo(() => {
    if (!value || devices.some((d) => d.id === value)) return devices
    return [{ id: value, label: `${value} (saved)` }, ...devices]
  }, [devices, value])

  return (
    <div className="model-row">
      <select
        value={value || options[0]?.id || 'cuda:0'}
        disabled={loading && options.length === 0}
        aria-label="CUDA device"
        onChange={(e) => onChange(e.target.value)}
      >
        {options.length === 0 ? (
          <option value="cuda:0">{loading ? 'Detecting…' : 'cuda:0'}</option>
        ) : (
          options.map((d) => (
            <option key={d.id} value={d.id}>
              {d.label}
            </option>
          ))
        )}
      </select>
      <button type="button" onClick={() => void refresh()} disabled={loading}>
        {loading ? 'Detecting…' : 'Refresh'}
      </button>
    </div>
  )
}
