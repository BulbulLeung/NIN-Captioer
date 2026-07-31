import { useEffect, useRef, useState } from 'react'
import type { ResourceStats } from '../types'

/** Always show at least this many rows when data exists (list scrolls if needed). */
const MIN_VRAM_APP_SLOTS = 12
/** Fallback row height when no sample row is measurable. */
const VRAM_APP_ROW_FALLBACK_PX = 22

interface Props {
  device: string
}

function clampPercent(value: number): number {
  if (!Number.isFinite(value)) return 0
  return Math.min(100, Math.max(0, value))
}

function formatGiB(bytes: number): string {
  const gib = bytes / (1024 * 1024 * 1024)
  return `${gib.toFixed(1)} GB`
}

function formatMiBAsGiB(mib: number): string {
  return `${(mib / 1024).toFixed(1)} GB`
}

function formatWatts(value: number): string {
  return `${value.toFixed(0)} W`
}

function formatVramAppMem(mib: number): string {
  if (mib >= 1024) return `${(mib / 1024).toFixed(1)} GB`
  return `${mib < 10 ? mib.toFixed(1) : Math.round(mib)} MB`
}

function MonitorHeader({ label, meta }: { label: string; meta?: string }) {
  return (
    <div className="lora-monitor-header">
      <span className="lora-monitor-header-label">{label}</span>
      {meta ? (
        <span className="lora-monitor-header-meta" title={meta}>
          {meta}
        </span>
      ) : null}
    </div>
  )
}

function Meter({
  label,
  valueLabel,
  percent
}: {
  label: string
  valueLabel: string
  percent: number
}) {
  const pct = clampPercent(percent)
  return (
    <div className="lora-monitor-meter">
      <div className="lora-monitor-meter-head">
        <span>{label}</span>
        <span className="lora-monitor-meter-value">{valueLabel}</span>
      </div>
      <div className="lora-monitor-bar" aria-hidden>
        <div className="lora-monitor-bar-fill" style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}

export function ResourceMonitorPane({ device }: Props) {
  const [stats, setStats] = useState<ResourceStats | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [killingPid, setKillingPid] = useState<number | null>(null)
  const [appSlots, setAppSlots] = useState(MIN_VRAM_APP_SLOTS)
  const appListRef = useRef<HTMLUListElement | null>(null)

  useEffect(() => {
    let cancelled = false

    const poll = async () => {
      try {
        const next = await window.api.getResourceStats(device)
        if (cancelled) return
        setStats(next)
        setError(null)
      } catch (err) {
        if (cancelled) return
        setError(err instanceof Error ? err.message : 'Failed to read stats')
      }
    }

    void poll()
    const id = window.setInterval(() => void poll(), 1000)
    return () => {
      cancelled = true
      window.clearInterval(id)
    }
  }, [device])

  useEffect(() => {
    const el = appListRef.current
    if (!el) return

    const updateSlots = () => {
      const h = el.clientHeight
      if (h <= 0) {
        setAppSlots(MIN_VRAM_APP_SLOTS)
        return
      }
      const styles = getComputedStyle(el)
      const gap = Number.parseFloat(styles.rowGap || styles.gap) || 3.2
      const sample = el.querySelector('.lora-monitor-app-row') as HTMLElement | null
      const rowH = sample?.getBoundingClientRect().height || VRAM_APP_ROW_FALLBACK_PX
      const stride = rowH + gap
      const fit = stride > 0 ? Math.floor((h + gap) / stride) : MIN_VRAM_APP_SLOTS
      setAppSlots(Math.max(MIN_VRAM_APP_SLOTS, Number.isFinite(fit) ? fit : MIN_VRAM_APP_SLOTS))
    }

    updateSlots()
    const ro = new ResizeObserver(() => updateSlots())
    ro.observe(el)
    return () => ro.disconnect()
  }, [stats?.gpu?.apps?.length])

  const onKill = async (pid: number) => {
    setKillingPid(pid)
    try {
      const result = await window.api.killProcess(pid)
      if (!result.ok) {
        setError(result.error || 'Failed to kill process')
      } else {
        setStats((prev) => {
          if (!prev?.gpu) return prev
          return {
            ...prev,
            gpu: {
              ...prev.gpu,
              apps: prev.gpu.apps.filter((a) => a.pid !== pid)
            }
          }
        })
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to kill process')
    } finally {
      setKillingPid(null)
    }
  }

  const ramPct =
    stats && stats.ramTotalBytes > 0
      ? (stats.ramUsedBytes / stats.ramTotalBytes) * 100
      : 0
  const vramPct =
    stats?.gpu && stats.gpu.memTotalMiB > 0
      ? (stats.gpu.memUsedMiB / stats.gpu.memTotalMiB) * 100
      : 0
  const powerPct =
    stats?.gpu &&
    stats.gpu.powerDrawW !== null &&
    stats.gpu.powerLimitW !== null &&
    stats.gpu.powerLimitW > 0
      ? (stats.gpu.powerDrawW / stats.gpu.powerLimitW) * 100
      : 0

  const powerLabel =
    stats?.gpu && stats.gpu.powerDrawW !== null
      ? stats.gpu.powerLimitW !== null
        ? `${formatWatts(stats.gpu.powerDrawW)} / ${formatWatts(stats.gpu.powerLimitW)}`
        : formatWatts(stats.gpu.powerDrawW)
      : '—'

  const appsAll = stats?.gpu?.apps ?? []
  const apps = appsAll.slice(0, appSlots)
  const cpuMeta = stats?.cpuName || '—'
  const gpuMeta = stats?.gpu?.name
    ? `${device || '—'}  ${stats.gpu.name}`
    : device || '—'

  return (
    <aside className="lora-monitor" aria-label="System resource monitor">
      <h3 className="lora-monitor-title">Monitor</h3>

      {error ? <p className="lora-monitor-error">{error}</p> : null}

      <section className="lora-monitor-section">
        <MonitorHeader label="CPU" meta={cpuMeta} />
        <Meter
          label="Usage"
          valueLabel={stats ? `${stats.cpuPercent.toFixed(0)}%` : '—'}
          percent={stats?.cpuPercent ?? 0}
        />
        <Meter
          label="RAM"
          valueLabel={
            stats
              ? `${formatGiB(stats.ramUsedBytes)} / ${formatGiB(stats.ramTotalBytes)}`
              : '—'
          }
          percent={ramPct}
        />
      </section>

      <section className="lora-monitor-section">
        <MonitorHeader label="GPU" meta={gpuMeta} />
        {stats?.gpu ? (
          <>
            <Meter
              label="Usage"
              valueLabel={`${stats.gpu.utilPercent.toFixed(0)}%`}
              percent={stats.gpu.utilPercent}
            />
            <Meter
              label="VRAM"
              valueLabel={`${formatMiBAsGiB(stats.gpu.memUsedMiB)} / ${formatMiBAsGiB(stats.gpu.memTotalMiB)}`}
              percent={vramPct}
            />
            <Meter label="Power" valueLabel={powerLabel} percent={powerPct} />
            <p className="lora-monitor-temp">
              Temp{' '}
              {stats.gpu.tempC !== null ? `${stats.gpu.tempC.toFixed(0)}°C` : '—'}
            </p>

            <div className="lora-monitor-apps">
              <MonitorHeader label="VRAM apps" />
              {appsAll.length === 0 ? (
                <p className="lora-monitor-unavailable">No processes using VRAM</p>
              ) : (
                <ul className="lora-monitor-app-list" ref={appListRef}>
                  {apps.map((app) => {
                    const isSelf = app.name === 'Captioer' && !app.killable
                    const isProtected = !app.killable && !isSelf
                    const killHint = app.killable
                      ? `End ${app.name}`
                      : isSelf
                        ? 'Captioer itself — cannot end'
                        : 'System process — cannot end'
                    const nameClass = isSelf
                      ? 'lora-monitor-app-name lora-monitor-app-name--self'
                      : isProtected
                        ? 'lora-monitor-app-name lora-monitor-app-name--protected'
                        : 'lora-monitor-app-name'
                    return (
                      <li
                        key={app.pid}
                        className="lora-monitor-app-row"
                        title={`PID ${app.pid}`}
                      >
                        <span className={nameClass}>{app.name}</span>
                        <span className="lora-monitor-app-mem">
                          {formatVramAppMem(app.memUsedMiB)}
                        </span>
                        <button
                          type="button"
                          className="lora-monitor-app-kill"
                          disabled={!app.killable || killingPid === app.pid}
                          title={killHint}
                          aria-label={killHint}
                          onClick={() => void onKill(app.pid)}
                        >
                          ×
                        </button>
                      </li>
                    )
                  })}
                </ul>
              )}
            </div>
          </>
        ) : (
          <p className="lora-monitor-unavailable">Unable to read GPU status</p>
        )}
      </section>
    </aside>
  )
}
