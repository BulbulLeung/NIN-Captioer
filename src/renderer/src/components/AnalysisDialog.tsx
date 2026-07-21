import { useEffect, useRef, useState } from 'react'
import type { AppSettings, ImageItem } from '../types'
import {
  CAPTION_CATEGORIES,
  DETAIL_COLORS,
  emptyClassification,
  type CaptionAnalysisResult,
  type CaptionClassification,
  type DetailCount
} from '../services/captionAnalysis'
import { classifyCaption } from '../services/captionClassify'
import { buildCaptionAnalysisResult } from '../services/loraHealthScore'

interface Props {
  open: boolean
  images: ImageItem[]
  settings: AppSettings
  onClose: () => void
}

interface PieSlice {
  id: string
  label: string
  color: string
  count: number
  percent: number
  path: string
}

function polarToCartesian(cx: number, cy: number, r: number, angleDeg: number) {
  const rad = ((angleDeg - 90) * Math.PI) / 180
  return {
    x: cx + r * Math.cos(rad),
    y: cy + r * Math.sin(rad)
  }
}

function describeArc(
  cx: number,
  cy: number,
  r: number,
  startAngle: number,
  endAngle: number
): string {
  const start = polarToCartesian(cx, cy, r, endAngle)
  const end = polarToCartesian(cx, cy, r, startAngle)
  const largeArc = endAngle - startAngle > 180 ? 1 : 0
  return `M ${cx} ${cy} L ${start.x} ${start.y} A ${r} ${r} 0 ${largeArc} 0 ${end.x} ${end.y} Z`
}

function buildDetailSlices(details: DetailCount[]): PieSlice[] {
  const active = details.filter((d) => d.count > 0)
  const total = active.reduce((sum, d) => sum + d.count, 0)
  if (total === 0) return []

  let angle = 0
  const cx = 100
  const cy = 100
  const r = 88

  return active.map((detail, index) => {
    const percent = (detail.count / total) * 100
    const sweep = (detail.count / total) * 360
    const startAngle = angle
    const endAngle = angle + sweep
    angle = endAngle

    let path: string
    if (sweep >= 359.99) {
      path = [
        `M ${cx} ${cy - r}`,
        `A ${r} ${r} 0 1 1 ${cx} ${cy + r}`,
        `A ${r} ${r} 0 1 1 ${cx} ${cy - r}`,
        'Z'
      ].join(' ')
    } else {
      path = describeArc(cx, cy, r, startAngle, endAngle)
    }

    return {
      id: detail.detailId,
      label: detail.label,
      color: DETAIL_COLORS[index % DETAIL_COLORS.length],
      count: detail.count,
      percent,
      path
    }
  })
}

function CategoryPieChart({
  title,
  details
}: {
  title: string
  details: DetailCount[]
}) {
  const slices = buildDetailSlices(details)
  const hitTotal = slices.reduce((sum, s) => sum + s.count, 0)

  return (
    <div className="category-pie-card">
      <h4 className="category-pie-title">{title}</h4>
      <div className="category-pie-body">
        <div className="pie-chart-wrap category-pie-wrap">
          <svg
            viewBox="0 0 200 200"
            className="pie-chart"
            role="img"
            aria-label={`${title} detail distribution`}
          >
            {slices.length === 0 ? (
              <circle cx="100" cy="100" r="88" className="pie-empty" />
            ) : (
              slices.map((slice) => (
                <path
                  key={slice.id}
                  d={slice.path}
                  fill={slice.color}
                  stroke="var(--bg-input)"
                  strokeWidth="1.5"
                />
              ))
            )}
          </svg>
          {slices.length === 0 && <div className="pie-empty-label">No data</div>}
        </div>
        <ul className="pie-legend category-pie-legend">
          {slices.length === 0 ? (
            <li className="legend-empty">No details detected</li>
          ) : (
            slices.map((slice) => (
              <li key={slice.id}>
                <span className="legend-swatch" style={{ background: slice.color }} />
                <span className="legend-label">{slice.label}</span>
                <span className="legend-stats">
                  {slice.count}
                  <span className="legend-pct">
                    ({hitTotal > 0 ? slice.percent.toFixed(1) : '0.0'}%)
                  </span>
                </span>
              </li>
            ))
          )}
        </ul>
      </div>
    </div>
  )
}

export function AnalysisDialog({ open, images, settings, onClose }: Props) {
  const [analyzing, setAnalyzing] = useState(false)
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<CaptionAnalysisResult | null>(null)
  const loadId = useRef(0)
  const abortRef = useRef<AbortController | null>(null)

  useEffect(() => {
    if (!open) {
      abortRef.current?.abort('cancel')
      abortRef.current = null
      setResult(null)
      setError(null)
      setAnalyzing(false)
      setProgress(null)
      return
    }

    if (images.length === 0) {
      setResult(null)
      setError(null)
      setAnalyzing(false)
      setProgress(null)
      return
    }

    const id = ++loadId.current
    const abort = new AbortController()
    abortRef.current = abort
    setAnalyzing(true)
    setError(null)
    setResult(null)
    setProgress(null)

    void (async () => {
      try {
        if (!settings.model.trim()) {
          throw new Error('Model is required in Settings before analyzing captions.')
        }

        const captions = await Promise.all(
          images.map((img) => window.api.readCaption(img.path))
        )
        if (id !== loadId.current) return

        const nonEmptyIndexes: number[] = []
        for (let i = 0; i < captions.length; i++) {
          if (captions[i].trim()) nonEmptyIndexes.push(i)
        }

        const classifications: CaptionClassification[] = captions.map(() =>
          emptyClassification()
        )

        // Show page immediately with baseline stats (pies empty until AI finishes items)
        setResult(buildCaptionAnalysisResult(captions, classifications))
        setProgress({ done: 0, total: nonEmptyIndexes.length })

        for (let step = 0; step < nonEmptyIndexes.length; step++) {
          if (abort.signal.aborted || id !== loadId.current) return
          const idx = nonEmptyIndexes[step]
          classifications[idx] = await classifyCaption(
            settings,
            captions[idx],
            abort.signal
          )
          if (id !== loadId.current) return
          setResult(buildCaptionAnalysisResult(captions, classifications))
          setProgress({ done: step + 1, total: nonEmptyIndexes.length })
        }
      } catch (err) {
        if (id !== loadId.current) return
        if (abort.signal.aborted) return
        setError(err instanceof Error ? err.message : String(err))
      } finally {
        if (id === loadId.current) {
          setAnalyzing(false)
          setProgress(null)
          if (abortRef.current === abort) abortRef.current = null
        }
      }
    })()

    return () => {
      abort.abort('cancel')
    }
  }, [open, images, settings])

  if (!open) return null

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal modal-wide modal-analysis"
        role="dialog"
        aria-labelledby="analysis-title"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="analysis-header">
          <h2 id="analysis-title">Caption Analysis</h2>
          {analyzing && (
            <span className="analysis-progress" aria-live="polite">
              {progress
                ? `Analyzing ${progress.done} / ${progress.total}…`
                : 'Preparing analysis…'}
            </span>
          )}
        </div>

        {images.length === 0 ? (
          <p className="modal-text">Open a folder that contains images first.</p>
        ) : (
          <>
            {error && <p className="test-err">{error}</p>}

            {result ? (
              <>
                <div className="analysis-dashboard">
                  <div className="stat-card">
                    <div className="stat-card-label">Total Images</div>
                    <div className="stat-card-value">{result.totalImages}</div>
                    <div className="stat-card-meta">
                      Captioned {result.captionedCount} / Missing {result.emptyCount}
                    </div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-card-label">LoRA Health Score</div>
                    <div
                      className={`stat-card-value fitness-score fitness-${result.fitnessLevel}`}
                    >
                      <span
                        className={`fitness-dot fitness-${result.fitnessLevel}`}
                        aria-hidden
                      />
                      {result.fitnessScore}
                      <span className="fitness-unit">/100</span>
                    </div>
                    <div className="stat-card-meta">{result.summary}</div>
                  </div>
                </div>

                {result.healthBreakdown.length > 0 && (
                  <section className="health-breakdown">
                    <h3>Score breakdown</h3>
                    <ul className="health-breakdown-list">
                      {result.healthBreakdown.map((item) => (
                        <li key={item.id}>
                          <div className="health-breakdown-row">
                            <span className="health-breakdown-label">
                              {item.label}
                              <span className="health-breakdown-weight">
                                {Math.round(item.weight * 100)}%
                              </span>
                            </span>
                            <span className="health-breakdown-score">
                              {item.id === 'penalty' ? `−${item.score}` : item.score}
                            </span>
                          </div>
                          <div className="health-bar-track">
                            <div
                              className={`health-bar-fill${item.id === 'penalty' ? ' health-bar-penalty' : ''}`}
                              style={{
                                width: `${Math.min(100, Math.max(0, item.score))}%`
                              }}
                            />
                          </div>
                          <div className="health-breakdown-notes">{item.notes}</div>
                        </li>
                      ))}
                    </ul>
                  </section>
                )}

                {(result.strengths.length > 0 || result.improvements.length > 0) && (
                  <div className="health-advice">
                    <div className="health-advice-col">
                      <h4>Strengths</h4>
                      <ul>
                        {result.strengths.map((s) => (
                          <li key={s}>{s}</li>
                        ))}
                      </ul>
                    </div>
                    <div className="health-advice-col">
                      <h4>Improvements</h4>
                      <ul>
                        {result.improvements.map((s) => (
                          <li key={s}>{s}</li>
                        ))}
                      </ul>
                    </div>
                  </div>
                )}

                <h3>Category Detail Distribution</h3>
                <div className="analysis-category-grid">
                  {CAPTION_CATEGORIES.map((cat) => (
                    <CategoryPieChart
                      key={cat.id}
                      title={cat.label}
                      details={result.categoryDetails[cat.id]}
                    />
                  ))}
                </div>
              </>
            ) : analyzing ? (
              <p className="modal-text">Loading captions…</p>
            ) : null}
          </>
        )}

        <div className="modal-actions">
          <span className="spacer" />
          <button type="button" className="primary" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  )
}
