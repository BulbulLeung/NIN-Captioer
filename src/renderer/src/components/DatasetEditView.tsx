import type { CSSProperties, PointerEvent as ReactPointerEvent } from 'react'
import { useCallback, useMemo, useState } from 'react'
import { CaptionEditor } from './CaptionEditor'
import { ImageList } from './ImageList'
import { ImagePreview } from './ImagePreview'
import { assignBucket } from '../utils/arBuckets'
import type { AppSettings, ImageItem, ListViewMode } from '../types'

export interface DatasetEditViewProps {
  settings: AppSettings
  images: ImageItem[]
  selectedPath: string | null
  dirtyPaths: Set<string>
  busyPaths: Set<string>
  folder: string | null
  missingCount: number
  batchCaptioning: boolean
  captionBusy: boolean
  english: string
  translated: string
  translating: boolean
  translatingPath: string | null
  captioningPath: string | null
  dirty: boolean
  imageUrl: string | null
  /** Active LoraTrain job resolution tiers for AR bucket preview. */
  trainResolutions: number[]
  draggingPane: 'sidebar' | 'right' | null
  onStopBatchCaption: () => void
  onStartAutoCaption: () => void
  onSetListViewMode: (mode: ListViewMode) => void
  onThumbnailWidthChange: (value: number) => void
  onThumbnailWidthCommit: (value: number) => void
  onBucketPreviewChange: (value: boolean) => void
  onSelectImage: (path: string) => void
  onStartPaneResize: (
    which: 'sidebar' | 'right'
  ) => (e: ReactPointerEvent<HTMLButtonElement>) => void
  onEnglishChange: (value: string) => void
  onTranslatedChange: (value: string) => void
  onLanguageChange: (code: string) => void
  onSave: () => void
  onRecaption: () => void
}

export function DatasetEditView({
  settings,
  images,
  selectedPath,
  dirtyPaths,
  busyPaths,
  folder,
  missingCount,
  batchCaptioning,
  captionBusy,
  english,
  translated,
  translating,
  translatingPath,
  captioningPath,
  dirty,
  imageUrl,
  trainResolutions,
  draggingPane,
  onStopBatchCaption,
  onStartAutoCaption,
  onSetListViewMode,
  onThumbnailWidthChange,
  onThumbnailWidthCommit,
  onBucketPreviewChange,
  onSelectImage,
  onStartPaneResize,
  onEnglishChange,
  onTranslatedChange,
  onLanguageChange,
  onSave,
  onRecaption
}: DatasetEditViewProps) {
  const [naturalSize, setNaturalSize] = useState<{ w: number; h: number } | null>(null)

  const onNaturalSize = useCallback((size: { w: number; h: number } | null) => {
    setNaturalSize(size)
  }, [])

  const bucket = useMemo(() => {
    if (!naturalSize) return null
    return assignBucket(naturalSize.w, naturalSize.h, trainResolutions)
  }, [naturalSize, trainResolutions])

  const bucketPreview = settings.bucketPreview

  return (
    <div
      className="main"
      style={
        {
          '--sidebar-width': `${settings.sidebarWidth}px`,
          '--right-pane-width': `${settings.rightPaneWidth}px`
        } as CSSProperties
      }
    >
      <aside className="sidebar">
        <div className="list-toolbar">
          <div className="list-toolbar-left">
            {batchCaptioning ? (
              <button type="button" onClick={onStopBatchCaption}>
                Cancel Auto Caption
              </button>
            ) : (
              <button
                type="button"
                onClick={onStartAutoCaption}
                disabled={!folder || missingCount === 0 || captionBusy}
                title={
                  missingCount === 0
                    ? 'All images already have .txt captions'
                    : `Caption ${missingCount} image(s) without .txt`
                }
              >
                Auto Caption{missingCount > 0 ? ` (${missingCount})` : ''}
              </button>
            )}
          </div>
          <div className="list-toolbar-right">
            <button
              type="button"
              className={`list-toolbar-btn${settings.listViewMode === 'list' ? ' active' : ''}`}
              aria-pressed={settings.listViewMode === 'list'}
              title="List view"
              onClick={() => onSetListViewMode('list')}
            >
              <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
                <path
                  fill="currentColor"
                  d="M1 2.5h12v1.5H1V2.5zm0 4h12V8H1V6.5zm0 4h12V12H1v-1.5z"
                />
              </svg>
            </button>
            <button
              type="button"
              className={`list-toolbar-btn${
                settings.listViewMode === 'thumbnails' ? ' active' : ''
              }`}
              aria-pressed={settings.listViewMode === 'thumbnails'}
              title="Thumbnail view"
              onClick={() => onSetListViewMode('thumbnails')}
            >
              <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
                <path
                  fill="currentColor"
                  d="M1 1h5v5H1V1zm7 0h5v5H8V1zM1 8h5v5H1V8zm7 0h5v5H8V8z"
                />
              </svg>
            </button>
            <input
              type="range"
              className="list-toolbar-slider"
              min={48}
              max={160}
              step={4}
              value={settings.thumbnailWidth}
              disabled={settings.listViewMode !== 'thumbnails'}
              title="Thumbnail width"
              aria-label="Thumbnail width"
              onChange={(e) => onThumbnailWidthChange(Number(e.target.value))}
              onPointerUp={(e) => {
                onThumbnailWidthCommit(Number(e.currentTarget.value))
                e.currentTarget.blur()
              }}
              onKeyUp={(e) => {
                onThumbnailWidthCommit(Number(e.currentTarget.value))
                e.currentTarget.blur()
              }}
            />
          </div>
        </div>
        <div className="sidebar-list">
          <ImageList
            images={images}
            selectedPath={selectedPath}
            dirtyPaths={dirtyPaths}
            busyPaths={busyPaths}
            viewMode={settings.listViewMode}
            thumbnailWidth={settings.thumbnailWidth}
            onSelect={onSelectImage}
          />
        </div>
      </aside>

      <button
        type="button"
        className={`pane-resizer${draggingPane === 'sidebar' ? ' dragging' : ''}`}
        aria-label="Resize image list"
        title="Drag to resize"
        onPointerDown={onStartPaneResize('sidebar')}
      />

      <section className="center-pane">
        <div className="preview-toolbar">
          <div className="preview-toolbar-left" aria-live="polite">
            {naturalSize ? (
              <>
                <span className="preview-size-label" title="Original image size">
                  {naturalSize.w}×{naturalSize.h}
                </span>
                <span className="preview-size-arrow" aria-hidden="true">
                  →
                </span>
                <span
                  className={`preview-size-bucket${
                    bucket && (bucket.w > naturalSize.w || bucket.h > naturalSize.h)
                      ? ' is-upscale'
                      : ''
                  }`}
                  title={
                    bucket
                      ? `AR bucket (tier ${bucket.tier}${
                          bucket.w > naturalSize.w || bucket.h > naturalSize.h
                            ? ', upscale'
                            : ''
                        })`
                      : 'Bucket'
                  }
                >
                  {bucket ? `${bucket.w}×${bucket.h}` : '—'}
                </span>
              </>
            ) : (
              <span className="preview-size-muted">—</span>
            )}
          </div>
          <div className="preview-toolbar-right">
            <button
              type="button"
              className={`preview-toolbar-toggle${bucketPreview ? ' active' : ''}`}
              aria-pressed={bucketPreview}
              disabled={!naturalSize || !bucket}
              title="Show bucket crop region on preview"
              onClick={() => onBucketPreviewChange(!bucketPreview)}
            >
              Bucket preview
            </button>
          </div>
        </div>
        <ImagePreview
          imagePath={selectedPath}
          imageUrl={imageUrl}
          bucketPreview={bucketPreview}
          bucket={bucket}
          onNaturalSize={onNaturalSize}
        />
      </section>

      <button
        type="button"
        className={`pane-resizer${draggingPane === 'right' ? ' dragging' : ''}`}
        aria-label="Resize caption pane"
        title="Drag to resize"
        onPointerDown={onStartPaneResize('right')}
      />

      <section className="right-pane">
        <CaptionEditor
          english={english}
          translated={translated}
          targetLanguage={settings.targetLanguage}
          translating={Boolean(translating && translatingPath === selectedPath)}
          captioning={captioningPath !== null && captioningPath === selectedPath}
          canSave={Boolean(selectedPath) && dirty && !captionBusy}
          canRecaption={Boolean(selectedPath) && !captionBusy}
          onEnglishChange={onEnglishChange}
          onTranslatedChange={onTranslatedChange}
          onLanguageChange={onLanguageChange}
          onSave={onSave}
          onRecaption={onRecaption}
        />
      </section>
    </div>
  )
}
