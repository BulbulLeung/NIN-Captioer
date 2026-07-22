import { useEffect, useRef, type CSSProperties } from 'react'
import type { ImageItem, ListViewMode } from '../types'

interface Props {
  images: ImageItem[]
  selectedPath: string | null
  dirtyPaths: Set<string>
  busyPaths: Set<string>
  viewMode: ListViewMode
  thumbnailWidth: number
  onSelect: (path: string) => void
}

export function ImageList({
  images,
  selectedPath,
  dirtyPaths,
  busyPaths,
  viewMode,
  thumbnailWidth,
  onSelect
}: Props) {
  const selectedRef = useRef<HTMLButtonElement | null>(null)

  useEffect(() => {
    selectedRef.current?.scrollIntoView({ block: 'nearest', inline: 'nearest' })
  }, [selectedPath, viewMode])

  if (images.length === 0) {
    return <div className="image-list empty">No images. Open a folder to begin.</div>
  }

  const isThumbnails = viewMode === 'thumbnails'

  return (
    <ul
      className={`image-list${isThumbnails ? ' thumbnails' : ''}`}
      style={
        isThumbnails
          ? ({ '--thumb-w': `${thumbnailWidth}px` } as CSSProperties)
          : undefined
      }
    >
      {images.map((img) => {
        const selected = img.path === selectedPath
        const dirty = dirtyPaths.has(img.path)
        const busy = busyPaths.has(img.path)
        return (
          <li key={img.path}>
            <button
              type="button"
              ref={selected ? selectedRef : undefined}
              className={`image-list-item${selected ? ' selected' : ''}${isThumbnails ? ' thumb' : ''}${busy ? ' is-busy' : ''}`}
              tabIndex={-1}
              aria-busy={busy}
              onMouseDown={(e) => e.preventDefault()}
              onClick={() => onSelect(img.path)}
            >
              {isThumbnails && (
                <span className="image-list-thumb">
                  <img src={window.api.toLocalUrl(img.path)} alt="" loading="lazy" />
                  {busy && (
                    <span className="image-list-thumb-overlay" aria-hidden="true">
                      <span className="caption-spinner image-list-thumb-spinner" />
                    </span>
                  )}
                </span>
              )}
              <span className="image-list-meta">
                <span className="image-list-name" title={img.name}>
                  {dirty ? '● ' : ''}
                  {img.name}
                </span>
                {!img.hasCaption && !dirty && (
                  <span className="badge missing" title="No caption file">
                    no txt
                  </span>
                )}
                {dirty && <span className="badge dirty">unsaved</span>}
              </span>
            </button>
          </li>
        )
      })}
    </ul>
  )
}
