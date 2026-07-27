import { useEffect, useMemo, useRef, type CSSProperties } from 'react'
import {
  Virtuoso,
  VirtuosoGrid,
  type VirtuosoGridHandle,
  type VirtuosoHandle
} from 'react-virtuoso'
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

function ImageListItem({
  img,
  selected,
  dirty,
  busy,
  isThumbnails,
  onSelect
}: {
  img: ImageItem
  selected: boolean
  dirty: boolean
  busy: boolean
  isThumbnails: boolean
  onSelect: (path: string) => void
}) {
  return (
    <button
      type="button"
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
  )
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
  const listRef = useRef<VirtuosoHandle | null>(null)
  const gridRef = useRef<VirtuosoGridHandle | null>(null)
  const isThumbnails = viewMode === 'thumbnails'

  const selectedIndex = useMemo(() => {
    if (!selectedPath) return -1
    return images.findIndex((img) => img.path === selectedPath)
  }, [images, selectedPath])

  useEffect(() => {
    if (selectedIndex < 0) return
    if (isThumbnails) {
      gridRef.current?.scrollToIndex({
        index: selectedIndex,
        align: 'center',
        behavior: 'auto'
      })
    } else {
      listRef.current?.scrollIntoView({
        index: selectedIndex,
        align: 'center',
        behavior: 'auto'
      })
    }
  }, [selectedIndex, isThumbnails, viewMode])

  if (images.length === 0) {
    return <div className="image-list empty">No images. Open a folder to begin.</div>
  }

  const renderItem = (index: number) => {
    const img = images[index]
    return (
      <ImageListItem
        img={img}
        selected={img.path === selectedPath}
        dirty={dirtyPaths.has(img.path)}
        busy={busyPaths.has(img.path)}
        isThumbnails={isThumbnails}
        onSelect={onSelect}
      />
    )
  }

  if (isThumbnails) {
    return (
      <VirtuosoGrid
        ref={gridRef}
        style={
          {
            height: '100%',
            '--thumb-w': `${thumbnailWidth}px`
          } as CSSProperties
        }
        totalCount={images.length}
        overscan={200}
        listClassName="image-list thumbnails"
        itemClassName="image-list-grid-item"
        itemContent={renderItem}
      />
    )
  }

  return (
    <Virtuoso
      ref={listRef}
      style={{ height: '100%' }}
      className="image-list"
      totalCount={images.length}
      overscan={120}
      itemContent={renderItem}
    />
  )
}
