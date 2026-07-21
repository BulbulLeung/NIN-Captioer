import { useEffect, useRef } from 'react'
import type { ImageItem } from '../types'

interface Props {
  images: ImageItem[]
  selectedPath: string | null
  dirtyPaths: Set<string>
  onSelect: (path: string) => void
}

export function ImageList({ images, selectedPath, dirtyPaths, onSelect }: Props) {
  const selectedRef = useRef<HTMLButtonElement | null>(null)

  useEffect(() => {
    selectedRef.current?.scrollIntoView({ block: 'nearest', inline: 'nearest' })
  }, [selectedPath])

  if (images.length === 0) {
    return <div className="image-list empty">No images. Open a folder to begin.</div>
  }

  return (
    <ul className="image-list">
      {images.map((img) => {
        const selected = img.path === selectedPath
        const dirty = dirtyPaths.has(img.path)
        return (
          <li key={img.path}>
            <button
              type="button"
              ref={selected ? selectedRef : undefined}
              className={`image-list-item${selected ? ' selected' : ''}`}
              tabIndex={-1}
              onMouseDown={(e) => e.preventDefault()}
              onClick={() => onSelect(img.path)}
            >
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
            </button>
          </li>
        )
      })}
    </ul>
  )
}
