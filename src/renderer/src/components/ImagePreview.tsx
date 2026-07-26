import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { coverCropRect } from '../utils/arBuckets'

export interface BucketSize {
  w: number
  h: number
  allowUpscale: boolean
}

interface Props {
  imagePath: string | null
  imageUrl: string | null
  bucketPreview: boolean
  bucket: BucketSize | null
  onNaturalSize: (size: { w: number; h: number } | null) => void
}

interface DisplayRect {
  left: number
  top: number
  width: number
  height: number
}

/** Contain-fit: may shrink or enlarge so the image fills the pane. */
function computeContainRect(
  containerW: number,
  containerH: number,
  naturalW: number,
  naturalH: number
): DisplayRect | null {
  if (containerW <= 0 || containerH <= 0 || naturalW <= 0 || naturalH <= 0) return null
  const scale = Math.min(containerW / naturalW, containerH / naturalH)
  const width = naturalW * scale
  const height = naturalH * scale
  return {
    left: (containerW - width) / 2,
    top: (containerH - height) / 2,
    width,
    height
  }
}

export function ImagePreview({
  imagePath,
  imageUrl,
  bucketPreview,
  bucket,
  onNaturalSize
}: Props) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const imgRef = useRef<HTMLImageElement>(null)
  const [natural, setNatural] = useState<{ w: number; h: number } | null>(null)
  const [display, setDisplay] = useState<DisplayRect | null>(null)

  const measure = useCallback(() => {
    const wrap = wrapRef.current
    const img = imgRef.current
    if (!wrap || !img || !img.naturalWidth || !img.naturalHeight) {
      setDisplay(null)
      return
    }
    setDisplay(
      computeContainRect(
        wrap.clientWidth,
        wrap.clientHeight,
        img.naturalWidth,
        img.naturalHeight
      )
    )
  }, [])

  useLayoutEffect(() => {
    measure()
  }, [measure, imageUrl, natural, bucketPreview])

  useEffect(() => {
    const wrap = wrapRef.current
    if (!wrap || typeof ResizeObserver === 'undefined') {
      window.addEventListener('resize', measure)
      return () => window.removeEventListener('resize', measure)
    }
    const ro = new ResizeObserver(() => measure())
    ro.observe(wrap)
    return () => ro.disconnect()
  }, [measure, imageUrl])

  useEffect(() => {
    setNatural(null)
    onNaturalSize(null)
  }, [imagePath, imageUrl, onNaturalSize])

  useEffect(() => {
    const img = imgRef.current
    if (!img || !imageUrl) return
    if (img.complete && img.naturalWidth > 0) {
      const size = { w: img.naturalWidth, h: img.naturalHeight }
      setNatural(size)
      onNaturalSize(size)
      measure()
    }
  }, [imageUrl, imagePath, measure, onNaturalSize])

  if (!imagePath || !imageUrl) {
    return (
      <div className="image-preview empty">
        <p>Select an image from the list to preview</p>
      </div>
    )
  }

  const crop =
    bucketPreview && bucket && natural
      ? coverCropRect(natural.w, natural.h, bucket.w, bucket.h, !bucket.allowUpscale)
      : null

  const overlayStyle =
    crop && display && natural
      ? {
          left: display.left + (crop.left / natural.w) * display.width,
          top: display.top + (crop.top / natural.h) * display.height,
          width: (crop.width / natural.w) * display.width,
          height: (crop.height / natural.h) * display.height
        }
      : null

  const imgStyle = display
    ? {
        position: 'absolute' as const,
        left: display.left,
        top: display.top,
        width: display.width,
        height: display.height,
        maxWidth: 'none',
        maxHeight: 'none'
      }
    : undefined

  return (
    <div className="image-preview" ref={wrapRef}>
      <img
        ref={imgRef}
        src={imageUrl}
        alt=""
        draggable={false}
        style={imgStyle}
        onLoad={(e) => {
          const el = e.currentTarget
          const size = { w: el.naturalWidth, h: el.naturalHeight }
          setNatural(size)
          onNaturalSize(size)
          measure()
        }}
      />
      {overlayStyle ? (
        <div className="bucket-preview-overlay" style={overlayStyle} aria-hidden="true">
          <span className="bucket-preview-grid-h" />
          <span className="bucket-preview-grid-h mid" />
          <span className="bucket-preview-grid-v" />
          <span className="bucket-preview-grid-v mid" />
        </div>
      ) : null}
    </div>
  )
}
