import { useEffect, useRef, useState } from 'react'

interface Props {
  open: boolean
  folderName: string
  onConfirm: () => void
  onCancel: () => void
}

type FocusBtn = 'cancel' | 'delete'

export function MissingDatasetFolderDialog({ open, folderName, onConfirm, onCancel }: Props) {
  const [focusBtn, setFocusBtn] = useState<FocusBtn>('cancel')
  const focusRef = useRef<FocusBtn>('cancel')
  focusRef.current = focusBtn
  const onConfirmRef = useRef(onConfirm)
  const onCancelRef = useRef(onCancel)
  onConfirmRef.current = onConfirm
  onCancelRef.current = onCancel

  useEffect(() => {
    if (!open) return
    setFocusBtn('cancel')
  }, [open])

  useEffect(() => {
    if (!open) return

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault()
        e.stopPropagation()
        onCancelRef.current()
        return
      }

      if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
        e.preventDefault()
        e.stopPropagation()
        setFocusBtn((prev) => (prev === 'cancel' ? 'delete' : 'cancel'))
        return
      }

      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault()
        e.stopPropagation()
        if (focusRef.current === 'delete') onConfirmRef.current()
        else onCancelRef.current()
      }
    }

    window.addEventListener('keydown', onKeyDown, true)
    return () => window.removeEventListener('keydown', onKeyDown, true)
  }, [open])

  if (!open) return null

  return (
    <div className="modal-backdrop" onClick={onCancel}>
      <div
        className="modal"
        role="dialog"
        aria-labelledby="missing-dataset-title"
        onClick={(e) => e.stopPropagation()}
      >
        <h2 id="missing-dataset-title">Dataset folder not found</h2>
        <p className="modal-text">
          The dataset folder &quot;{folderName}&quot; does not exist. Delete this preset from the list?
        </p>
        <div className="modal-actions">
          <button
            type="button"
            className={focusBtn === 'cancel' ? 'kbd-focus' : undefined}
            onClick={onCancel}
            onMouseEnter={() => setFocusBtn('cancel')}
          >
            Cancel
          </button>
          <div className="spacer" />
          <button
            type="button"
            className={`danger${focusBtn === 'delete' ? ' kbd-focus' : ''}`}
            onClick={onConfirm}
            onMouseEnter={() => setFocusBtn('delete')}
          >
            Delete
          </button>
        </div>
      </div>
    </div>
  )
}
