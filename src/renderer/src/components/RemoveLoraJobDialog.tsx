import { useEffect, useRef, useState } from 'react'

interface Props {
  open: boolean
  jobName: string
  onConfirm: () => void
  onCancel: () => void
}

type FocusBtn = 'cancel' | 'remove'

export function RemoveLoraJobDialog({ open, jobName, onConfirm, onCancel }: Props) {
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
        setFocusBtn((prev) => (prev === 'cancel' ? 'remove' : 'cancel'))
        return
      }

      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault()
        e.stopPropagation()
        if (focusRef.current === 'remove') onConfirmRef.current()
        else onCancelRef.current()
      }
    }

    window.addEventListener('keydown', onKeyDown, true)
    return () => window.removeEventListener('keydown', onKeyDown, true)
  }, [open])

  if (!open) return null

  return (
    <div
      className="modal-backdrop"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onCancel()
      }}
    >
      <div
        className="modal"
        role="dialog"
        aria-labelledby="remove-lora-job-title"
        onMouseDown={(e) => e.stopPropagation()}
      >
        <h2 id="remove-lora-job-title">Remove job preset</h2>
        <p className="modal-text">
          Remove &quot;{jobName}&quot; from the job list? Training files on disk will not be deleted.
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
            className={`danger${focusBtn === 'remove' ? ' kbd-focus' : ''}`}
            onClick={onConfirm}
            onMouseEnter={() => setFocusBtn('remove')}
          >
            Remove
          </button>
        </div>
      </div>
    </div>
  )
}
