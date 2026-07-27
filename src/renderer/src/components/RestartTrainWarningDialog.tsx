import { useEffect, useRef, useState } from 'react'

interface Props {
  open: boolean
  jobName: string
  checkpointStep: number | null
  onConfirm: () => void
  onCancel: () => void
}

type FocusBtn = 'cancel' | 'restart'

export function RestartTrainWarningDialog({
  open,
  jobName,
  checkpointStep,
  onConfirm,
  onCancel
}: Props) {
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
        setFocusBtn((prev) => (prev === 'cancel' ? 'restart' : 'cancel'))
        return
      }

      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault()
        e.stopPropagation()
        if (focusRef.current === 'restart') onConfirmRef.current()
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
        aria-labelledby="restart-train-warning-title"
        onMouseDown={(e) => e.stopPropagation()}
      >
        <h2 id="restart-train-warning-title">Overwrite existing checkpoints?</h2>
        <p className="modal-text">
          Restarting &quot;{jobName}&quot; will start training again from step 0 and may overwrite
          checkpoints that already exist in the output folder.
        </p>
        {checkpointStep !== null ? (
          <p className="modal-text">Latest detected checkpoint: step {checkpointStep}.</p>
        ) : null}
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
            className={`danger${focusBtn === 'restart' ? ' kbd-focus' : ''}`}
            onClick={onConfirm}
            onMouseEnter={() => setFocusBtn('restart')}
          >
            Restart
          </button>
        </div>
      </div>
    </div>
  )
}
