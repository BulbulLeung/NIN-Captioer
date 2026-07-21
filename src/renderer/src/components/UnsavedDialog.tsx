interface Props {
  open: boolean
  onSave: () => void
  onDiscard: () => void
  onCancel: () => void
}

export function UnsavedDialog({ open, onSave, onDiscard, onCancel }: Props) {
  if (!open) return null

  return (
    <div className="modal-backdrop" onClick={onCancel}>
      <div
        className="modal"
        role="dialog"
        aria-labelledby="unsaved-title"
        onClick={(e) => e.stopPropagation()}
      >
        <h2 id="unsaved-title">Unsaved changes</h2>
        <p className="modal-text">
          The English caption has not been written to the .txt file. What would you like to do?
        </p>
        <div className="modal-actions">
          <button type="button" onClick={onCancel}>
            Cancel
          </button>
          <div className="spacer" />
          <button type="button" onClick={onDiscard}>
            Discard
          </button>
          <button type="button" className="primary" onClick={onSave}>
            Save
          </button>
        </div>
      </div>
    </div>
  )
}
