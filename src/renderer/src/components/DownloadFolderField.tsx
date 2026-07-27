interface Props {
  value: string
  onChange: (value: string) => void
  disabled?: boolean
}

export function DownloadFolderField({ value, onChange, disabled = false }: Props) {
  const browse = async () => {
    const dir = await window.api.openFolder()
    if (!dir) return
    onChange(dir)
  }

  const useDefault = async () => {
    const path = await window.api.defaultDownloadFolder()
    onChange(path)
  }

  return (
    <label className="field">
      <span>Download folder</span>
      <div className="model-row">
        <input
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder="Empty = AppData\Roaming\Captioer"
          spellCheck={false}
          disabled={disabled}
        />
        <button type="button" onClick={() => void browse()} disabled={disabled}>
          Browse
        </button>
        <button type="button" onClick={() => void useDefault()} disabled={disabled}>
          Default
        </button>
      </div>
      <p className="field-hint">
        Shared by Python install and model downloads. Uses <code>{'{folder}'}/python</code> and{' '}
        <code>{'{folder}'}/models</code>. Empty defaults to{' '}
        <code>AppData\Roaming\Captioer</code>.
      </p>
    </label>
  )
}
