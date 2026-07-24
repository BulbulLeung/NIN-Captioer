import type { ReactNode } from 'react'

interface Props {
  value: string
  onChange: (value: string) => void
  /** Optional hint under the field; default explains shared WD14 + LoRA use. */
  hint?: ReactNode
}

const DEFAULT_HINT: ReactNode = (
  <>
    Shared by Dataset Edit (WD14 tagging) and LoRA Train. Leave empty to use{' '}
    <code>python</code> from PATH.
  </>
)

export function PythonExecutableField({ value, onChange, hint = DEFAULT_HINT }: Props) {
  const browsePython = async () => {
    const file = await window.api.openFile({
      title: 'Select Python executable',
      filters: [
        { name: 'Python', extensions: ['exe'] },
        { name: 'All Files', extensions: ['*'] }
      ]
    })
    if (!file) return
    onChange(file)
  }

  return (
    <label className="field">
      <span>Python executable</span>
      <div className="model-row">
        <input
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder="e.g. C:\Python311\python.exe or python"
          spellCheck={false}
        />
        <button type="button" onClick={() => void browsePython()}>
          Browse
        </button>
      </div>
      {hint != null && <p className="field-hint">{hint}</p>}
    </label>
  )
}
