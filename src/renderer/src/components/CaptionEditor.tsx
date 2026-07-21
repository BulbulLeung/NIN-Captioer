import { LANGUAGES } from '../types'

interface Props {
  english: string
  translated: string
  targetLanguage: string
  translating: boolean
  captioning: boolean
  error: string | null
  canSave: boolean
  canRecaption: boolean
  onEnglishChange: (value: string) => void
  onTranslatedChange: (value: string) => void
  onLanguageChange: (code: string) => void
  onDismissError: () => void
  onSave: () => void
  onRecaption: () => void
}

export function CaptionEditor({
  english,
  translated,
  targetLanguage,
  translating,
  captioning,
  error,
  canSave,
  canRecaption,
  onEnglishChange,
  onTranslatedChange,
  onLanguageChange,
  onDismissError,
  onSave,
  onRecaption
}: Props) {
  return (
    <div className="caption-editor">
      <div className="caption-panel">
        <div className="caption-header">
          <label htmlFor="caption-en">English Caption (saved to .txt)</label>
          <div className="caption-header-actions">
            {(translating || captioning) && (
              <span className="status translating">
                {captioning
                  ? 'Captioning…'
                  : 'Translating (faster after model warmup)…'}
              </span>
            )}
            <button
              type="button"
              onClick={onRecaption}
              disabled={!canRecaption || captioning}
            >
              reCaption
            </button>
            <button
              type="button"
              className="primary"
              onClick={onSave}
              disabled={!canSave}
            >
              Save
            </button>
          </div>
        </div>
        <textarea
          id="caption-en"
          className="caption-textarea"
          value={english}
          onChange={(e) => onEnglishChange(e.target.value)}
          placeholder="English caption…"
          spellCheck={false}
        />
      </div>

      <div className="caption-panel">
        <div className="caption-header">
          <label htmlFor="caption-tr">Translated Caption</label>
          <select
            className="lang-select"
            value={targetLanguage}
            onChange={(e) => onLanguageChange(e.target.value)}
            aria-label="Target language"
          >
            {LANGUAGES.map((lang) => (
              <option key={lang.code} value={lang.code}>
                {lang.label}
              </option>
            ))}
          </select>
        </div>
        <textarea
          id="caption-tr"
          className="caption-textarea"
          value={translated}
          onChange={(e) => onTranslatedChange(e.target.value)}
          placeholder="Edits here are translated back to English above…"
          spellCheck={false}
        />
      </div>

      {error && (
        <div className="caption-error" role="alert">
          <span>{error}</span>
          <button type="button" onClick={onDismissError}>
            Dismiss
          </button>
        </div>
      )}
    </div>
  )
}
