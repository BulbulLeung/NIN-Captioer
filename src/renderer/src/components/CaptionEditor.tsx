import { useEffect, useRef } from 'react'
import { LANGUAGES } from '../types'

interface Props {
  english: string
  translated: string
  targetLanguage: string
  translating: boolean
  captioning: boolean
  canSave: boolean
  canRecaption: boolean
  onEnglishChange: (value: string) => void
  onTranslatedChange: (value: string) => void
  onLanguageChange: (code: string) => void
  onSave: () => void
  onRecaption: () => void
}

export function CaptionEditor({
  english,
  translated,
  targetLanguage,
  translating,
  captioning,
  canSave,
  canRecaption,
  onEnglishChange,
  onTranslatedChange,
  onLanguageChange,
  onSave,
  onRecaption
}: Props) {
  const englishRef = useRef<HTMLTextAreaElement>(null)
  const translatedRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    if (captioning) englishRef.current?.blur()
  }, [captioning])

  useEffect(() => {
    if (translating) translatedRef.current?.blur()
  }, [translating])

  return (
    <div className="caption-editor">
      <div className="caption-panel">
        <div className="caption-header">
          <label htmlFor="caption-en">English Caption (saved to .txt)</label>
          <div className="caption-header-actions">
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
        <div className="caption-field" aria-busy={captioning}>
          <textarea
            ref={englishRef}
            id="caption-en"
            className="caption-textarea"
            value={english}
            onChange={(e) => onEnglishChange(e.target.value)}
            placeholder="English caption…"
            spellCheck={false}
            disabled={captioning}
          />
          {captioning && (
            <div className="caption-field-overlay" aria-hidden="true">
              <div className="caption-spinner" />
            </div>
          )}
        </div>
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
        <div className="caption-field" aria-busy={translating}>
          <textarea
            ref={translatedRef}
            id="caption-tr"
            className="caption-textarea"
            value={translated}
            onChange={(e) => onTranslatedChange(e.target.value)}
            placeholder="Edits here are translated back to English above…"
            spellCheck={false}
            disabled={translating}
          />
          {translating && (
            <div className="caption-field-overlay" aria-hidden="true">
              <div className="caption-spinner" />
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
