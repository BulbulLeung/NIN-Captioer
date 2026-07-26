import { useEffect, useState } from 'react'
import type { LoraTrainAppSettings } from '../types'
import { DEFAULT_LORA_TRAIN_APP, normalizeLoraTrainApp } from '../types'
import { PythonExecutableField } from './PythonExecutableField'

interface Props {
  open: boolean
  settings: LoraTrainAppSettings
  onClose: () => void
  onSave: (settings: LoraTrainAppSettings) => Promise<void>
}

export function LoraTrainSettingsDialog({ open, settings, onClose, onSave }: Props) {
  const [draft, setDraft] = useState(() => normalizeLoraTrainApp(settings))
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    if (!open) return
    setDraft(normalizeLoraTrainApp(settings))
  }, [open, settings])

  if (!open) return null

  const browseModelDownloadPath = async () => {
    const dir = await window.api.openFolder()
    if (!dir) return
    setDraft((prev) => ({ ...prev, modelDownloadPath: dir }))
  }

  const useDefaultModelPath = async () => {
    const path = await window.api.defaultModelDownloadPath()
    setDraft((prev) => ({ ...prev, modelDownloadPath: path }))
  }

  const handleSave = async () => {
    setSaving(true)
    try {
      await onSave(normalizeLoraTrainApp(draft))
      onClose()
    } finally {
      setSaving(false)
    }
  }

  const resetDefaults = () => {
    setDraft({ ...DEFAULT_LORA_TRAIN_APP })
  }

  return (
    <div
      className="modal-backdrop"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose()
      }}
    >
      <div
        className="modal modal-wide"
        role="dialog"
        aria-labelledby="lora-settings-title"
        onMouseDown={(e) => e.stopPropagation()}
      >
        <h2 id="lora-settings-title">LoRA Train Settings</h2>
        <p className="field-hint">
          Captioer native Krea 2 trainer preferences. Job hyper-parameters live on the LoRA Train
          panel. Train on Raw, use LoRA on Turbo.
        </p>

        <PythonExecutableField
          value={draft.pythonPath}
          onChange={(pythonPath) => setDraft((prev) => ({ ...prev, pythonPath }))}
          installPath={draft.pythonInstallPath}
          onInstallPathChange={(pythonInstallPath) =>
            setDraft((prev) => ({ ...prev, pythonInstallPath }))
          }
          enabled={open}
          hint={
            <>
              Same setting as Dataset Edit Settings. Download installs CUDA torch 2.9.1
              (cu128), packages from <code>trainer/requirements.txt</code> and{' '}
              <code>trainer/requirements-wd14.txt</code>, and on Windows{' '}
              <code>triton-windows</code> and <code>flash-attn</code> (FA2 wheel).
            </>
          }
        />

        <label className="field">
          <span>Hugging Face token (for gated Krea models)</span>
          <input
            type="password"
            value={draft.huggingfaceToken}
            onChange={(e) =>
              setDraft((prev) => ({ ...prev, huggingfaceToken: e.target.value }))
            }
            placeholder="hf_… from huggingface.co/settings/tokens"
            autoComplete="off"
          />
          <p className="field-hint">
            Also open{' '}
            <a
              href="https://huggingface.co/krea/Krea-2-Raw"
              target="_blank"
              rel="noreferrer"
            >
              krea/Krea-2-Raw
            </a>{' '}
            (and Turbo) while logged in and click Agree — token alone is not enough.
          </p>
        </label>

        <label className="field">
          <span>Model download path</span>
          <div className="model-row">
            <input
              type="text"
              value={draft.modelDownloadPath}
              onChange={(e) =>
                setDraft((prev) => ({ ...prev, modelDownloadPath: e.target.value }))
              }
              placeholder="Empty = app userData/models"
            />
            <button type="button" onClick={() => void browseModelDownloadPath()}>
              Browse
            </button>
            <button type="button" onClick={() => void useDefaultModelPath()}>
              Default
            </button>
          </div>
          <p className="field-hint">
            Raw / Turbo snapshots are stored here. LoRA Train checks this folder on enter.
          </p>
        </label>

        <div className="modal-actions">
          <button type="button" onClick={resetDefaults}>
            Reset defaults
          </button>
          <div className="spacer" />
          <button type="button" onClick={onClose}>
            Cancel
          </button>
          <button
            type="button"
            className="primary"
            onClick={() => void handleSave()}
            disabled={saving}
          >
            {saving ? 'Saving…' : 'Save'}
          </button>
        </div>
      </div>
    </div>
  )
}
