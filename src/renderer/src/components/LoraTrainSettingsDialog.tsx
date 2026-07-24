import { useEffect, useState } from 'react'
import type { LoraTrainAppSettings } from '../types'
import { DEFAULT_LORA_TRAIN_APP, normalizeLoraTrainApp } from '../types'
import { GpuDeviceSelect } from './GpuDeviceSelect'
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
  const [checking, setChecking] = useState(false)
  const [checkMsg, setCheckMsg] = useState<string | null>(null)
  const [checkErr, setCheckErr] = useState<string | null>(null)

  useEffect(() => {
    if (!open) return
    setDraft(normalizeLoraTrainApp(settings))
    setCheckMsg(null)
    setCheckErr(null)
  }, [open, settings])

  if (!open) return null

  const browseExportDir = async () => {
    const dir = await window.api.openFolder()
    if (!dir) return
    setDraft((prev) => ({ ...prev, exportDir: dir }))
  }

  const browseModelDownloadPath = async () => {
    const dir = await window.api.openFolder()
    if (!dir) return
    setDraft((prev) => ({ ...prev, modelDownloadPath: dir }))
  }

  const useDefaultModelPath = async () => {
    const path = await window.api.defaultModelDownloadPath()
    setDraft((prev) => ({ ...prev, modelDownloadPath: path }))
  }

  const verifyEnv = async () => {
    setChecking(true)
    setCheckMsg(null)
    setCheckErr(null)
    try {
      const result = await window.api.checkTrainEnv(draft.pythonPath.trim() || undefined)
      if (result.ok) {
        setCheckMsg(result.message || 'Environment OK')
      } else {
        setCheckErr(result.message || 'Environment check failed')
      }
    } catch (err) {
      setCheckErr(err instanceof Error ? err.message : String(err))
    } finally {
      setChecking(false)
    }
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
          hint={
            <>
              Same setting as Dataset Edit Settings. CUDA-enabled Python with packages from{' '}
              <code>trainer/requirements.txt</code> (training) and{' '}
              <code>trainer/requirements-wd14.txt</code> (WD14 tagging).
            </>
          }
        />

        <div className="field">
          <div className="model-row">
            <button type="button" onClick={() => void verifyEnv()} disabled={checking}>
              {checking ? 'Checking…' : 'Verify environment'}
            </button>
          </div>
          {checkMsg && <p className="test-ok">{checkMsg}</p>}
          {checkErr && <p className="test-err">{checkErr}</p>}
        </div>

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

        <label className="field">
          <span>Default training folder</span>
          <input
            type="text"
            value={draft.defaultTrainingFolder}
            onChange={(e) =>
              setDraft((prev) => ({ ...prev, defaultTrainingFolder: e.target.value }))
            }
          />
        </label>

        <label className="field">
          <span>Default device</span>
          <GpuDeviceSelect
            enabled={open}
            value={draft.defaultDevice}
            onChange={(defaultDevice) => setDraft((prev) => ({ ...prev, defaultDevice }))}
          />
        </label>

        <label className="field">
          <span>Config export directory</span>
          <div className="model-row">
            <input
              type="text"
              value={draft.exportDir}
              onChange={(e) => setDraft((prev) => ({ ...prev, exportDir: e.target.value }))}
              placeholder="Optional default folder for Export config"
            />
            <button type="button" onClick={() => void browseExportDir()}>
              Browse
            </button>
          </div>
        </label>

        <label className="field">
          <span>Config export file name</span>
          <input
            type="text"
            value={draft.exportFileName}
            onChange={(e) =>
              setDraft((prev) => ({ ...prev, exportFileName: e.target.value }))
            }
          />
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
