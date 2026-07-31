# Captioer

**Caption → Train → Test** for LoRA workflows (v0.2.0). Manage datasets and English captions, train native **Krea 2** LoRAs, then verify results in **ComfyUI** — all in one Windows app.

Three views: **Dataset Edit** · **Lora Train** · **Lora Test**.

## Screenshots

<p align="center">
  <img src="docs/screenshots/01-main-window.png" alt="Captioer full window" width="900" />
</p>

---

## Features

### Dataset Edit

- **Dataset folders**: add multiple training folders, switch from the toolbar dropdown, or remove a folder from the list (disk files are kept unless you delete images)
- **Image browser**: list or thumbnail view on the left, preview in the center, editor on the right; thumbnail size is adjustable
- **English caption editing**: save as a sidecar `.txt` with the same name as the image (standard LoRA / caption workflow)
- **Bidirectional translation**: English (top) ↔ target language (bottom, Traditional Chinese by default); edits on either side stay in sync via LM Studio / Ollama
- **Auto Caption / reCaption**: Natural Language (Flux/Krea2 vision/LLM) or Danbooru Tags via WD14 ONNX (SD/XL); switch format from the Dataset Edit toolbar
- **Caption Analysis**: caption coverage, LoRA Health Score, and per-category detail distributions
- **AR bucket preview**: scan aspect-ratio buckets aligned with training bucket settings
- **Resizable layout**: drag splitters to resize panes; window position is remembered

### Lora Train

- Native **Krea 2** LoRA trainer (train on Raw, apply on Turbo)
- Multiple training jobs, Start/Stop, progress log, loss chart, sample previews, and config export
- **Loss spike** markers and **OOM detection** (memory pressure / CUDA OOM surfaced in UI and log)
- **Resource monitor**: CPU / RAM / GPU usage; optionally kill processes holding VRAM
- Low VRAM options: layer offload, quantization, gradient checkpointing
- Hugging Face model check / **Download** / **Update**; Python path probe and one-click install of trainer deps
- LoRA weight export compatible with ComfyUI key format

### Lora Test

- Install / start / stop **ComfyUI** from the app
- Generate with trained checkpoints, multi-prompt runs, and a result gallery

---

## Requirements

| Item | Notes |
|------|--------|
| OS | Windows x64 (packaged with `electron-builder --win`) |
| Development | Node.js 18+ |
| AI backend (optional) | [LM Studio](https://lmstudio.ai/) or [Ollama](https://ollama.com/), reachable on localhost |
| WD14 tagging (optional) | Python + packages in `trainer/requirements-wd14.txt` (`onnxruntime`, etc.); model downloads on first use |
| Krea 2 training (optional) | CUDA Python + packages in `trainer/requirements.txt`; recent `diffusers` with `Krea2Pipeline` |
| ComfyUI (optional) | Install from **Lora Test** in-app, or point at an existing ComfyUI install |

You can still edit and save captions without an AI backend. Natural Auto Caption, translation, and Analyze need a running vision/LLM service; WD14 Auto Caption needs the ONNX deps above. Training and Lora Test need their respective Python / ComfyUI setups.

---

## Quick start

### Run from source

```bash
npm install
npm run dev
```

### Build Windows portable

```bash
npm run dist
```

Output: portable exe under `release/` (artifact name follows `productName` in `package.json`, e.g. `Captioer-0.2.0-portable.exe`).

---

## Dataset layout

Each image pairs with a same-named caption file:

```text
my-dataset/
  img_001.png
  img_001.txt          ← English caption (read/written by this app)
  img_002.jpg
  img_002.txt
  …
```

Supported image extensions: `.jpg` / `.jpeg` / `.png` / `.webp` / `.bmp` / `.gif`.

---

## How to use

### 1. Open a dataset

1. Click **Add Dataset Folder** (`+`) in the toolbar
2. Select a folder that contains images
3. The left list shows images (with / without `.txt` is indicated)
4. Switch folders from the toolbar dropdown; **Remove dataset folder** drops it from the app list only (does not delete files on disk)

### 2. Browse images

- Toggle **List** / **Thumbnail** view in the sidebar toolbar
- In thumbnail mode, drag the slider to resize thumbnails
- Use arrow keys to move between images when focus is not in an input

### 3. Edit and save captions

1. Select an image in the list; the center pane shows the preview
2. Edit text in **English Caption** on the right
3. Click **Save**, or press `Ctrl+S`, to write the sidecar `.txt`
4. Unsaved work shows **Unsaved** in the toolbar; you can save or discard when switching images

### 4. Bidirectional translation

1. Open **Settings**, choose LM Studio or Ollama, Base URL, and Model, then **Test connection**
2. Pick the target language in the lower pane (Traditional/Simplified Chinese, Japanese, Korean, European languages, etc.)
3. Existing English captions are translated automatically
4. Edit the translation pane to back-translate and update the English caption (useful for drafting in your native language)

### 5. Settings (connection & caption prompt)

In **Settings** you can configure:

- **Translation provider**: LM Studio / Ollama
- **Base URL** (defaults: LM Studio `http://localhost:1234/v1`, Ollama `http://localhost:11434`)
- **Model**: auto-detected list, with Refresh
- **Auto analysis**: on by default (background); turn off to analyze only while the Analyze dialog is open
- **Auto Caption prompt**: multiple presets (name + prompt); PNG Info is appended at runtime

The same model is used for translation, Auto Caption, reCaption, and Analyze.

**Interactive AI priority:** Translation, Auto Caption, and reCaption take the model first. Caption Analysis pauses while those run and resumes about 0.8s after they finish (cache and queue are kept). Optional: set Ollama `OLLAMA_NUM_PARALLEL=2` only if you want hard server-side concurrency; it is not required for this app’s yield behavior, and parallel inference can slow individual interactive requests.

### 6. Auto Caption / reCaption

**Auto Caption**

- Only processes images that do not yet have a `.txt`
- The button shows the pending count, e.g. `Auto Caption (12)`
- While running, use **Cancel Auto Caption**

**reCaption**

- Regenerates the caption for the currently selected image (overwrites and saves)

Before running, confirm the model and prompt preset in Settings, and that your backend has loaded a vision-capable model (depending on your LM Studio / Ollama setup).

### 7. Caption Analysis

With **Auto analysis** on (default), analysis runs in the background when interactive AI is idle. With it off, analysis starts only while the Analyze dialog is open. Click **Analyze** to view:

1. Caption coverage and AI-classified detail tags
2. **Total Images**, **LoRA Health Score (/100)**, score breakdown, strengths / improvements
3. Per-category detail distributions (Subject, Camera, Clothing, …)

Use this to check dataset diversity and whether captions are skewed.

---

## Keyboard shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+S` | Save the current English caption |
| `↑` / `←` | Previous image (when focus is not in an input) |
| `↓` / `→` | Next image |
| `Delete` | Delete the current image and its caption (confirmation required) |

---

## Suggested workflow

1. **Add Dataset Folder** with your training images
2. **Settings** → connect LM Studio / Ollama → pick model and caption preset
3. **Auto Caption** to fill missing `.txt` files
4. Review / refine via the translation pane, then save English
5. **Analyze** for Health Score and category balance; **reCaption** or edit by hand as needed
6. Switch to **Lora Train** → set dataset folder, Train base = Raw, Sample = Turbo → **Settings** (Python path) → **Start Train**
7. Switch to **Lora Test** → start ComfyUI → pick a checkpoint → generate and review the gallery

---

## Krea 2 LoRA training (native)

Captioer ships a native trainer under `trainer/` (also packaged into `resources/trainer`).

**Recommended workflow (official):** train LoRA on **Krea-2-Raw**, apply on **Krea-2-Turbo**.

1. Install CUDA PyTorch, then:
   ```bash
   pip install -r trainer/requirements.txt
   ```
   If `Krea2Pipeline` is missing, install diffusers from git:
   ```bash
   pip install "git+https://github.com/huggingface/diffusers.git"
   ```
2. In the app: **Lora Train** → **Settings** → set **Python executable** (auto-probes packages; use **Download** if missing)
3. Optionally set **Model download path** (default: app `userData/models`). On entering LoRA Train, Captioer checks Raw / Turbo and prompts **Download** or **Update**.
4. Set **Train base** to `krea/Krea-2-Raw` (or a local path), **Sample / apply on** to `krea/Krea-2-Turbo`
5. Point **folder_path** at your captioned dataset, then **Start Train**
6. Weights write under `{training_folder}/{name}/`

VRAM: Krea 2 is large; use a high-VRAM NVIDIA GPU, `bf16`, and gradient checkpointing (defaults). Enable Low VRAM / layer offload when needed.

**Runtime signals:** CUDA OOM and severe memory pressure stop the run and show in the log/UI. Unusual **loss spikes** are marked on the loss chart so you can stop early or adjust hyperparameters.

---

## Lora Test (ComfyUI)

Use **Lora Test** to verify checkpoints without leaving the app.

1. Open **Lora Test** → install ComfyUI if needed (or use an existing install path)
2. **Start** ComfyUI and wait until the status shows ready
3. Select a trained LoRA / DiT checkpoint from your training output
4. Enter one or more prompts and run generation
5. Review results in the gallery; keep or filter outputs as needed

Stop ComfyUI from the same view when you are done to free VRAM.

---

## Development commands

| Command | Description |
|---------|-------------|
| `npm run dev` | Development mode |
| `npm run build` | Compile |
| `npm run preview` | Preview the production build |
| `npm run typecheck` | TypeScript check |
| `npm run dist` | Package Windows portable |

Stack: Electron + React + TypeScript (electron-vite).

---

## License

If no license file is present, check the repository license or the author’s statement before use.
