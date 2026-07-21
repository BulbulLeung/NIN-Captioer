# NIN Capto

**LoRA 訓練用圖片 Caption 編輯器** — 瀏覽資料集、編輯英文 caption、以本地 LLM（LM Studio / Ollama）雙向翻譯，並可批次 Auto Caption、分析資料集健康度。

## 畫面預覽

### 全畫面

<p align="center">
  <!-- TODO: 追加截圖 docs/screenshots/01-main-window.png -->
  <img src="docs/screenshots/01-main-window.png" alt="NIN Capto 全畫面" width="900" />
</p>

> **請追加截圖**：主視窗最大化全景（工具列 + 左側清單 + 中央預覽 + 右側雙欄 caption）。檔名：`docs/screenshots/01-main-window.png`

### Caption Analysis Page

<p align="center">
  <!-- TODO: 追加截圖 docs/screenshots/02-caption-analysis.png -->
  <img src="docs/screenshots/02-caption-analysis.png" alt="Caption Analysis" width="900" />
</p>

> **請追加截圖**：Analyze 對話框完整畫面（Health Score、breakdown、類別圓餅圖）。檔名：`docs/screenshots/02-caption-analysis.png`

---

## 功能亮點

- **資料夾瀏覽**：開啟含訓練圖的資料夾，左側清單、中央預覽、右側編輯
- **英文 Caption 編輯**：儲存為與圖片同名的 `.txt`（標準 LoRA / caption 工作流）
- **雙向翻譯**：上欄英文 ↔ 下欄目標語言（預設繁中），編輯任一邊會自動同步
- **Auto Caption / reCaption**：用本地 vision／LLM 依自訂 prompt 批次或單張產生 caption（會讀取 PNG Info）
- **Caption Analysis**：統計已標註比例、LoRA Health Score、各類別細節分布（圓餅圖）
- **可調版面**：拖曳分隔線調整清單與編輯區寬度，視窗位置會記住

---

## 需求

| 項目 | 說明 |
|------|------|
| 系統 | Windows x64（目前以 `electron-builder --win` 打包） |
| 開發 | Node.js 18+ |
| AI 後端（選用） | [LM Studio](https://lmstudio.ai/) 或 [Ollama](https://ollama.com/)，需本機可連線 |

沒有 AI 後端仍可手動編輯與儲存 caption；翻譯、Auto Caption、Analyze 需要已啟動的模型服務。

---

## 快速開始

### 從原始碼執行

```bash
npm install
npm run dev
```

### 打包 Windows Portable

```bash
npm run dist
```

產物：`release/NIN Capto-0.1.0-portable.exe`

---

## 資料夾結構約定

每個訓練圖對應一個同名 caption 檔：

```text
my-dataset/
  img_001.png
  img_001.txt          ← 英文 caption（由本軟體讀寫）
  img_002.jpg
  img_002.txt
  …
```

支援常見圖片副檔名：`.jpg` / `.jpeg` / `.png` / `.webp` / `.bmp` / `.gif`。

---

## 使用教學

### 1. 開啟資料集

1. 點工具列 **Open folder**
2. 選擇含圖片的資料夾
3. 左側出現圖片清單（有／無 `.txt` 會標示）

### 2. 編輯與儲存 Caption

1. 在清單點選一張圖，中央顯示預覽
2. 右側 **English Caption** 編輯英文文字
3. 點 **Save**，或按 `Ctrl+S`，寫入同名 `.txt`
4. 未儲存時工具列會顯示 **Unsaved**；切換圖片前可選擇儲存／放棄

### 3. 雙向翻譯

1. 先到 **Settings** 選 LM Studio 或 Ollama、Base URL、Model，並 **Test connection**
2. 右側下欄選擇目標語言（繁中／簡中／日／韓／歐語等）
3. 有英文 caption 時會自動翻成目標語言
4. 直接改下欄翻譯文，會回譯並更新上欄英文（方便用母語潤稿再寫回訓練用英文）

### 4. Settings（連線與 Caption Prompt）

在 **Settings** 可設定：

- **Translation provider**：LM Studio / Ollama
- **Base URL**（預設 LM Studio `http://localhost:1234/v1`，Ollama `http://localhost:11434`）
- **Model**：自動偵測清單，可 Refresh
- **Auto Caption prompt**：多組 Preset（名稱 + Prompt）；執行時會在 prompt 後附加圖片 PNG Info

同一個 Model 會用於翻譯、Auto Caption、reCaption、Analyze。

### 5. Auto Caption / reCaption

**Auto Caption**

- 只處理「還沒有 `.txt`」的圖片
- 按鈕會顯示待處理數量，例如 `Auto Caption (12)`
- 進行中可 **Cancel Auto Caption**

**reCaption**

- 對目前選中的圖重新產生 caption（會覆寫並儲存）

執行前請確認 Settings 中的 model 與 prompt preset 正確，且後端已載入支援影像的模型（依你的 LM Studio／Ollama 設定）。

### 6. Caption Analysis（資料集分析）

點 **Analyze** 後會：

1. 讀取所有 caption
2. 用 AI 分類各 caption 的細節標籤
3. 顯示 **Total Images**、**LoRA Health Score（/100）**、分數拆解、優點／改進建議
4. 各類別（Subject、Camera、Clothing…）細節分布圓餅圖

可用來檢查資料集多樣性與 caption 品質是否偏斜。

---

## 快捷鍵

| 快捷鍵 | 功能 |
|--------|------|
| `Ctrl+S` | 儲存目前英文 caption |
| `↑` / `←` | 上一張圖（焦點不在輸入框時） |
| `↓` / `→` | 下一張圖 |
| `Delete` | 刪除目前圖片與其 caption（需確認） |

---

## 建議工作流

1. 整理訓練圖到一個資料夾  
2. **Settings** → 連上 LM Studio／Ollama → 選好 model 與 caption preset  
3. **Auto Caption** 補齊缺漏的 `.txt`  
4. 用翻譯下欄檢查／潤稿，再存回英文  
5. **Analyze** 看 Health Score 與類別分布，必要時 **reCaption** 或手修  
6. 將資料夾接到你的 LoRA 訓練腳本／GUI  

---

## 開發指令

| 指令 | 說明 |
|------|------|
| `npm run dev` | 開發模式 |
| `npm run build` | 編譯 |
| `npm run preview` | 預覽編譯結果 |
| `npm run typecheck` | TypeScript 檢查 |
| `npm run dist` | 打包 Windows portable |

技術棧：Electron + React + TypeScript（electron-vite）。

---

## 要追加的範例圖（僅 2 張）

請放入 `docs/screenshots/`：

| 檔名 | 內容 |
|------|------|
| `01-main-window.png` | 全畫面主介面 |
| `02-caption-analysis.png` | Caption Analysis Page |

截圖建議：視窗最大化、UI 清晰、避免含個資路徑。

---

## License

未指定授權時，使用前請先確認 repository 的 License 檔案或以作者聲明為準。
