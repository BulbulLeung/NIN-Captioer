"use strict";
const electron = require("electron");
const path = require("path");
const fs = require("fs");
const promises = require("fs/promises");
const IMAGE_EXTS = /* @__PURE__ */ new Set([".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"]);
const DEFAULT_WINDOW = {
  width: 1280,
  height: 840
};
const DEFAULT_SETTINGS = {
  provider: "lmstudio",
  lmStudioBaseUrl: "http://localhost:1234/v1",
  ollamaBaseUrl: "http://localhost:11434",
  model: "",
  targetLanguage: "zh-TW",
  lastFolder: null,
  datasetFolders: [],
  captionPresets: [],
  activeCaptionPresetId: "",
  sidebarWidth: 260,
  rightPaneWidth: 380,
  autoAnalysis: true,
  windowWidth: DEFAULT_WINDOW.width,
  windowHeight: DEFAULT_WINDOW.height,
  windowX: null,
  windowY: null,
  windowMaximized: false
};
function settingsPath() {
  return path.join(electron.app.getPath("userData"), "settings.json");
}
async function loadSettings() {
  try {
    const raw = await promises.readFile(settingsPath(), "utf-8");
    return { ...DEFAULT_SETTINGS, ...JSON.parse(raw) };
  } catch {
    return { ...DEFAULT_SETTINGS };
  }
}
async function saveSettings(settings) {
  await promises.writeFile(settingsPath(), JSON.stringify(settings, null, 2), "utf-8");
}
function getWindowState(settings) {
  return {
    width: settings.windowWidth || DEFAULT_WINDOW.width,
    height: settings.windowHeight || DEFAULT_WINDOW.height,
    x: settings.windowX ?? null,
    y: settings.windowY ?? null,
    isMaximized: Boolean(settings.windowMaximized)
  };
}
function isVisibleOnAnyDisplay(bounds) {
  const displays = electron.screen.getAllDisplays();
  return displays.some((d) => {
    const a = d.workArea;
    const overlapX = Math.max(0, Math.min(bounds.x + bounds.width, a.x + a.width) - Math.max(bounds.x, a.x));
    const overlapY = Math.max(0, Math.min(bounds.y + bounds.height, a.y + a.height) - Math.max(bounds.y, a.y));
    return overlapX >= 80 && overlapY >= 80;
  });
}
async function persistWindowState(win) {
  const isMaximized = win.isMaximized();
  const bounds = isMaximized ? win.getNormalBounds() : win.getBounds();
  const current = await loadSettings();
  await saveSettings({
    ...current,
    windowWidth: bounds.width,
    windowHeight: bounds.height,
    windowX: bounds.x,
    windowY: bounds.y,
    windowMaximized: isMaximized
  });
}
function captionPathForImage(imagePath) {
  const dir = path.dirname(imagePath);
  const stem = path.basename(imagePath, path.extname(imagePath));
  return path.join(dir, `${stem}.txt`);
}
async function fileExists(path2) {
  try {
    await promises.access(path2, promises.constants.F_OK);
    return true;
  } catch {
    return false;
  }
}
function mimeForExt(ext) {
  switch (ext.toLowerCase()) {
    case ".jpg":
    case ".jpeg":
      return "image/jpeg";
    case ".png":
      return "image/png";
    case ".webp":
      return "image/webp";
    case ".gif":
      return "image/gif";
    case ".bmp":
      return "image/bmp";
    default:
      return "application/octet-stream";
  }
}
function extractPngTextChunks(buf) {
  const signature = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
  if (buf.length < 8 || !buf.subarray(0, 8).equals(signature)) return {};
  const texts = {};
  let offset = 8;
  while (offset + 8 <= buf.length) {
    const length = buf.readUInt32BE(offset);
    const type = buf.toString("ascii", offset + 4, offset + 8);
    const dataStart = offset + 8;
    const dataEnd = dataStart + length;
    if (dataEnd + 4 > buf.length) break;
    if (type === "tEXt") {
      const data = buf.subarray(dataStart, dataEnd);
      const nullIdx = data.indexOf(0);
      if (nullIdx > 0) {
        const key = data.toString("latin1", 0, nullIdx);
        const value = data.toString("latin1", nullIdx + 1);
        texts[key] = value;
      }
    } else if (type === "iTXt") {
      const data = buf.subarray(dataStart, dataEnd);
      let p = 0;
      const nextNull = () => {
        const i = data.indexOf(0, p);
        if (i < 0) return null;
        const s = data.toString("utf8", p, i);
        p = i + 1;
        return s;
      };
      const key = nextNull();
      if (key) {
        const compressionFlag = data[p];
        p += 1;
        p += 1;
        nextNull();
        nextNull();
        if (compressionFlag === 0 && p <= data.length) {
          texts[key] = data.toString("utf8", p);
        }
      }
    }
    if (type === "IEND") break;
    offset = dataEnd + 4;
  }
  return texts;
}
function positivePromptFromParameters(params) {
  const match = params.match(/\nNegative prompt:/i);
  if (match && match.index !== void 0) {
    return params.slice(0, match.index).trim();
  }
  return params.trim();
}
function extractPositivePrompt(texts) {
  if (texts.prompt?.trim()) return texts.prompt.trim();
  if (texts.parameters?.trim()) return positivePromptFromParameters(texts.parameters);
  if (texts.Description?.trim()) return texts.Description.trim();
  return "";
}
let mainWindow = null;
let saveWindowTimer = null;
function scheduleSaveWindowState(win) {
  if (saveWindowTimer) clearTimeout(saveWindowTimer);
  saveWindowTimer = setTimeout(() => {
    saveWindowTimer = null;
    void persistWindowState(win);
  }, 400);
}
async function createWindow() {
  const settings = await loadSettings();
  const saved = getWindowState(settings);
  const options = {
    width: Math.max(900, saved.width),
    height: Math.max(600, saved.height),
    minWidth: 900,
    minHeight: 600,
    title: `${electron.app.getName()} Ver${electron.app.getVersion()}`,
    show: false,
    autoHideMenuBar: true,
    // Dev: build/icon.ico. Packaged Windows builds use the exe icon from electron-builder.
    ...!electron.app.isPackaged ? { icon: path.join(__dirname, "../../build/icon.ico") } : {},
    webPreferences: {
      preload: path.join(__dirname, "../preload/index.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false
    }
  };
  if (saved.x !== null && saved.y !== null && isVisibleOnAnyDisplay({
    x: saved.x,
    y: saved.y,
    width: options.width,
    height: options.height
  })) {
    options.x = saved.x;
    options.y = saved.y;
  }
  mainWindow = new electron.BrowserWindow(options);
  mainWindow.setMenuBarVisibility(false);
  electron.Menu.setApplicationMenu(null);
  mainWindow.on("page-title-updated", (event) => {
    event.preventDefault();
  });
  mainWindow.once("ready-to-show", () => {
    if (!mainWindow) return;
    if (saved.isMaximized) mainWindow.maximize();
    mainWindow.show();
  });
  mainWindow.on("resize", () => {
    if (mainWindow && !mainWindow.isMinimized()) scheduleSaveWindowState(mainWindow);
  });
  mainWindow.on("move", () => {
    if (mainWindow && !mainWindow.isMinimized()) scheduleSaveWindowState(mainWindow);
  });
  mainWindow.on("maximize", () => {
    if (mainWindow) scheduleSaveWindowState(mainWindow);
  });
  mainWindow.on("unmaximize", () => {
    if (mainWindow) scheduleSaveWindowState(mainWindow);
  });
  mainWindow.on("close", () => {
    if (saveWindowTimer) {
      clearTimeout(saveWindowTimer);
      saveWindowTimer = null;
    }
    if (!mainWindow) return;
    const win = mainWindow;
    const isMaximized = win.isMaximized();
    const bounds = isMaximized ? win.getNormalBounds() : win.getBounds();
    try {
      let current = { ...DEFAULT_SETTINGS };
      try {
        current = { ...DEFAULT_SETTINGS, ...JSON.parse(fs.readFileSync(settingsPath(), "utf-8")) };
      } catch {
      }
      fs.writeFileSync(
        settingsPath(),
        JSON.stringify(
          {
            ...current,
            windowWidth: bounds.width,
            windowHeight: bounds.height,
            windowX: bounds.x,
            windowY: bounds.y,
            windowMaximized: isMaximized
          },
          null,
          2
        ),
        "utf-8"
      );
    } catch {
    }
  });
  if (process.env.ELECTRON_RENDERER_URL) {
    mainWindow.loadURL(process.env.ELECTRON_RENDERER_URL);
  } else {
    mainWindow.loadFile(path.join(__dirname, "../renderer/index.html"));
  }
}
electron.protocol.registerSchemesAsPrivileged([
  {
    scheme: "local-file",
    privileges: {
      standard: true,
      secure: true,
      supportFetchAPI: true,
      bypassCSP: true,
      stream: true
    }
  }
]);
electron.app.whenReady().then(async () => {
  electron.protocol.handle("local-file", async (request) => {
    const parsed = new URL(request.url);
    let filePath = decodeURIComponent(parsed.pathname);
    if (filePath.startsWith("/")) filePath = filePath.slice(1);
    try {
      const buf = await promises.readFile(filePath);
      const mime = mimeForExt(path.extname(filePath));
      return new Response(buf, { headers: { "Content-Type": mime } });
    } catch {
      return new Response("Not Found", { status: 404 });
    }
  });
  electron.ipcMain.handle("dialog:openFolder", async () => {
    const result = await electron.dialog.showOpenDialog(mainWindow, {
      properties: ["openDirectory"]
    });
    if (result.canceled || result.filePaths.length === 0) return null;
    return result.filePaths[0];
  });
  electron.ipcMain.handle("fs:listImages", async (_event, dir) => {
    const entries = await promises.readdir(dir, { withFileTypes: true });
    const images = [];
    for (const entry of entries) {
      if (!entry.isFile()) continue;
      const ext = path.extname(entry.name).toLowerCase();
      if (!IMAGE_EXTS.has(ext)) continue;
      const imagePath = path.join(dir, entry.name);
      const hasCaption = await fileExists(captionPathForImage(imagePath));
      images.push({ path: imagePath, name: entry.name, hasCaption });
    }
    images.sort((a, b) => a.name.localeCompare(b.name, void 0, { numeric: true }));
    return images;
  });
  electron.ipcMain.handle("fs:readCaption", async (_event, imagePath) => {
    const txtPath = captionPathForImage(imagePath);
    try {
      return await promises.readFile(txtPath, "utf-8");
    } catch {
      return "";
    }
  });
  electron.ipcMain.handle("fs:writeCaption", async (_event, imagePath, text) => {
    const txtPath = captionPathForImage(imagePath);
    await promises.writeFile(txtPath, text, "utf-8");
    return true;
  });
  electron.ipcMain.handle("fs:deleteImage", async (_event, imagePath) => {
    await electron.shell.trashItem(imagePath);
    const txtPath = captionPathForImage(imagePath);
    try {
      await electron.shell.trashItem(txtPath);
    } catch {
    }
    return { ok: true };
  });
  electron.ipcMain.handle("fs:readImageMeta", async (_event, imagePath) => {
    const ext = path.extname(imagePath).toLowerCase();
    if (ext !== ".png") {
      return { positivePrompt: "" };
    }
    try {
      const buf = await promises.readFile(imagePath);
      const texts = extractPngTextChunks(buf);
      return { positivePrompt: extractPositivePrompt(texts) };
    } catch {
      return { positivePrompt: "" };
    }
  });
  electron.ipcMain.handle("fs:readImageBase64", async (_event, imagePath) => {
    const buf = await promises.readFile(imagePath);
    const ext = path.extname(imagePath);
    return {
      mimeType: mimeForExt(ext),
      base64: buf.toString("base64")
    };
  });
  electron.ipcMain.handle("settings:get", async () => loadSettings());
  electron.ipcMain.handle("settings:set", async (_event, settings) => {
    const current = await loadSettings();
    await saveSettings({
      ...current,
      ...settings,
      // Window geometry is owned by main; never let renderer wipe it
      windowWidth: current.windowWidth,
      windowHeight: current.windowHeight,
      windowX: current.windowX,
      windowY: current.windowY,
      windowMaximized: current.windowMaximized
    });
    return true;
  });
  await createWindow();
  electron.app.on("activate", () => {
    if (electron.BrowserWindow.getAllWindows().length === 0) void createWindow();
  });
});
electron.app.on("window-all-closed", () => {
  if (process.platform !== "darwin") electron.app.quit();
});
