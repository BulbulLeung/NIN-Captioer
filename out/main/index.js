"use strict";
const electron = require("electron");
const path = require("path");
const fs = require("fs");
const promises = require("fs/promises");
const child_process = require("child_process");
const util = require("util");
const os = require("os");
const execFileAsync = util.promisify(child_process.execFile);
const IMAGE_EXTS = /* @__PURE__ */ new Set([".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"]);
const FALLBACK_GPU = [{ id: "cuda:0", label: "cuda:0 (not detected)" }];
async function listCudaDevices() {
  try {
    const { stdout } = await execFileAsync(
      "nvidia-smi",
      ["--query-gpu=index,name", "--format=csv,noheader,nounits"],
      { timeout: 5e3, windowsHide: true, encoding: "utf8" }
    );
    const devices = [];
    for (const line of stdout.split(/\r?\n/)) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      const comma = trimmed.indexOf(",");
      if (comma < 0) continue;
      const indexStr = trimmed.slice(0, comma).trim();
      const name = trimmed.slice(comma + 1).trim();
      const index = Number(indexStr);
      if (!Number.isInteger(index) || index < 0) continue;
      const id = `cuda:${index}`;
      devices.push({
        id,
        label: name ? `${id} — ${name}` : id
      });
    }
    return devices.length > 0 ? devices : FALLBACK_GPU;
  } catch {
    return FALLBACK_GPU;
  }
}
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
let trainProc = null;
let modelDlProc = null;
let modelDlCancelled = false;
function trainerRoot() {
  if (electron.app.isPackaged) {
    return path.join(process.resourcesPath, "trainer");
  }
  const candidates = [
    path.join(__dirname, "../../trainer"),
    path.join(electron.app.getAppPath(), "trainer"),
    path.join(process.cwd(), "trainer")
  ];
  for (const c of candidates) {
    if (fs.existsSync(path.join(c, "train_krea2_lora.py"))) return c;
  }
  return candidates[0];
}
function trainerScriptPath() {
  return path.join(trainerRoot(), "train_krea2_lora.py");
}
function modelOpsScriptPath() {
  return path.join(trainerRoot(), "hf_model_ops.py");
}
function defaultModelDownloadPath() {
  return path.join(electron.app.getPath("userData"), "models");
}
function resolveModelDownloadPath(configured) {
  const trimmed = (configured || "").trim();
  return trimmed || defaultModelDownloadPath();
}
function parseCudaIndex(device) {
  const d = (device || "").trim().toLowerCase();
  if (d.startsWith("cuda:")) return d.slice(5);
  if (d === "cuda") return "0";
  return null;
}
let lastCpuSample = null;
let cachedCpuName = null;
function getCpuName() {
  if (cachedCpuName !== null) return cachedCpuName;
  const list = os.cpus();
  const model = list[0]?.model?.trim() || "";
  cachedCpuName = model || "Unknown CPU";
  return cachedCpuName;
}
function sampleCpu() {
  let idle = 0;
  let total = 0;
  for (const cpu of os.cpus()) {
    idle += cpu.times.idle;
    total += cpu.times.user + cpu.times.nice + cpu.times.sys + cpu.times.idle + cpu.times.irq;
  }
  return { idle, total };
}
function getCpuUsagePercent() {
  const current = sampleCpu();
  if (!lastCpuSample) {
    lastCpuSample = current;
    return 0;
  }
  const idleDelta = current.idle - lastCpuSample.idle;
  const totalDelta = current.total - lastCpuSample.total;
  lastCpuSample = current;
  if (totalDelta <= 0) return 0;
  const usage = 1 - idleDelta / totalDelta;
  return Math.round(Math.min(100, Math.max(0, usage * 100)) * 10) / 10;
}
const PROTECTED_PROCESS_NAMES = new Set(
  [
    "system",
    "registry",
    "smss",
    "smss.exe",
    "csrss",
    "csrss.exe",
    "wininit",
    "wininit.exe",
    "services",
    "services.exe",
    "lsass",
    "lsass.exe",
    "svchost",
    "svchost.exe",
    "dwm",
    "dwm.exe",
    "explorer",
    "explorer.exe",
    "winlogon",
    "winlogon.exe",
    "fontdrvhost",
    "fontdrvhost.exe",
    "sihost",
    "sihost.exe",
    "taskhostw",
    "taskhostw.exe",
    "runtimebroker",
    "runtimebroker.exe",
    "searchhost",
    "searchhost.exe",
    "startmenuexperiencehost",
    "startmenuexperiencehost.exe",
    "shellexperiencehost",
    "shellexperiencehost.exe",
    "textinputhost",
    "textinputhost.exe",
    "lockapp",
    "lockapp.exe",
    "applicationframehost",
    "applicationframehost.exe",
    "systemsettings",
    "systemsettings.exe",
    "memory compression",
    "secure system",
    "idle",
    "systemd",
    "init"
  ].map((s) => s.toLowerCase())
);
const WIN_PROTECTED_PATH_MARKERS = [
  "\\windows\\",
  "\\windowsapps\\",
  "\\systemapps\\",
  "\\program files\\windows defender\\",
  "\\program files (x86)\\windows defender\\"
];
const LINUX_PROTECTED_PATH_PREFIXES = ["/usr/lib/systemd", "/sbin/", "/usr/sbin/", "/lib/systemd"];
function normalizeProcessName(name) {
  return name.trim().toLowerCase();
}
function isProtectedGpuProcess(pid, name, exePath) {
  if (!Number.isInteger(pid) || pid <= 0) return true;
  if (pid === process.pid) return true;
  if (typeof process.ppid === "number" && pid === process.ppid) return true;
  const normName = normalizeProcessName(name);
  const base = normalizeProcessName(path.basename((exePath || name).replace(/\\/g, "/")));
  if (PROTECTED_PROCESS_NAMES.has(normName) || PROTECTED_PROCESS_NAMES.has(base)) {
    return true;
  }
  const pathLower = (exePath || "").trim().toLowerCase().replace(/\//g, "\\");
  if (process.platform === "win32") {
    if (!pathLower) {
      return true;
    }
    if (WIN_PROTECTED_PATH_MARKERS.some((m) => pathLower.includes(m))) return true;
    return false;
  }
  const unixPath = (exePath || "").trim().toLowerCase();
  if (!unixPath) return true;
  if (LINUX_PROTECTED_PATH_PREFIXES.some((p) => unixPath.startsWith(p))) return true;
  return false;
}
function finalizeGpuVramApps(apps) {
  return apps.filter((a) => Number.isFinite(a.memUsedMiB) && a.memUsedMiB >= 1 && a.name).sort((a, b) => b.memUsedMiB - a.memUsedMiB).slice(0, 12).map((a) => ({
    pid: a.pid,
    name: a.name,
    memUsedMiB: Math.round(a.memUsedMiB * 10) / 10,
    killable: !isProtectedGpuProcess(a.pid, a.name, a.path ?? null)
  }));
}
async function resolveProcessIdentity(pid) {
  if (!Number.isInteger(pid) || pid <= 0) return null;
  try {
    if (process.platform === "win32") {
      const script = `
$ErrorActionPreference = 'SilentlyContinue'
$proc = Get-Process -Id ${pid} -ErrorAction SilentlyContinue
if (-not $proc) { '' } else {
  $name = if ($proc.Path) { [System.IO.Path]::GetFileName($proc.Path) } else { $proc.ProcessName }
  (@{ name = $name; path = $proc.Path }) | ConvertTo-Json -Compress
}
`.trim();
      const { stdout: stdout2 } = await execFileAsync(
        "powershell.exe",
        ["-NoProfile", "-NonInteractive", "-Command", script],
        { timeout: 5e3, windowsHide: true, encoding: "utf8" }
      );
      const text = stdout2.trim();
      if (!text) return null;
      const parsed = JSON.parse(text);
      return {
        name: String(parsed.name || ""),
        path: parsed.path ? String(parsed.path) : null
      };
    }
    const { stdout } = await execFileAsync("ps", ["-p", String(pid), "-o", "comm=", "-o", "args="], {
      timeout: 3e3,
      encoding: "utf8"
    });
    const line = stdout.trim();
    if (!line) return null;
    const parts = line.split(/\s+/);
    const name = parts[0] || "";
    const pathGuess = parts.find((p) => p.startsWith("/")) || null;
    return { name, path: pathGuess };
  } catch {
    return null;
  }
}
async function getParentPid(pid) {
  try {
    if (process.platform === "win32") {
      const script = `
$ErrorActionPreference = 'SilentlyContinue'
$p = Get-CimInstance Win32_Process -Filter "ProcessId=${pid}" -ErrorAction SilentlyContinue
if ($p -and $p.ParentProcessId) { $p.ParentProcessId } else { '' }
`.trim();
      const { stdout: stdout2 } = await execFileAsync(
        "powershell.exe",
        ["-NoProfile", "-NonInteractive", "-Command", script],
        { timeout: 5e3, windowsHide: true, encoding: "utf8" }
      );
      const n2 = Number(stdout2.trim());
      return Number.isInteger(n2) && n2 > 0 ? n2 : null;
    }
    const { stdout } = await execFileAsync("ps", ["-o", "ppid=", "-p", String(pid)], {
      timeout: 3e3,
      encoding: "utf8"
    });
    const n = Number(stdout.trim());
    return Number.isInteger(n) && n > 0 ? n : null;
  } catch {
    return null;
  }
}
async function findKillableProcessRoot(pid, name, exePath) {
  let current = { pid, name, path: exePath };
  const visited = /* @__PURE__ */ new Set([pid]);
  for (let i = 0; i < 8; i++) {
    const parentPid = await getParentPid(current.pid);
    if (!parentPid || parentPid <= 4 || visited.has(parentPid)) break;
    visited.add(parentPid);
    const parent = await resolveProcessIdentity(parentPid);
    if (!parent?.name) break;
    if (isProtectedGpuProcess(parentPid, parent.name, parent.path)) break;
    current = { pid: parentPid, name: parent.name, path: parent.path };
  }
  return current;
}
async function killProcessByPid(pid) {
  if (!Number.isInteger(pid) || pid <= 0) {
    return { ok: false, error: "Invalid pid" };
  }
  const identity = await resolveProcessIdentity(pid);
  if (!identity || !identity.name) {
    return { ok: false, error: "Process not found" };
  }
  if (isProtectedGpuProcess(pid, identity.name, identity.path)) {
    return { ok: false, error: "Protected process" };
  }
  const root = await findKillableProcessRoot(pid, identity.name, identity.path);
  try {
    if (process.platform === "win32") {
      await execFileAsync("taskkill", ["/pid", String(root.pid), "/T", "/F"], {
        timeout: 1e4,
        windowsHide: true,
        encoding: "utf8"
      });
      const stillTarget = await resolveProcessIdentity(pid);
      if (stillTarget) {
        return { ok: false, error: "Process still running after kill" };
      }
      return { ok: true };
    }
    try {
      process.kill(root.pid, "SIGTERM");
    } catch {
      process.kill(root.pid, "SIGKILL");
    }
    return { ok: true };
  } catch (err) {
    return {
      ok: false,
      error: err instanceof Error ? err.message : "Failed to kill process"
    };
  }
}
function parseTrailingNumber(parts, offsetFromEnd) {
  return Number(parts[parts.length - 1 - offsetFromEnd]);
}
async function listGpuVramAppsWin(gpuIndex) {
  const script = `
$ErrorActionPreference = 'SilentlyContinue'
$idx = ${gpuIndex}
$byPid = @{}
$samples = (Get-Counter '\\GPU Process Memory(*)\\Dedicated Usage').CounterSamples
foreach ($s in $samples) {
  if ($s.InstanceName -match 'pid_(\\d+).*_phys_(\\d+)') {
    $procId = [int]$Matches[1]
    $phys = [int]$Matches[2]
    if ($phys -ne $idx) { continue }
    if (-not $byPid.ContainsKey($procId)) { $byPid[$procId] = [double]0 }
    $byPid[$procId] += [double]$s.CookedValue
  }
}
$rows = New-Object System.Collections.Generic.List[object]
foreach ($procId in @($byPid.Keys)) {
  $mib = [math]::Round($byPid[$procId] / 1MB, 1)
  if ($mib -lt 1) { continue }
  $proc = Get-Process -Id $procId -ErrorAction SilentlyContinue
  if (-not $proc) { continue }
  $name = if ($proc.Path) { [System.IO.Path]::GetFileName($proc.Path) } else { $proc.ProcessName }
  if (-not $name) { continue }
  [void]$rows.Add([pscustomobject]@{ pid = $procId; name = $name; memUsedMiB = $mib; path = $proc.Path })
}
$sorted = @($rows | Sort-Object memUsedMiB -Descending | Select-Object -First 12)
if ($sorted.Count -eq 0) { '[]' } else { $sorted | ConvertTo-Json -Compress }
`.trim();
  const { stdout } = await execFileAsync(
    "powershell.exe",
    ["-NoProfile", "-NonInteractive", "-Command", script],
    { timeout: 15e3, windowsHide: true, encoding: "utf8" }
  );
  const text = stdout.trim();
  if (!text) return [];
  const parsed = JSON.parse(text);
  const rows = Array.isArray(parsed) ? parsed : [parsed];
  return finalizeGpuVramApps(
    rows.map((r) => ({
      pid: Number(r.pid),
      name: String(r.name || ""),
      memUsedMiB: Number(r.memUsedMiB),
      path: r.path ? String(r.path) : null
    }))
  );
}
async function listGpuVramAppsSmi(gpuIndex) {
  const { stdout } = await execFileAsync(
    "nvidia-smi",
    [
      "-i",
      String(gpuIndex),
      "--query-compute-apps=pid,process_name,used_gpu_memory",
      "--format=csv,noheader,nounits"
    ],
    { timeout: 5e3, windowsHide: true, encoding: "utf8" }
  );
  const byPid = /* @__PURE__ */ new Map();
  for (const line of stdout.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    const parts = trimmed.split(",").map((s) => s.trim());
    if (parts.length < 3) continue;
    const pid = Number(parts[0]);
    const memRaw = parts[parts.length - 1];
    if (/n\/a/i.test(memRaw)) continue;
    const memUsedMiB = Number(memRaw);
    const pathOrName = parts.slice(1, -1).join(",");
    if (!Number.isInteger(pid) || pid < 0 || !Number.isFinite(memUsedMiB)) continue;
    const name = path.basename(pathOrName.replace(/\\/g, "/")) || pathOrName;
    if (!name || /insufficient permissions/i.test(name)) continue;
    const exePath = pathOrName.includes("/") || pathOrName.includes("\\") ? pathOrName : null;
    const existing = byPid.get(pid);
    if (existing) {
      existing.memUsedMiB += memUsedMiB;
    } else {
      byPid.set(pid, { pid, name, memUsedMiB, path: exePath });
    }
  }
  return finalizeGpuVramApps([...byPid.values()]);
}
async function listGpuVramApps(gpuIndex) {
  try {
    if (process.platform === "win32") {
      return await listGpuVramAppsWin(gpuIndex);
    }
    return await listGpuVramAppsSmi(gpuIndex);
  } catch {
    return [];
  }
}
async function getGpuStats(deviceId) {
  const indexStr = parseCudaIndex(deviceId);
  if (indexStr === null) return null;
  const targetIndex = Number(indexStr);
  if (!Number.isInteger(targetIndex) || targetIndex < 0) return null;
  try {
    const { stdout } = await execFileAsync(
      "nvidia-smi",
      [
        "-i",
        String(targetIndex),
        "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
        "--format=csv,noheader,nounits"
      ],
      { timeout: 5e3, windowsHide: true, encoding: "utf8" }
    );
    const trimmed = stdout.trim().split(/\r?\n/)[0]?.trim();
    if (!trimmed) return null;
    const parts = trimmed.split(",").map((s) => s.trim());
    if (parts.length < 7) return null;
    const powerLimit = parseTrailingNumber(parts, 0);
    const powerDraw = parseTrailingNumber(parts, 1);
    const temp = parseTrailingNumber(parts, 2);
    const memTotal = parseTrailingNumber(parts, 3);
    const memUsed = parseTrailingNumber(parts, 4);
    const util2 = parseTrailingNumber(parts, 5);
    const name = parts.slice(0, parts.length - 6).join(", ");
    const apps = await listGpuVramApps(targetIndex);
    return {
      id: `cuda:${targetIndex}`,
      name: name || `cuda:${targetIndex}`,
      utilPercent: Number.isFinite(util2) ? util2 : 0,
      memUsedMiB: Number.isFinite(memUsed) ? memUsed : 0,
      memTotalMiB: Number.isFinite(memTotal) ? memTotal : 0,
      tempC: Number.isFinite(temp) ? temp : null,
      powerDrawW: Number.isFinite(powerDraw) ? powerDraw : null,
      powerLimitW: Number.isFinite(powerLimit) ? powerLimit : null,
      apps
    };
  } catch {
    return null;
  }
}
async function getResourceStats(deviceId) {
  const ramTotalBytes = os.totalmem();
  const ramUsedBytes = Math.max(0, ramTotalBytes - os.freemem());
  const gpu = deviceId?.trim() ? await getGpuStats(deviceId.trim()) : null;
  return {
    cpuName: getCpuName(),
    cpuPercent: getCpuUsagePercent(),
    ramUsedBytes,
    ramTotalBytes,
    gpu
  };
}
function killChildProcess(proc) {
  if (!proc || proc.killed) return;
  try {
    if (process.platform === "win32" && proc.pid) {
      child_process.spawn("taskkill", ["/pid", String(proc.pid), "/T", "/F"], {
        windowsHide: true,
        stdio: "ignore"
      });
    } else {
      proc.kill("SIGTERM");
    }
  } catch {
    try {
      proc.kill("SIGKILL");
    } catch {
    }
  }
}
function killTrainProcess() {
  if (!trainProc || trainProc.killed) {
    trainProc = null;
    return;
  }
  const proc = trainProc;
  trainProc = null;
  killChildProcess(proc);
}
function killModelDownloadProcess() {
  if (!modelDlProc || modelDlProc.killed) {
    modelDlProc = null;
    return;
  }
  modelDlCancelled = true;
  const proc = modelDlProc;
  modelDlProc = null;
  killChildProcess(proc);
}
function emitTrain(channel, payload) {
  mainWindow?.webContents.send(channel, payload);
}
function emitModel(channel, payload) {
  mainWindow?.webContents.send(channel, payload);
}
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
  electron.ipcMain.handle(
    "dialog:openFile",
    async (_event, opts) => {
      const result = await electron.dialog.showOpenDialog(mainWindow, {
        title: opts?.title,
        properties: ["openFile"],
        filters: opts?.filters ?? [{ name: "All Files", extensions: ["*"] }]
      });
      if (result.canceled || result.filePaths.length === 0) return null;
      return result.filePaths[0];
    }
  );
  electron.ipcMain.handle("gpu:listDevices", async () => listCudaDevices());
  electron.ipcMain.handle(
    "system:getResourceStats",
    async (_event, deviceId) => getResourceStats(deviceId)
  );
  electron.ipcMain.handle("system:killProcess", async (_event, pid) => killProcessByPid(pid));
  electron.ipcMain.handle("train:status", async () => ({
    running: Boolean(trainProc && !trainProc.killed)
  }));
  electron.ipcMain.handle("train:stop", async () => {
    killTrainProcess();
    emitTrain("train:log", { line: "Training stopped by user", stream: "system" });
    return { ok: true };
  });
  electron.ipcMain.handle("train:checkEnv", async (_event, pythonPath) => {
    const py = pythonPath && pythonPath.trim() || "python";
    const code = [
      "import importlib, sys",
      'mods = ["torch", "diffusers", "peft", "safetensors", "PIL"]',
      "missing = []",
      "for m in mods:",
      "  try:",
      "    importlib.import_module(m)",
      "  except Exception:",
      "    missing.append(m)",
      "ok_krea = False",
      "try:",
      "  from diffusers import Krea2Pipeline  # noqa: F401",
      "  ok_krea = True",
      "except Exception as e:",
      '  missing.append("diffusers.Krea2Pipeline")',
      "cuda = False",
      "try:",
      "  import torch",
      "  cuda = bool(torch.cuda.is_available())",
      "except Exception:",
      "  pass",
      'print("MISSING:" + ",".join(missing))',
      'print("CUDA:" + ("1" if cuda else "0"))',
      'print("KREA:" + ("1" if ok_krea else "0"))',
      "sys.exit(0 if not missing else 1)"
    ].join("\n");
    try {
      const { stdout, stderr } = await execFileAsync(py, ["-c", code], {
        timeout: 6e4,
        windowsHide: true,
        encoding: "utf8",
        maxBuffer: 2 * 1024 * 1024
      });
      const out = `${stdout}
${stderr}`;
      const missLine = out.split(/\r?\n/).find((l) => l.startsWith("MISSING:"));
      const missing = (missLine?.slice("MISSING:".length) || "").split(",").map((s) => s.trim()).filter(Boolean);
      const cuda = /CUDA:1/.test(out);
      const krea = /KREA:1/.test(out);
      if (missing.length === 0) {
        return {
          ok: true,
          message: `OK (${py})${cuda ? " · CUDA available" : " · CUDA not detected"}${krea ? " · Krea2Pipeline found" : ""}`
        };
      }
      return {
        ok: false,
        message: `Missing: ${missing.join(", ")}. Install trainer/requirements.txt (and recent diffusers for Krea2).`
      };
    } catch (err) {
      const e = err;
      const out = `${e.stdout || ""}
${e.stderr || ""}`;
      const missLine = out.split(/\r?\n/).find((l) => l.startsWith("MISSING:"));
      if (missLine) {
        const missing = missLine.slice("MISSING:".length).split(",").map((s) => s.trim()).filter(Boolean);
        return {
          ok: false,
          message: `Missing: ${missing.join(", ")}. Install trainer/requirements.txt (and recent diffusers for Krea2).`
        };
      }
      return {
        ok: false,
        message: `Failed to run Python (${py}): ${e.message || String(err)}`
      };
    }
  });
  electron.ipcMain.handle(
    "train:start",
    async (_event, opts) => {
      if (trainProc && !trainProc.killed) {
        return { ok: false, error: "Training already running" };
      }
      const script = trainerScriptPath();
      if (!fs.existsSync(script)) {
        return { ok: false, error: `Trainer script not found: ${script}` };
      }
      const py = opts.pythonPath && opts.pythonPath.trim() || "python";
      const dir = await promises.mkdtemp(path.join(os.tmpdir(), "captioer-train-"));
      const configPath = path.join(dir, "config.json");
      await promises.writeFile(configPath, opts.configJson, "utf-8");
      const env = { ...process.env };
      const cudaIdx = parseCudaIndex(opts.device || "");
      if (cudaIdx !== null) env.CUDA_VISIBLE_DEVICES = cudaIdx;
      try {
        const child = child_process.spawn(py, [script, "--config", configPath], {
          env,
          windowsHide: true,
          cwd: trainerRoot()
        });
        trainProc = child;
        emitTrain("train:log", { line: `Started: ${py} ${script}`, stream: "system" });
        const onChunk = (chunk, stream) => {
          const text = chunk.toString("utf8");
          for (const raw of text.split(/\r?\n/)) {
            const line = raw.trimEnd();
            if (!line) continue;
            emitTrain("train:log", { line, stream });
            const m = line.match(
              /^CAPTIOER_PROGRESS\s+step=(\d+)\s+total=(\d+)\s+loss=([0-9.eE+-]+)/
            );
            if (m) {
              emitTrain("train:progress", {
                step: Number(m[1]),
                total: Number(m[2]),
                loss: Number(m[3])
              });
            }
            const done = line.match(/^CAPTIOER_DONE\s+path=(.+)$/);
            if (done) {
              emitTrain("train:done", { path: done[1].trim() });
            }
          }
        };
        child.stdout.on("data", (c) => onChunk(c, "stdout"));
        child.stderr.on("data", (c) => onChunk(c, "stderr"));
        child.on("error", (err) => {
          emitTrain("train:error", { message: err.message });
          trainProc = null;
        });
        child.on("close", (code, signal) => {
          const was = trainProc === child;
          if (was) trainProc = null;
          if (code === 0) {
            emitTrain("train:log", { line: "Process exited OK", stream: "system" });
          } else {
            emitTrain("train:error", {
              message: `Process exited code=${code}${signal ? ` signal=${signal}` : ""}`
            });
          }
        });
        return { ok: true, configPath };
      } catch (err) {
        trainProc = null;
        return {
          ok: false,
          error: err instanceof Error ? err.message : String(err)
        };
      }
    }
  );
  electron.ipcMain.handle("model:defaultDownloadPath", async () => defaultModelDownloadPath());
  electron.ipcMain.handle(
    "model:checkStatus",
    async (_event, opts) => {
      const script = modelOpsScriptPath();
      if (!fs.existsSync(script)) {
        return { ok: false, error: `Model ops script not found: ${script}`, results: [] };
      }
      const py = opts.pythonPath && opts.pythonPath.trim() || "python";
      const downloadPath = resolveModelDownloadPath(opts.downloadPath);
      const targetsJson = JSON.stringify(opts.targets || []);
      const args = [
        script,
        "check",
        "--download-path",
        downloadPath,
        "--targets",
        targetsJson
      ];
      const token = (opts.token || "").trim();
      if (token) {
        args.push("--token", token);
      }
      try {
        const { stdout, stderr } = await execFileAsync(py, args, {
          timeout: 12e4,
          windowsHide: true,
          encoding: "utf8",
          maxBuffer: 4 * 1024 * 1024,
          cwd: trainerRoot()
        });
        const out = `${stdout}
${stderr}`;
        const jsonLine = out.split(/\r?\n/).map((l) => l.trim()).filter(Boolean).reverse().find((l) => l.startsWith("{"));
        if (!jsonLine) {
          return { ok: false, error: "No response from model check", results: [] };
        }
        const parsed = JSON.parse(jsonLine);
        return {
          ok: Boolean(parsed.ok),
          error: parsed.error,
          results: Array.isArray(parsed.results) ? parsed.results : [],
          downloadPath
        };
      } catch (err) {
        const e = err;
        const out = `${e.stdout || ""}
${e.stderr || ""}`;
        const jsonLine = out.split(/\r?\n/).map((l) => l.trim()).filter(Boolean).reverse().find((l) => l.startsWith("{"));
        if (jsonLine) {
          try {
            const parsed = JSON.parse(jsonLine);
            return {
              ok: Boolean(parsed.ok),
              error: parsed.error || e.message,
              results: Array.isArray(parsed.results) ? parsed.results : [],
              downloadPath
            };
          } catch {
          }
        }
        return {
          ok: false,
          error: e.message || String(err),
          results: [],
          downloadPath
        };
      }
    }
  );
  electron.ipcMain.handle(
    "model:download",
    async (_event, opts) => {
      if (modelDlProc && !modelDlProc.killed) {
        return { ok: false, error: "A model download is already running" };
      }
      const script = modelOpsScriptPath();
      if (!fs.existsSync(script)) {
        return { ok: false, error: `Model ops script not found: ${script}` };
      }
      const repoId = (opts.repoId || "").trim();
      if (!repoId) {
        return { ok: false, error: "repoId required" };
      }
      const py = opts.pythonPath && opts.pythonPath.trim() || "python";
      const downloadPath = resolveModelDownloadPath(opts.downloadPath);
      const args = [script, "download", "--download-path", downloadPath, "--repo-id", repoId];
      const token = (opts.token || "").trim();
      if (token) args.push("--token", token);
      try {
        const env = { ...process.env };
        if (token) {
          env.HF_TOKEN = token;
          env.HUGGING_FACE_HUB_TOKEN = token;
        }
        const child = child_process.spawn(py, args, {
          windowsHide: true,
          cwd: trainerRoot(),
          env
        });
        modelDlProc = child;
        modelDlCancelled = false;
        let finished = false;
        emitModel("model:downloadProgress", { repoId, pct: 0 });
        const onChunk = (chunk) => {
          const text = chunk.toString("utf8");
          for (const raw of text.split(/\r?\n/)) {
            const line = raw.trim();
            if (!line) continue;
            const prog = line.match(/^CAPTIOER_MODEL_PROGRESS\s+repo=(\S+)\s+pct=(\d+)/);
            if (prog) {
              emitModel("model:downloadProgress", {
                repoId: prog[1],
                pct: Number(prog[2])
              });
              continue;
            }
            const done = line.match(
              /^CAPTIOER_MODEL_DONE\s+repo=(\S+)\s+path=(.+?)\s+revision=(\S+)\s*$/
            );
            if (done) {
              finished = true;
              emitModel("model:downloadDone", {
                repoId: done[1],
                path: done[2].trim(),
                revision: done[3]
              });
              continue;
            }
            const errLine = line.match(/^CAPTIOER_MODEL_ERROR\s+message=(.+)$/);
            if (errLine) {
              finished = true;
              emitModel("model:downloadError", { message: errLine[1].trim(), repoId });
            }
          }
        };
        child.stdout.on("data", onChunk);
        child.stderr.on("data", onChunk);
        child.on("error", (err) => {
          if (modelDlCancelled) {
            modelDlCancelled = false;
            modelDlProc = null;
            return;
          }
          finished = true;
          emitModel("model:downloadError", { message: err.message, repoId });
          modelDlProc = null;
        });
        child.on("close", (code) => {
          const was = modelDlProc === child;
          if (was) modelDlProc = null;
          if (modelDlCancelled) {
            modelDlCancelled = false;
            return;
          }
          if (!finished && code !== 0 && code !== null) {
            emitModel("model:downloadError", {
              message: `Download exited code=${code}`,
              repoId
            });
          }
        });
        return { ok: true, downloadPath };
      } catch (err) {
        modelDlProc = null;
        return {
          ok: false,
          error: err instanceof Error ? err.message : String(err)
        };
      }
    }
  );
  electron.ipcMain.handle("model:cancelDownload", async () => {
    killModelDownloadProcess();
    return { ok: true };
  });
  electron.ipcMain.handle("model:downloadStatus", async () => ({
    running: Boolean(modelDlProc && !modelDlProc.killed)
  }));
  electron.ipcMain.handle(
    "dialog:saveTextFile",
    async (_event, opts) => {
      const result = await electron.dialog.showSaveDialog(mainWindow, {
        defaultPath: opts.defaultPath || "train_lora_config.yaml",
        filters: opts.filters ?? [
          { name: "YAML", extensions: ["yaml", "yml"] },
          { name: "All Files", extensions: ["*"] }
        ]
      });
      if (result.canceled || !result.filePath) return null;
      await promises.writeFile(result.filePath, opts.content ?? "", "utf-8");
      return result.filePath;
    }
  );
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
