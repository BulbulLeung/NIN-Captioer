"use strict";
const electron = require("electron");
const path = require("path");
const fs = require("fs");
const promises = require("fs/promises");
const child_process = require("child_process");
const util = require("util");
const os = require("os");
const https = require("https");
const promises$1 = require("stream/promises");
const execFileAsync$1 = util.promisify(child_process.execFile);
let installProc = null;
let installCancelled = false;
function defaultPythonInstallPath() {
  return path.join(electron.app.getPath("userData"), "python");
}
function resolvePythonInstallPath(raw) {
  const trimmed = (raw || "").trim();
  return trimmed || defaultPythonInstallPath();
}
function venvPythonPath(installRoot) {
  return process.platform === "win32" ? path.join(installRoot, "venv", "Scripts", "python.exe") : path.join(installRoot, "venv", "bin", "python");
}
function uvBinaryPath(installRoot) {
  return process.platform === "win32" ? path.join(installRoot, "tools", "uv.exe") : path.join(installRoot, "tools", "uv");
}
function uvDownloadUrl() {
  if (process.platform === "win32") {
    return "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip";
  }
  if (process.platform === "darwin") {
    const arch2 = process.arch === "arm64" ? "aarch64" : "x86_64";
    return `https://github.com/astral-sh/uv/releases/latest/download/uv-${arch2}-apple-darwin.tar.gz`;
  }
  const arch = process.arch === "arm64" ? "aarch64" : "x86_64";
  return `https://github.com/astral-sh/uv/releases/latest/download/uv-${arch}-unknown-linux-gnu.tar.gz`;
}
function httpsGetFollow(url, redirects = 0) {
  return new Promise((resolve, reject) => {
    https.get(url, (res) => {
      if (res.statusCode && res.statusCode >= 300 && res.statusCode < 400 && res.headers.location && redirects < 8) {
        res.resume();
        resolve(httpsGetFollow(res.headers.location, redirects + 1));
        return;
      }
      if (!res.statusCode || res.statusCode >= 400) {
        reject(new Error(`HTTP ${res.statusCode} for ${url}`));
        res.resume();
        return;
      }
      resolve(res);
    }).on("error", reject);
  });
}
async function downloadFile(url, dest) {
  await promises.mkdir(path.dirname(dest), { recursive: true });
  const res = await httpsGetFollow(url);
  await promises$1.pipeline(res, fs.createWriteStream(dest));
}
async function extractArchive(archivePath, destDir) {
  await promises.mkdir(destDir, { recursive: true });
  if (archivePath.endsWith(".zip")) {
    if (process.platform === "win32") {
      await execFileAsync$1(
        "powershell.exe",
        [
          "-NoProfile",
          "-Command",
          `Expand-Archive -LiteralPath '${archivePath.replace(/'/g, "''")}' -DestinationPath '${destDir.replace(/'/g, "''")}' -Force`
        ],
        { windowsHide: true, timeout: 12e4 }
      );
      return;
    }
    await execFileAsync$1("unzip", ["-o", archivePath, "-d", destDir], { timeout: 12e4 });
    return;
  }
  await execFileAsync$1("tar", ["-xzf", archivePath, "-C", destDir], { timeout: 12e4 });
}
function runSpawn(command, args, opts) {
  return new Promise((resolve, reject) => {
    if (installCancelled) {
      reject(new Error("Cancelled"));
      return;
    }
    const child = child_process.spawn(command, args, {
      cwd: opts?.cwd,
      env: opts?.env ?? process.env,
      windowsHide: true
    });
    installProc = child;
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (c) => {
      stdout += c.toString("utf8");
    });
    child.stderr.on("data", (c) => {
      stderr += c.toString("utf8");
    });
    child.on("error", (err) => {
      if (installProc === child) installProc = null;
      reject(err);
    });
    child.on("close", (code) => {
      if (installProc === child) installProc = null;
      if (installCancelled) {
        reject(new Error("Cancelled"));
        return;
      }
      resolve({ code: code ?? 1, stdout, stderr });
    });
  });
}
function cancelPythonInstall() {
  installCancelled = true;
  if (installProc && !installProc.killed) {
    try {
      installProc.kill();
    } catch {
    }
  }
  installProc = null;
  return { ok: true };
}
function pythonInstallRunning() {
  return Boolean(installProc && !installProc.killed);
}
async function probePython(pythonPath) {
  const py = pythonPath && pythonPath.trim() || "python";
  const code = [
    "import importlib, sys",
    'ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"',
    'mods = ["torch", "diffusers", "peft", "safetensors", "PIL", "onnxruntime", "numpy", "huggingface_hub"]',
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
    "except Exception:",
    '  missing.append("diffusers.Krea2Pipeline")',
    "cuda = False",
    "try:",
    "  import torch",
    "  cuda = bool(torch.cuda.is_available())",
    "except Exception:",
    "  pass",
    'print("VER:" + ver)',
    'print("MISSING:" + ",".join(missing))',
    'print("CUDA:" + ("1" if cuda else "0"))',
    'print("KREA:" + ("1" if ok_krea else "0"))',
    "ok_triton = False",
    "try:",
    "  import triton  # noqa: F401",
    "  ok_triton = True",
    "except Exception:",
    "  pass",
    'print("TRITON:" + ("1" if ok_triton else "0"))'
  ].join("\n");
  try {
    const { stdout, stderr } = await execFileAsync$1(py, ["-c", code], {
      timeout: 6e4,
      windowsHide: true,
      encoding: "utf8",
      maxBuffer: 2 * 1024 * 1024
    });
    const out = `${stdout}
${stderr}`;
    const ver = out.split(/\r?\n/).find((l) => l.startsWith("VER:"))?.slice(4) || "";
    const missLine = out.split(/\r?\n/).find((l) => l.startsWith("MISSING:"));
    const missing = (missLine?.slice("MISSING:".length) || "").split(",").map((s) => s.trim()).filter(Boolean);
    const cuda = /CUDA:1/.test(out);
    const krea = /KREA:1/.test(out);
    const triton = /TRITON:1/.test(out);
    if (missing.length === 0) {
      return {
        status: "ready",
        message: `OK (${py} ${ver})${cuda ? " · CUDA" : " · no CUDA"}${krea ? " · Krea2" : ""}${triton ? " · Triton" : process.platform === "win32" ? " · no Triton" : ""}`,
        pythonPath: py,
        version: ver,
        cuda,
        krea,
        missing: []
      };
    }
    return {
      status: "missingPackages",
      message: `Missing: ${missing.join(", ")}`,
      pythonPath: py,
      version: ver,
      cuda,
      krea,
      missing
    };
  } catch (err) {
    const e = err;
    const msg = e.message || String(err);
    if (e.code === "ENOENT" || /not found|ENOENT|is not recognized/i.test(msg)) {
      return {
        status: "missingPython",
        message: `Python not found (${py})`,
        pythonPath: py
      };
    }
    return {
      status: "error",
      message: `Failed to probe Python (${py}): ${msg}`,
      pythonPath: py
    };
  }
}
async function ensureUv(installRoot, onProgress) {
  const uvPath = uvBinaryPath(installRoot);
  if (fs.existsSync(uvPath)) return uvPath;
  onProgress({ stage: "uv", message: "Downloading uv…", pct: 5 });
  const toolsDir = path.join(installRoot, "tools");
  await promises.mkdir(toolsDir, { recursive: true });
  const archiveName = process.platform === "win32" ? "uv.zip" : "uv.tar.gz";
  const archivePath = path.join(toolsDir, archiveName);
  await downloadFile(uvDownloadUrl(), archivePath);
  onProgress({ stage: "uv", message: "Extracting uv…", pct: 12 });
  const extractDir = path.join(toolsDir, "extract");
  await promises.rm(extractDir, { recursive: true, force: true });
  await extractArchive(archivePath, extractDir);
  const walk = (dir) => {
    for (const name of fs.readdirSync(dir)) {
      const p = path.join(dir, name);
      const st = fs.statSync(p);
      if (st.isDirectory()) {
        const found2 = walk(p);
        if (found2) return found2;
      } else if (name === "uv.exe" || name === "uv") {
        return p;
      }
    }
    return null;
  };
  const found = walk(extractDir);
  if (!found) throw new Error("uv binary not found in archive");
  await promises.rename(found, uvPath);
  await promises.rm(extractDir, { recursive: true, force: true });
  await promises.rm(archivePath, { force: true });
  return uvPath;
}
async function writeRequirementsNoTorch(src, dest) {
  const text = await promises.readFile(src, "utf8");
  const filtered = text.split(/\r?\n/).filter((line) => {
    const t = line.trim();
    if (!t || t.startsWith("#")) return true;
    return !/^torch\b/i.test(t) && !/^flash-attn\b/i.test(t) && !/^flash_attn\b/i.test(t);
  }).join("\n");
  await promises.writeFile(dest, filtered + "\n", "utf8");
}
function tritonWindowsSpecForTorch(torchVersion) {
  const m = torchVersion.match(/(\d+)\.(\d+)/);
  const major = m ? Number(m[1]) : 2;
  const minor = m ? Number(m[2]) : 9;
  if (major === 2 && minor <= 5) return "triton-windows>=3.1,<3.2";
  if (major === 2 && minor === 6) return "triton-windows>=3.2,<3.3";
  if (major === 2 && minor === 7) return "triton-windows>=3.3,<3.4";
  if (major === 2 && minor === 8) return "triton-windows>=3.4,<3.5";
  if (major === 2 && minor === 9) return "triton-windows>=3.5,<3.6";
  if (major === 2 && minor <= 11) return "triton-windows>=3.6,<3.7";
  return "triton-windows>=3.7,<3.8";
}
function torchaoSpecForTorch(torchVersion) {
  const m = torchVersion.match(/(\d+)\.(\d+)(?:\.(\d+))?/);
  const major = m ? Number(m[1]) : 2;
  const minor = m ? Number(m[2]) : 9;
  const patch = m && m[3] != null ? Number(m[3]) : 0;
  if (major === 2 && minor === 6) return "torchao>=0.9.0,<0.10.0";
  if (major === 2 && minor === 9 && patch >= 1) return "torchao==0.15.0";
  if (major === 2 && minor === 9) return "torchao==0.14.1";
  if (major === 2 && minor === 10) return "torchao==0.16.0";
  return "torchao>=0.15.0,<0.17.0";
}
async function readTorchVersion(py) {
  try {
    const { stdout } = await execFileAsync$1(
      py,
      ["-c", "import torch; print(torch.__version__)"],
      { timeout: 6e4, windowsHide: true, encoding: "utf8" }
    );
    return (stdout || "").trim();
  } catch {
    return "";
  }
}
async function readTorchCuda(py) {
  try {
    const { stdout } = await execFileAsync$1(
      py,
      ["-c", 'import torch; print(torch.version.cuda or "")'],
      { timeout: 6e4, windowsHide: true, encoding: "utf8" }
    );
    return (stdout || "").trim();
  } catch {
    return "";
  }
}
async function installTritonWindows(opts) {
  const { uv, py, installRoot, onProgress } = opts;
  onProgress({
    stage: "triton",
    message: "Installing triton-windows (GPU kernels)…",
    pct: 82
  });
  await runSpawn(uv, ["pip", "uninstall", "-y", "--python", py, "triton"], {
    cwd: installRoot
  });
  let r = await runSpawn(
    uv,
    [
      "pip",
      "install",
      "--python",
      py,
      "setuptools",
      "wheel",
      "nvidia-cuda-nvcc-cu12",
      "nvidia-cuda-runtime-cu12"
    ],
    { cwd: installRoot }
  );
  if (r.code !== 0) {
    return {
      ok: false,
      message: `triton Windows deps (setuptools/CUDA) failed: ${r.stderr || r.stdout}`
    };
  }
  const torchVer = await readTorchVersion(py);
  const spec = tritonWindowsSpecForTorch(torchVer || "2.9");
  onProgress({
    stage: "triton",
    message: `Installing ${spec} (torch ${torchVer || "unknown"})…`,
    pct: 85
  });
  r = await runSpawn(uv, ["pip", "install", "--python", py, "-U", spec], {
    cwd: installRoot
  });
  if (r.code !== 0) {
    return {
      ok: false,
      message: `triton-windows install failed: ${r.stderr || r.stdout}`
    };
  }
  try {
    await execFileAsync$1(py, ["-c", "import triton; print(triton.__version__)"], {
      timeout: 6e4,
      windowsHide: true,
      encoding: "utf8"
    });
  } catch (err) {
    const detail = err instanceof Error ? err.message : String(err);
    return { ok: false, message: `triton import failed after install: ${detail}` };
  }
  return { ok: true, message: `triton-windows OK (${spec})` };
}
function flashAttnWindowsWheelUrl(torchVersion, cudaVersion, pythonVersion) {
  const tm = torchVersion.match(/(\d+)\.(\d+)/);
  const cm = cudaVersion.match(/(\d+)\.(\d+)/);
  const pm = pythonVersion.match(/(\d+)\.(\d+)/);
  if (!tm || !pm) return null;
  const tMinor = Number(tm[2]);
  const pyTag = `cp${pm[1]}${pm[2]}`;
  const cuMajor = cm ? Number(cm[1]) : 12;
  const cuMinor = cm ? Number(cm[2]) : 8;
  let cuTag = "cu128";
  if (cuMajor === 13) cuTag = "cu130";
  else if (cuMinor <= 6) cuTag = "cu126";
  else if (cuMinor === 9) cuTag = "cu129";
  else cuTag = "cu128";
  if (tMinor !== 9) {
    return null;
  }
  const name = `flash_attn-2.8.3+${cuTag}torch2.9-${pyTag}-${pyTag}-win_amd64.whl`;
  const encoded = name.replace(/\+/g, "%2B");
  return `https://github.com/PozzettiAndrea/cuda-wheels/releases/download/flash_attn-latest/${encoded}`;
}
async function readPythonVersion(py) {
  try {
    const { stdout } = await execFileAsync$1(
      py,
      ["-c", 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'],
      { timeout: 3e4, windowsHide: true, encoding: "utf8" }
    );
    return (stdout || "").trim();
  } catch {
    return "";
  }
}
async function installFlashAttn(opts) {
  const { uv, py, installRoot, onProgress } = opts;
  onProgress({
    stage: "flash",
    message: "Installing flash-attn (FA2 wheel)…",
    pct: 86
  });
  await runSpawn(uv, ["pip", "install", "--python", py, "einops", "packaging", "ninja"], {
    cwd: installRoot
  });
  if (process.platform === "win32") {
    const torchVer = await readTorchVersion(py);
    const cudaVer = await readTorchCuda(py);
    const pyVer = await readPythonVersion(py);
    const wheelUrl = flashAttnWindowsWheelUrl(torchVer, cudaVer, pyVer || "3.11");
    if (!wheelUrl) {
      return {
        ok: false,
        message: `no flash-attn Windows wheel for torch ${torchVer || "?"} cuda ${cudaVer || "?"} py ${pyVer || "?"}`
      };
    }
    onProgress({
      stage: "flash",
      message: `Downloading flash-attn wheel (torch ${torchVer})…`,
      pct: 86
    });
    const wheelName = decodeURIComponent(wheelUrl.split("/").pop() || "flash_attn.whl");
    const wheelPath = path.join(installRoot, wheelName);
    try {
      await downloadFile(wheelUrl, wheelPath);
    } catch (err) {
      const detail = err instanceof Error ? err.message : String(err);
      return { ok: false, message: `flash-attn wheel download failed: ${detail}` };
    }
    const r = await runSpawn(
      uv,
      ["pip", "install", "--python", py, "--force-reinstall", "--no-deps", wheelPath],
      { cwd: installRoot }
    );
    await promises.rm(wheelPath, { force: true }).catch(() => void 0);
    if (r.code !== 0) {
      return {
        ok: false,
        message: `flash-attn wheel install failed: ${r.stderr || r.stdout}`
      };
    }
  } else {
    const r = await runSpawn(
      uv,
      ["pip", "install", "--python", py, "flash-attn", "--no-build-isolation"],
      { cwd: installRoot }
    );
    if (r.code !== 0) {
      return {
        ok: false,
        message: `flash-attn install failed: ${r.stderr || r.stdout}`
      };
    }
  }
  try {
    const { stdout } = await execFileAsync$1(
      py,
      [
        "-c",
        'from flash_attn import flash_attn_func; import flash_attn; print(getattr(flash_attn, "__version__", "ok"))'
      ],
      { timeout: 12e4, windowsHide: true, encoding: "utf8" }
    );
    const ver = (stdout || "").trim() || "ok";
    return { ok: true, message: `flash-attn ${ver} OK` };
  } catch (err) {
    const detail = err instanceof Error ? err.message : String(err);
    return { ok: false, message: `flash-attn import failed: ${detail}` };
  }
}
async function installPythonEnv(opts) {
  installCancelled = false;
  const installRoot = resolvePythonInstallPath(opts.installPath);
  const onProgress = opts.onProgress;
  try {
    await promises.mkdir(installRoot, { recursive: true });
    const uv = await ensureUv(installRoot, onProgress);
    onProgress({ stage: "python", message: "Installing Python 3.11…", pct: 20 });
    let r = await runSpawn(uv, ["python", "install", "3.11"], { cwd: installRoot });
    if (r.code !== 0) {
      throw new Error(`uv python install failed: ${r.stderr || r.stdout}`);
    }
    const venvDir = path.join(installRoot, "venv");
    onProgress({ stage: "venv", message: "Creating venv…", pct: 30 });
    r = await runSpawn(uv, ["venv", venvDir, "--python", "3.11"], { cwd: installRoot });
    if (r.code !== 0 && !fs.existsSync(venvPythonPath(installRoot))) {
      throw new Error(`uv venv failed: ${r.stderr || r.stdout}`);
    }
    const py = venvPythonPath(installRoot);
    if (!fs.existsSync(py)) {
      throw new Error(`venv python missing: ${py}`);
    }
    onProgress({ stage: "torch", message: "Installing CUDA torch 2.9.1 (cu128)…", pct: 40 });
    r = await runSpawn(
      uv,
      [
        "pip",
        "install",
        "--python",
        py,
        "torch==2.9.1",
        "torchvision==0.24.1",
        "--index-url",
        "https://download.pytorch.org/whl/cu128"
      ],
      { cwd: installRoot }
    );
    let torchMode = "CUDA cu128 (torch 2.9.1)";
    if (r.code !== 0) {
      onProgress({ stage: "torch", message: "CUDA torch failed; installing CPU torch…", pct: 45 });
      r = await runSpawn(
        uv,
        [
          "pip",
          "install",
          "--python",
          py,
          "torch==2.9.1",
          "torchvision==0.24.1",
          "--index-url",
          "https://download.pytorch.org/whl/cpu"
        ],
        { cwd: installRoot }
      );
      if (r.code !== 0) {
        throw new Error(`torch install failed: ${r.stderr || r.stdout}`);
      }
      torchMode = "CPU (torch 2.9.1)";
    }
    const reqTrain = path.join(opts.trainerRoot, "requirements.txt");
    const reqWd14 = path.join(opts.trainerRoot, "requirements-wd14.txt");
    const tmpReq = path.join(installRoot, "requirements-no-torch.txt");
    if (!fs.existsSync(reqTrain)) throw new Error(`Missing ${reqTrain}`);
    await writeRequirementsNoTorch(reqTrain, tmpReq);
    onProgress({ stage: "reqs", message: "Installing training requirements…", pct: 60 });
    r = await runSpawn(uv, ["pip", "install", "--python", py, "-r", tmpReq], {
      cwd: installRoot
    });
    if (r.code !== 0) {
      throw new Error(`requirements.txt install failed: ${r.stderr || r.stdout}`);
    }
    const torchVer = await readTorchVersion(py);
    const torchaoSpec = torchaoSpecForTorch(torchVer || "2.9.1");
    onProgress({
      stage: "torchao",
      message: `Pinning ${torchaoSpec} for torch ${torchVer || "unknown"}…`,
      pct: 72
    });
    r = await runSpawn(uv, ["pip", "install", "--python", py, torchaoSpec], {
      cwd: installRoot
    });
    if (r.code !== 0) {
      throw new Error(`torchao pin failed: ${r.stderr || r.stdout}`);
    }
    if (fs.existsSync(reqWd14)) {
      onProgress({ stage: "wd14", message: "Installing WD14 requirements…", pct: 75 });
      r = await runSpawn(uv, ["pip", "install", "--python", py, "-r", reqWd14], {
        cwd: installRoot
      });
      if (r.code !== 0) {
        throw new Error(`requirements-wd14.txt install failed: ${r.stderr || r.stdout}`);
      }
    }
    let tritonNote = "";
    let flashNote = "";
    if (process.platform === "win32" && torchMode.startsWith("CUDA")) {
      const tritonResult = await installTritonWindows({ uv, py, installRoot, onProgress });
      if (!tritonResult.ok) {
        tritonNote = ` Triton warning: ${tritonResult.message}`;
        onProgress({
          stage: "triton",
          message: `triton-windows skipped: ${tritonResult.message}`,
          pct: 86
        });
      } else {
        tritonNote = ` ${tritonResult.message}.`;
      }
      const flashResult = await installFlashAttn({ uv, py, installRoot, onProgress });
      if (!flashResult.ok) {
        flashNote = ` flash-attn warning: ${flashResult.message}`;
        onProgress({
          stage: "flash",
          message: `flash-attn skipped: ${flashResult.message}`,
          pct: 86
        });
      } else {
        flashNote = ` ${flashResult.message}.`;
      }
    } else if (process.platform === "win32") {
      tritonNote = " Triton skipped (CPU torch).";
    } else if (torchMode.startsWith("CUDA")) {
      const flashResult = await installFlashAttn({ uv, py, installRoot, onProgress });
      if (!flashResult.ok) {
        flashNote = ` flash-attn warning: ${flashResult.message}`;
      } else {
        flashNote = ` ${flashResult.message}.`;
      }
    }
    onProgress({ stage: "krea", message: "Checking Krea2Pipeline…", pct: 88 });
    const probe = await probePython(py);
    if (probe.missing?.includes("diffusers.Krea2Pipeline")) {
      onProgress({ stage: "krea", message: "Installing latest diffusers (git)…", pct: 90 });
      r = await runSpawn(
        uv,
        ["pip", "install", "--python", py, "git+https://github.com/huggingface/diffusers.git"],
        { cwd: installRoot }
      );
      if (r.code !== 0) {
        throw new Error(`diffusers git install failed: ${r.stderr || r.stdout}`);
      }
    }
    const finalProbe = await probePython(py);
    onProgress({ stage: "done", message: "Install complete", pct: 100 });
    return {
      ok: finalProbe.status === "ready" || finalProbe.status === "missingPackages",
      pythonPath: py,
      message: finalProbe.status === "ready" ? `Installed (${torchMode}).${tritonNote}${flashNote} ${finalProbe.message}` : `Installed (${torchMode}) with warnings:${tritonNote}${flashNote} ${finalProbe.message}`
    };
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return { ok: false, message };
  } finally {
    installProc = null;
    installCancelled = false;
  }
}
const execFileAsync = util.promisify(child_process.execFile);
const IMAGE_EXTS = /* @__PURE__ */ new Set([".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"]);
function safeJobName(name) {
  const cleaned = (name || "job").trim().replace(/[<>:"/\\|?*\x00-\x1f]+/g, "_") || "job";
  return cleaned.replace(/[ .]+$/g, "");
}
async function listStepLoraCheckpoints(trainingFolder, jobName) {
  const outDir = path.join(trainingFolder, jobName);
  const prefix = `${safeJobName(jobName)}_`;
  const suffix = ".safetensors";
  const found = [];
  try {
    await promises.access(outDir, promises.constants.R_OK);
  } catch {
    return [];
  }
  const entries = await promises.readdir(outDir, { withFileTypes: true });
  for (const ent of entries) {
    if (!ent.isFile()) continue;
    const name = ent.name;
    if (!name.startsWith(prefix) || !name.endsWith(suffix)) continue;
    const stepPart = name.slice(prefix.length, -suffix.length);
    if (!/^\d{6}$/.test(stepPart)) continue;
    const full = path.join(outDir, name);
    let mtime = 0;
    try {
      mtime = (await promises.stat(full)).mtimeMs;
    } catch {
      mtime = 0;
    }
    found.push({ step: Number(stepPart), path: full, mtime });
  }
  found.sort((a, b) => a.step - b.step || a.mtime - b.mtime);
  return found.map(({ step, path: path2 }) => ({ step, path: path2 }));
}
function parseSampleStepAndIndex(name, jobName) {
  const safeName = safeJobName(jobName);
  const escaped = safeName.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const match = name.match(new RegExp(`^${escaped}_Sampling_(\\d{6})(?:_(\\d+))?\\.png$`, "i"));
  if (!match) return null;
  return {
    step: Number(match[1]),
    // No `_N` suffix → single-prompt job, treat as prompt 1
    promptIndex: match[2] ? Number(match[2]) : 1
  };
}
async function listTrainSamples(trainingFolder, jobName) {
  const sampleDir = path.join(trainingFolder, jobName, "samples");
  try {
    await promises.access(sampleDir, promises.constants.R_OK);
  } catch {
    return [];
  }
  const entries = await promises.readdir(sampleDir, { withFileTypes: true });
  const found = [];
  for (const ent of entries) {
    if (!ent.isFile()) continue;
    const full = path.join(sampleDir, ent.name);
    const ext = path.extname(ent.name).toLowerCase();
    if (!IMAGE_EXTS.has(ext)) continue;
    let mtimeMs = 0;
    try {
      mtimeMs = (await promises.stat(full)).mtimeMs;
    } catch {
      mtimeMs = 0;
    }
    const parsed = parseSampleStepAndIndex(ent.name, jobName);
    found.push({
      path: full,
      name: ent.name,
      mtimeMs,
      step: parsed?.step,
      promptIndex: parsed?.promptIndex ?? 1
    });
  }
  found.sort((a, b) => {
    const stepA = a.step ?? -1;
    const stepB = b.step ?? -1;
    return a.promptIndex - b.promptIndex || stepA - stepB || a.mtimeMs - b.mtimeMs || a.name.localeCompare(b.name);
  });
  return found.map(({ path: path2, name, mtimeMs, step, promptIndex }) => ({
    path: path2,
    name,
    mtimeMs,
    step,
    promptIndex
  }));
}
electron.app.commandLine.appendSwitch("disable-http-cache");
electron.app.commandLine.appendSwitch("disk-cache-size", "0");
try {
  const cacheDir = path.join(electron.app.getPath("userData"), "Cache");
  if (fs.existsSync(cacheDir)) {
    fs.rmSync(cacheDir, { recursive: true, force: true });
  }
} catch {
}
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
function normalizeUiGpuMode(raw) {
  const mode = raw?.uiGpuMode;
  if (mode === "auto" || mode === "onboard" || mode === "software") return mode;
  if (raw?.disableUiGpu === true) return "software";
  return "auto";
}
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
  uiGpuMode: "auto",
  disableUiGpu: false,
  windowWidth: DEFAULT_WINDOW.width,
  windowHeight: DEFAULT_WINDOW.height,
  windowX: null,
  windowY: null,
  windowMaximized: false
};
function settingsPath() {
  return path.join(electron.app.getPath("userData"), "settings.json");
}
function readUiGpuModeSync() {
  try {
    const raw = fs.readFileSync(settingsPath(), "utf-8");
    return normalizeUiGpuMode(JSON.parse(raw));
  } catch {
    return "auto";
  }
}
const earlyUiGpuMode = readUiGpuModeSync();
if (earlyUiGpuMode === "software") {
  electron.app.disableHardwareAcceleration();
  electron.app.commandLine.appendSwitch("disable-gpu");
} else if (earlyUiGpuMode === "onboard") {
  electron.app.commandLine.appendSwitch("force_low_power_gpu");
}
async function loadSettings() {
  try {
    const raw = await promises.readFile(settingsPath(), "utf-8");
    const parsed = JSON.parse(raw);
    const uiGpuMode = normalizeUiGpuMode(parsed);
    return {
      ...DEFAULT_SETTINGS,
      ...parsed,
      uiGpuMode,
      disableUiGpu: uiGpuMode === "software"
    };
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
let wd14Proc = null;
let wd14Cancelled = false;
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
function wd14TaggerScriptPath() {
  return path.join(trainerRoot(), "wd14_tagger.py");
}
function sanitizeRepoIdForDir(repoId) {
  return repoId.trim().replace(/[/\\]/g, "__");
}
function localWd14ModelDir(downloadPath, repoId) {
  return path.join(resolveModelDownloadPath(downloadPath), sanitizeRepoIdForDir(repoId));
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
const GPU_VRAM_APPS_FETCH_LIMIT = 64;
function finalizeGpuVramApps(apps) {
  return apps.filter((a) => Number.isFinite(a.memUsedMiB) && a.memUsedMiB >= 1 && a.name).sort((a, b) => b.memUsedMiB - a.memUsedMiB).slice(0, GPU_VRAM_APPS_FETCH_LIMIT).map((a) => ({
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
function killWd14Process() {
  if (!wd14Proc || wd14Proc.killed) {
    wd14Proc = null;
    return;
  }
  wd14Cancelled = true;
  const proc = wd14Proc;
  wd14Proc = null;
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
      return new Response(buf, {
        headers: {
          "Content-Type": mime,
          "Cache-Control": "no-store"
        }
      });
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
  electron.ipcMain.handle(
    "train:listCheckpoints",
    async (_event, opts) => {
      const trainingFolder = (opts?.trainingFolder || "").trim();
      const jobName = (opts?.jobName || "").trim();
      if (!trainingFolder || !jobName) {
        return { ok: true, checkpoints: [] };
      }
      try {
        const checkpoints = await listStepLoraCheckpoints(trainingFolder, jobName);
        return { ok: true, checkpoints };
      } catch (err) {
        return {
          ok: false,
          error: err instanceof Error ? err.message : String(err),
          checkpoints: []
        };
      }
    }
  );
  electron.ipcMain.handle(
    "train:listSamples",
    async (_event, opts) => {
      const trainingFolder = (opts?.trainingFolder || "").trim();
      const jobName = (opts?.jobName || "").trim();
      if (!trainingFolder || !jobName) {
        return { ok: true, samples: [] };
      }
      try {
        const samples = await listTrainSamples(trainingFolder, jobName);
        return { ok: true, samples };
      } catch (err) {
        return {
          ok: false,
          error: err instanceof Error ? err.message : String(err),
          samples: []
        };
      }
    }
  );
  electron.ipcMain.handle("train:stop", async () => {
    killTrainProcess();
    emitTrain("train:log", { line: "Training stopped by user", stream: "system" });
    return { ok: true };
  });
  electron.ipcMain.handle("download:defaultFolder", async () => electron.app.getPath("userData"));
  electron.ipcMain.handle("python:probe", async (_event, pythonPath) => probePython(pythonPath));
  electron.ipcMain.handle("python:cancelInstall", async () => cancelPythonInstall());
  electron.ipcMain.handle(
    "python:install",
    async (_event, opts) => {
      if (pythonInstallRunning()) {
        return { ok: false, message: "Python install already running" };
      }
      const result = await installPythonEnv({
        installPath: opts?.installPath,
        trainerRoot: trainerRoot(),
        onProgress: (p) => {
          mainWindow?.webContents.send("python:installProgress", p);
        }
      });
      return result;
    }
  );
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
        const recentLines = [];
        let structuredError = null;
        const onChunk = (chunk, stream) => {
          const text = chunk.toString("utf8");
          for (const raw of text.split(/\r?\n/)) {
            const line = raw.trimEnd();
            if (!line) continue;
            if (/MatMul8bitLt:/i.test(line)) continue;
            recentLines.push(`[${stream}] ${line}`);
            if (recentLines.length > 40) recentLines.shift();
            const trainErr = line.match(/^CAPTIOER_TRAIN_ERROR\s+message=(.+)$/);
            if (trainErr) structuredError = trainErr[1].trim();
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
            const detail = structuredError || recentLines.filter((l) => /ERROR:|Error|Traceback|Exception/i.test(l)).slice(-3).join(" | ");
            emitTrain("train:error", {
              message: detail ? `Process exited code=${code}${signal ? ` signal=${signal}` : ""}: ${detail}` : `Process exited code=${code}${signal ? ` signal=${signal}` : ""}`
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
        emitModel("model:downloadProgress", { repoId, pct: 0, done: 0, total: 0 });
        const onChunk = (chunk) => {
          const text = chunk.toString("utf8");
          for (const raw of text.split(/\r?\n/)) {
            const line = raw.trim();
            if (!line) continue;
            const prog = line.match(
              /^CAPTIOER_MODEL_PROGRESS\s+repo=(\S+)\s+pct=(\d+)(?:\s+done=(\d+))?(?:\s+total=(\d+))?/
            );
            if (prog) {
              emitModel("model:downloadProgress", {
                repoId: prog[1],
                pct: Number(prog[2]),
                done: prog[3] !== void 0 ? Number(prog[3]) : void 0,
                total: prog[4] !== void 0 ? Number(prog[4]) : void 0
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
  electron.ipcMain.handle(
    "wd14:ensureModel",
    async (_event, opts) => {
      if (wd14Proc && !wd14Proc.killed) {
        return { ok: false, error: "WD14 tagging is already running" };
      }
      const script = wd14TaggerScriptPath();
      if (!fs.existsSync(script)) {
        return { ok: false, error: `WD14 tagger script not found: ${script}` };
      }
      const repoId = (opts.repoId || "").trim();
      if (!repoId) {
        return { ok: false, error: "repoId required" };
      }
      const py = opts.pythonPath && opts.pythonPath.trim() || "python";
      const downloadPath = resolveModelDownloadPath(opts.downloadPath);
      const token = (opts.token || "").trim();
      const args = [
        script,
        "ensure",
        "--download-path",
        downloadPath,
        "--repo-id",
        repoId
      ];
      if (token) args.push("--token", token);
      return await new Promise((resolve) => {
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
          wd14Proc = child;
          wd14Cancelled = false;
          let fatalError = null;
          let modelDir = localWd14ModelDir(downloadPath, repoId);
          let stdoutBuf = "";
          const onChunk = (chunk) => {
            stdoutBuf += chunk.toString("utf8");
            const lines = stdoutBuf.split(/\r?\n/);
            stdoutBuf = lines.pop() ?? "";
            for (const raw of lines) {
              const line = raw.trim();
              if (!line) continue;
              const errLine = line.match(/^CAPTIOER_TAG_ERROR\s+message=(.+)$/);
              if (errLine) {
                fatalError = errLine[1].trim();
                continue;
              }
              const modelLine = line.match(/^CAPTIOER_TAG_MODEL\s+path=(.+?)\s+status=(\S+)/);
              if (modelLine) {
                modelDir = modelLine[1].trim();
              }
            }
          };
          child.stdout.on("data", onChunk);
          child.stderr.on("data", onChunk);
          child.on("error", (err) => {
            if (wd14Cancelled) {
              wd14Cancelled = false;
              wd14Proc = null;
              resolve({ ok: false, error: "Caption cancelled" });
              return;
            }
            wd14Proc = null;
            resolve({ ok: false, error: err.message });
          });
          child.on("close", (code) => {
            if (wd14Proc === child) wd14Proc = null;
            if (wd14Cancelled) {
              wd14Cancelled = false;
              resolve({ ok: false, error: "Caption cancelled" });
              return;
            }
            if (fatalError) {
              resolve({ ok: false, error: fatalError });
              return;
            }
            if (code !== 0 && code !== null) {
              resolve({
                ok: false,
                error: `WD14 ensure exited code=${code}`
              });
              return;
            }
            resolve({
              ok: true,
              modelDir: modelDir || localWd14ModelDir(downloadPath, repoId)
            });
          });
        } catch (err) {
          wd14Proc = null;
          resolve({
            ok: false,
            error: err instanceof Error ? err.message : String(err)
          });
        }
      });
    }
  );
  electron.ipcMain.handle(
    "wd14:tag",
    async (_event, opts) => {
      if (wd14Proc && !wd14Proc.killed) {
        return { ok: false, error: "WD14 tagging is already running", results: [] };
      }
      const script = wd14TaggerScriptPath();
      if (!fs.existsSync(script)) {
        return {
          ok: false,
          error: `WD14 tagger script not found: ${script}`,
          results: []
        };
      }
      const imagePaths = Array.isArray(opts.imagePaths) ? opts.imagePaths.filter((p) => typeof p === "string" && p) : [];
      if (imagePaths.length === 0) {
        return { ok: false, error: "No images provided", results: [] };
      }
      const modelDir = (opts.modelDir || "").trim();
      if (!modelDir) {
        return { ok: false, error: "modelDir required", results: [] };
      }
      const py = opts.pythonPath && opts.pythonPath.trim() || "python";
      const token = (opts.token || "").trim();
      let imagesFile = null;
      try {
        const tmpRoot = await promises.mkdtemp(path.join(os.tmpdir(), "captioer-wd14-"));
        imagesFile = path.join(tmpRoot, "images.txt");
        await promises.writeFile(imagesFile, imagePaths.join("\n"), "utf-8");
        const args = [
          script,
          "tag",
          "--model-dir",
          modelDir,
          "--threshold",
          String(opts.threshold ?? 0.35),
          "--character-threshold",
          String(opts.characterThreshold ?? 0.85),
          "--images-file",
          imagesFile
        ];
        if (opts.ensure) {
          args.push("--ensure");
          const downloadPath = resolveModelDownloadPath(opts.downloadPath);
          const repoId = (opts.repoId || "").trim();
          if (downloadPath) args.push("--download-path", downloadPath);
          if (repoId) args.push("--repo-id", repoId);
          if (token) args.push("--token", token);
        }
        return await new Promise((resolve) => {
          try {
            const env = {
              ...process.env,
              PYTHONIOENCODING: "utf-8",
              PYTHONUTF8: "1"
            };
            if (token) {
              env.HF_TOKEN = token;
              env.HUGGING_FACE_HUB_TOKEN = token;
            }
            const child = child_process.spawn(py, args, {
              windowsHide: true,
              cwd: trainerRoot(),
              env
            });
            wd14Proc = child;
            wd14Cancelled = false;
            let fatalError = null;
            const results = [];
            let stdoutBuf = "";
            const decodeB64Json = (b64) => {
              try {
                const json = Buffer.from(b64.trim(), "base64").toString("utf8");
                return JSON.parse(json);
              } catch {
                return null;
              }
            };
            const onStdoutLine = (line) => {
              if (!line) return;
              if (line.startsWith("CAPTIOER_TAG_ERROR message=")) {
                fatalError = line.slice("CAPTIOER_TAG_ERROR message=".length).trim();
                return;
              }
              if (line.startsWith("CAPTIOER_TAG_ITEM_ERROR_B64 ")) {
                const payload = decodeB64Json(line.slice("CAPTIOER_TAG_ITEM_ERROR_B64 ".length));
                if (payload && typeof payload.path === "string") {
                  results.push({
                    path: payload.path,
                    error: typeof payload.error === "string" ? payload.error : "Tag failed"
                  });
                }
                return;
              }
              if (line.startsWith("CAPTIOER_TAG_B64 ")) {
                const payload = decodeB64Json(line.slice("CAPTIOER_TAG_B64 ".length));
                if (payload && typeof payload.path === "string") {
                  results.push({
                    path: payload.path,
                    tags: typeof payload.tags === "string" ? payload.tags : ""
                  });
                }
                return;
              }
              if (line.startsWith("CAPTIOER_TAG_ITEM_ERROR ")) {
                try {
                  const payload = JSON.parse(line.slice("CAPTIOER_TAG_ITEM_ERROR ".length));
                  if (payload.path) {
                    results.push({ path: payload.path, error: payload.error || "Tag failed" });
                  }
                } catch {
                }
                return;
              }
              if (line.startsWith("CAPTIOER_TAG ") && !line.startsWith("CAPTIOER_TAG_")) {
                try {
                  const payload = JSON.parse(line.slice("CAPTIOER_TAG ".length));
                  if (payload.path) {
                    results.push({ path: payload.path, tags: payload.tags || "" });
                  }
                } catch {
                }
              }
            };
            const onStdoutChunk = (chunk) => {
              stdoutBuf += chunk.toString("utf8");
              const lines = stdoutBuf.split(/\r?\n/);
              stdoutBuf = lines.pop() ?? "";
              for (const raw of lines) onStdoutLine(raw.trim());
            };
            child.stdout.on("data", onStdoutChunk);
            child.stderr.on("data", () => {
            });
            child.on("error", (err) => {
              if (wd14Cancelled) {
                wd14Cancelled = false;
                wd14Proc = null;
                resolve({ ok: false, error: "Caption cancelled", results });
                return;
              }
              wd14Proc = null;
              resolve({ ok: false, error: err.message, results });
            });
            child.on("close", (code) => {
              if (wd14Proc === child) wd14Proc = null;
              if (stdoutBuf.trim()) onStdoutLine(stdoutBuf.trim());
              if (wd14Cancelled) {
                wd14Cancelled = false;
                resolve({ ok: false, error: "Caption cancelled", results });
                return;
              }
              if (fatalError) {
                resolve({ ok: false, error: fatalError, results });
                return;
              }
              if (code !== 0 && code !== null && results.length === 0) {
                resolve({
                  ok: false,
                  error: `WD14 tag exited code=${code}`,
                  results
                });
                return;
              }
              resolve({ ok: true, results });
            });
          } catch (err) {
            wd14Proc = null;
            resolve({
              ok: false,
              error: err instanceof Error ? err.message : String(err),
              results: []
            });
          }
        });
      } catch (err) {
        wd14Proc = null;
        return {
          ok: false,
          error: err instanceof Error ? err.message : String(err),
          results: []
        };
      }
    }
  );
  electron.ipcMain.handle("wd14:cancel", async () => {
    killWd14Process();
    return { ok: true };
  });
  electron.ipcMain.handle("fs:listImages", async (_event, dir) => {
    const entries = await promises.readdir(dir, { withFileTypes: true });
    const captionStems = /* @__PURE__ */ new Set();
    const imageEntries = [];
    for (const entry of entries) {
      if (!entry.isFile()) continue;
      const ext = path.extname(entry.name).toLowerCase();
      if (ext === ".txt") {
        captionStems.add(path.basename(entry.name, path.extname(entry.name)).toLowerCase());
        continue;
      }
      if (!IMAGE_EXTS.has(ext)) continue;
      imageEntries.push({ name: entry.name, path: path.join(dir, entry.name) });
    }
    const images = imageEntries.map(({ name, path: imagePath }) => {
      const stem = path.basename(name, path.extname(name)).toLowerCase();
      return { path: imagePath, name, hasCaption: captionStems.has(stem) };
    });
    images.sort((a, b) => a.name.localeCompare(b.name, void 0, { numeric: true }));
    return images;
  });
  electron.ipcMain.handle(
    "dataset:scanArBuckets",
    async (_event, opts) => {
      const script = path.join(trainerRoot(), "scan_ar_buckets.py");
      if (!fs.existsSync(script)) {
        return { ok: false, error: `Missing ${script}` };
      }
      const py = opts.pythonPath && opts.pythonPath.trim() || "python";
      const resList = (Array.isArray(opts.resolutions) ? opts.resolutions : []).map((n) => Math.round(Number(n))).filter((n) => Number.isFinite(n) && n > 0);
      const resolutionsArg = (resList.length ? resList : [1024]).join(",");
      const args = [
        script,
        "--folder",
        opts.folder,
        "--resolutions",
        resolutionsArg
      ];
      try {
        const { stdout, stderr } = await execFileAsync(py, args, {
          cwd: trainerRoot(),
          timeout: 3e5,
          windowsHide: true,
          encoding: "utf8",
          maxBuffer: 8 * 1024 * 1024
        });
        const text = `${stdout || ""}
${stderr || ""}`;
        const line = text.split(/\r?\n/).map((l) => l.trim()).filter(Boolean).reverse().find((l) => l.startsWith("{"));
        if (!line) {
          return { ok: false, error: "No JSON from scan_ar_buckets.py" };
        }
        const parsed = JSON.parse(line);
        if (!parsed.ok) {
          return { ok: false, error: parsed.error || "Scan failed" };
        }
        return {
          ok: true,
          imageCount: parsed.image_count ?? 0,
          forcedUpscale: parsed.forced_upscale ?? 0,
          countsOrdered: parsed.counts_ordered ?? []
        };
      } catch (err) {
        const e = err;
        return {
          ok: false,
          error: e.stderr || e.message || String(err)
        };
      }
    }
  );
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
    const merged = { ...current, ...settings };
    const uiGpuMode = normalizeUiGpuMode(merged);
    await saveSettings({
      ...merged,
      uiGpuMode,
      disableUiGpu: uiGpuMode === "software",
      // Window geometry is owned by main; never let renderer wipe it
      windowWidth: current.windowWidth,
      windowHeight: current.windowHeight,
      windowX: current.windowX,
      windowY: current.windowY,
      windowMaximized: current.windowMaximized
    });
    return true;
  });
  electron.ipcMain.handle("app:relaunch", async () => {
    electron.app.relaunch();
    electron.app.quit();
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
