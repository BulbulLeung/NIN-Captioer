import { app } from 'electron'
import { createWriteStream, existsSync, readdirSync, statSync } from 'fs'
import { mkdir, readFile, rename, rm, writeFile } from 'fs/promises'
import { join, dirname } from 'path'
import { spawn, execFile, type ChildProcessWithoutNullStreams } from 'child_process'
import { get } from 'https'
import { pipeline } from 'stream/promises'
import { promisify } from 'util'

const execFileAsync = promisify(execFile)

export type PythonProbeStatus = 'ready' | 'missingPython' | 'missingPackages' | 'error'

export interface PythonProbeResult {
  status: PythonProbeStatus
  message: string
  pythonPath?: string
  version?: string
  cuda?: boolean
  krea?: boolean
  missing?: string[]
}

export interface PythonInstallProgress {
  stage: string
  message: string
  pct: number
}

type ProgressFn = (p: PythonInstallProgress) => void

let installProc: ChildProcessWithoutNullStreams | null = null
let installCancelled = false

export function defaultPythonInstallPath(): string {
  return join(app.getPath('userData'), 'python')
}

export function resolvePythonInstallPath(raw?: string): string {
  const trimmed = (raw || '').trim()
  return trimmed || defaultPythonInstallPath()
}

function venvPythonPath(installRoot: string): string {
  return process.platform === 'win32'
    ? join(installRoot, 'venv', 'Scripts', 'python.exe')
    : join(installRoot, 'venv', 'bin', 'python')
}

function uvBinaryPath(installRoot: string): string {
  return process.platform === 'win32'
    ? join(installRoot, 'tools', 'uv.exe')
    : join(installRoot, 'tools', 'uv')
}

function uvDownloadUrl(): string {
  if (process.platform === 'win32') {
    return 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip'
  }
  if (process.platform === 'darwin') {
    const arch = process.arch === 'arm64' ? 'aarch64' : 'x86_64'
    return `https://github.com/astral-sh/uv/releases/latest/download/uv-${arch}-apple-darwin.tar.gz`
  }
  const arch = process.arch === 'arm64' ? 'aarch64' : 'x86_64'
  return `https://github.com/astral-sh/uv/releases/latest/download/uv-${arch}-unknown-linux-gnu.tar.gz`
}

function httpsGetFollow(url: string, redirects = 0): Promise<import('http').IncomingMessage> {
  return new Promise((resolve, reject) => {
    get(url, (res) => {
      if (
        res.statusCode &&
        res.statusCode >= 300 &&
        res.statusCode < 400 &&
        res.headers.location &&
        redirects < 8
      ) {
        res.resume()
        resolve(httpsGetFollow(res.headers.location, redirects + 1))
        return
      }
      if (!res.statusCode || res.statusCode >= 400) {
        reject(new Error(`HTTP ${res.statusCode} for ${url}`))
        res.resume()
        return
      }
      resolve(res)
    }).on('error', reject)
  })
}

async function downloadFile(url: string, dest: string): Promise<void> {
  await mkdir(dirname(dest), { recursive: true })
  const res = await httpsGetFollow(url)
  await pipeline(res, createWriteStream(dest))
}

async function extractArchive(archivePath: string, destDir: string): Promise<void> {
  await mkdir(destDir, { recursive: true })
  if (archivePath.endsWith('.zip')) {
    if (process.platform === 'win32') {
      await execFileAsync(
        'powershell.exe',
        [
          '-NoProfile',
          '-Command',
          `Expand-Archive -LiteralPath '${archivePath.replace(/'/g, "''")}' -DestinationPath '${destDir.replace(/'/g, "''")}' -Force`
        ],
        { windowsHide: true, timeout: 120000 }
      )
      return
    }
    await execFileAsync('unzip', ['-o', archivePath, '-d', destDir], { timeout: 120000 })
    return
  }
  await execFileAsync('tar', ['-xzf', archivePath, '-C', destDir], { timeout: 120000 })
}

function runSpawn(
  command: string,
  args: string[],
  opts?: { cwd?: string; env?: NodeJS.ProcessEnv }
): Promise<{ code: number; stdout: string; stderr: string }> {
  return new Promise((resolve, reject) => {
    if (installCancelled) {
      reject(new Error('Cancelled'))
      return
    }
    const child = spawn(command, args, {
      cwd: opts?.cwd,
      env: opts?.env ?? process.env,
      windowsHide: true
    })
    installProc = child
    let stdout = ''
    let stderr = ''
    child.stdout.on('data', (c: Buffer) => {
      stdout += c.toString('utf8')
    })
    child.stderr.on('data', (c: Buffer) => {
      stderr += c.toString('utf8')
    })
    child.on('error', (err) => {
      if (installProc === child) installProc = null
      reject(err)
    })
    child.on('close', (code) => {
      if (installProc === child) installProc = null
      if (installCancelled) {
        reject(new Error('Cancelled'))
        return
      }
      resolve({ code: code ?? 1, stdout, stderr })
    })
  })
}

export function cancelPythonInstall(): { ok: boolean } {
  installCancelled = true
  if (installProc && !installProc.killed) {
    try {
      installProc.kill()
    } catch {
      // ignore
    }
  }
  installProc = null
  return { ok: true }
}

export function pythonInstallRunning(): boolean {
  return Boolean(installProc && !installProc.killed)
}

export async function probePython(pythonPath?: string): Promise<PythonProbeResult> {
  const py = (pythonPath && pythonPath.trim()) || 'python'
  const code = [
    'import importlib, sys',
    'ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"',
    'mods = ["torch", "diffusers", "peft", "safetensors", "PIL", "onnxruntime", "numpy", "huggingface_hub"]',
    'missing = []',
    'for m in mods:',
    '  try:',
    '    importlib.import_module(m)',
    '  except Exception:',
    '    missing.append(m)',
    'ok_krea = False',
    'try:',
    '  from diffusers import Krea2Pipeline  # noqa: F401',
    '  ok_krea = True',
    'except Exception:',
    '  missing.append("diffusers.Krea2Pipeline")',
    'cuda = False',
    'try:',
    '  import torch',
    '  cuda = bool(torch.cuda.is_available())',
    'except Exception:',
    '  pass',
    'print("VER:" + ver)',
    'print("MISSING:" + ",".join(missing))',
    'print("CUDA:" + ("1" if cuda else "0"))',
    'print("KREA:" + ("1" if ok_krea else "0"))',
    'ok_triton = False',
    'try:',
    '  import triton  # noqa: F401',
    '  ok_triton = True',
    'except Exception:',
    '  pass',
    'print("TRITON:" + ("1" if ok_triton else "0"))'
  ].join('\n')

  try {
    const { stdout, stderr } = await execFileAsync(py, ['-c', code], {
      timeout: 60000,
      windowsHide: true,
      encoding: 'utf8',
      maxBuffer: 2 * 1024 * 1024
    })
    const out = `${stdout}\n${stderr}`
    const ver = out.split(/\r?\n/).find((l) => l.startsWith('VER:'))?.slice(4) || ''
    const missLine = out.split(/\r?\n/).find((l) => l.startsWith('MISSING:'))
    const missing = (missLine?.slice('MISSING:'.length) || '')
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean)
    const cuda = /CUDA:1/.test(out)
    const krea = /KREA:1/.test(out)
    const triton = /TRITON:1/.test(out)
    if (missing.length === 0) {
      return {
        status: 'ready',
        message: `OK (${py} ${ver})${cuda ? ' · CUDA' : ' · no CUDA'}${
          krea ? ' · Krea2' : ''
        }${triton ? ' · Triton' : process.platform === 'win32' ? ' · no Triton' : ''}`,
        pythonPath: py,
        version: ver,
        cuda,
        krea,
        missing: []
      }
    }
    return {
      status: 'missingPackages',
      message: `Missing: ${missing.join(', ')}`,
      pythonPath: py,
      version: ver,
      cuda,
      krea,
      missing
    }
  } catch (err: unknown) {
    const e = err as { message?: string; code?: string }
    const msg = e.message || String(err)
    if (e.code === 'ENOENT' || /not found|ENOENT|is not recognized/i.test(msg)) {
      return {
        status: 'missingPython',
        message: `Python not found (${py})`,
        pythonPath: py
      }
    }
    return {
      status: 'error',
      message: `Failed to probe Python (${py}): ${msg}`,
      pythonPath: py
    }
  }
}

async function ensureUv(installRoot: string, onProgress: ProgressFn): Promise<string> {
  const uvPath = uvBinaryPath(installRoot)
  if (existsSync(uvPath)) return uvPath
  onProgress({ stage: 'uv', message: 'Downloading uv…', pct: 5 })
  const toolsDir = join(installRoot, 'tools')
  await mkdir(toolsDir, { recursive: true })
  const archiveName = process.platform === 'win32' ? 'uv.zip' : 'uv.tar.gz'
  const archivePath = join(toolsDir, archiveName)
  await downloadFile(uvDownloadUrl(), archivePath)
  onProgress({ stage: 'uv', message: 'Extracting uv…', pct: 12 })
  const extractDir = join(toolsDir, 'extract')
  await rm(extractDir, { recursive: true, force: true })
  await extractArchive(archivePath, extractDir)
  const walk = (dir: string): string | null => {
    for (const name of readdirSync(dir)) {
      const p = join(dir, name)
      const st = statSync(p)
      if (st.isDirectory()) {
        const found = walk(p)
        if (found) return found
      } else if (name === 'uv.exe' || name === 'uv') {
        return p
      }
    }
    return null
  }
  const found = walk(extractDir)
  if (!found) throw new Error('uv binary not found in archive')
  await rename(found, uvPath)
  await rm(extractDir, { recursive: true, force: true })
  await rm(archivePath, { force: true })
  return uvPath
}

async function writeRequirementsNoTorch(src: string, dest: string): Promise<void> {
  const text = await readFile(src, 'utf8')
  const filtered = text
    .split(/\r?\n/)
    .filter((line) => {
      const t = line.trim()
      if (!t || t.startsWith('#')) return true
      // Torch is installed separately (CUDA index). flash-attn is a separate Windows wheel.
      return !/^torch\b/i.test(t) && !/^flash-attn\b/i.test(t) && !/^flash_attn\b/i.test(t)
    })
    .join('\n')
  await writeFile(dest, filtered + '\n', 'utf8')
}

/** Map installed torch major.minor → triton-windows version ceiling (exclusive). */
function tritonWindowsSpecForTorch(torchVersion: string): string {
  const m = torchVersion.match(/(\d+)\.(\d+)/)
  const major = m ? Number(m[1]) : 2
  const minor = m ? Number(m[2]) : 9
  // https://github.com/triton-lang/triton-windows — PyTorch ↔ Triton pairing
  if (major === 2 && minor <= 5) return 'triton-windows>=3.1,<3.2'
  if (major === 2 && minor === 6) return 'triton-windows>=3.2,<3.3'
  if (major === 2 && minor === 7) return 'triton-windows>=3.3,<3.4'
  if (major === 2 && minor === 8) return 'triton-windows>=3.4,<3.5'
  if (major === 2 && minor === 9) return 'triton-windows>=3.5,<3.6'
  if (major === 2 && minor <= 11) return 'triton-windows>=3.6,<3.7'
  return 'triton-windows>=3.7,<3.8'
}

/** torchao cpp ops are tightly coupled to torch minor — pin known-good pairs. */
function torchaoSpecForTorch(torchVersion: string): string {
  const m = torchVersion.match(/(\d+)\.(\d+)(?:\.(\d+))?/)
  const major = m ? Number(m[1]) : 2
  const minor = m ? Number(m[2]) : 9
  const patch = m && m[3] != null ? Number(m[3]) : 0
  // https://github.com/pytorch/ao/issues/2919
  if (major === 2 && minor === 6) return 'torchao>=0.9.0,<0.10.0'
  if (major === 2 && minor === 9 && patch >= 1) return 'torchao==0.15.0'
  if (major === 2 && minor === 9) return 'torchao==0.14.1'
  if (major === 2 && minor === 10) return 'torchao==0.16.0'
  return 'torchao>=0.15.0,<0.17.0'
}

async function readTorchVersion(py: string): Promise<string> {
  try {
    const { stdout } = await execFileAsync(
      py,
      ['-c', 'import torch; print(torch.__version__)'],
      { timeout: 60000, windowsHide: true, encoding: 'utf8' }
    )
    return (stdout || '').trim()
  } catch {
    return ''
  }
}

async function readTorchCuda(py: string): Promise<string> {
  try {
    const { stdout } = await execFileAsync(
      py,
      ['-c', 'import torch; print(torch.version.cuda or "")'],
      { timeout: 60000, windowsHide: true, encoding: 'utf8' }
    )
    return (stdout || '').trim()
  } catch {
    return ''
  }
}

/**
 * Official `triton` wheels are Linux-only. On Windows install `triton-windows`
 * (PyPI) plus CUDA nvcc/runtime packages required by that build.
 */
async function installTritonWindows(opts: {
  uv: string
  py: string
  installRoot: string
  onProgress: ProgressFn
}): Promise<{ ok: boolean; message: string }> {
  const { uv, py, installRoot, onProgress } = opts
  onProgress({
    stage: 'triton',
    message: 'Installing triton-windows (GPU kernels)…',
    pct: 82
  })

  // Drop any leftover Linux `triton` stub that would conflict.
  await runSpawn(uv, ['pip', 'uninstall', '-y', '--python', py, 'triton'], {
    cwd: installRoot
  })

  let r = await runSpawn(
    uv,
    [
      'pip',
      'install',
      '--python',
      py,
      'setuptools',
      'wheel',
      'nvidia-cuda-nvcc-cu12',
      'nvidia-cuda-runtime-cu12'
    ],
    { cwd: installRoot }
  )
  if (r.code !== 0) {
    return {
      ok: false,
      message: `triton Windows deps (setuptools/CUDA) failed: ${r.stderr || r.stdout}`
    }
  }

  const torchVer = await readTorchVersion(py)
  const spec = tritonWindowsSpecForTorch(torchVer || '2.9')
  onProgress({
    stage: 'triton',
    message: `Installing ${spec} (torch ${torchVer || 'unknown'})…`,
    pct: 85
  })
  r = await runSpawn(uv, ['pip', 'install', '--python', py, '-U', spec], {
    cwd: installRoot
  })
  if (r.code !== 0) {
    return {
      ok: false,
      message: `triton-windows install failed: ${r.stderr || r.stdout}`
    }
  }

  // Verify import name is still `triton` (provided by triton-windows package).
  try {
    await execFileAsync(py, ['-c', 'import triton; print(triton.__version__)'], {
      timeout: 60000,
      windowsHide: true,
      encoding: 'utf8'
    })
  } catch (err) {
    const detail = err instanceof Error ? err.message : String(err)
    return { ok: false, message: `triton import failed after install: ${detail}` }
  }

  return { ok: true, message: `triton-windows OK (${spec})` }
}

/**
 * Soft-install flash-attn (FA2) Windows wheel.
 * Note: `bdsqlsz/flash-attention-wheels` is not publicly reachable (404/401).
 * We use PozzettiAndrea/cuda-wheels builds matched to torch+CUDA+CPython.
 * @see https://github.com/PozzettiAndrea/cuda-wheels/releases/tag/flash_attn-latest
 */
function flashAttnWindowsWheelUrl(
  torchVersion: string,
  cudaVersion: string,
  pythonVersion: string
): string | null {
  const tm = torchVersion.match(/(\d+)\.(\d+)/)
  const cm = cudaVersion.match(/(\d+)\.(\d+)/)
  const pm = pythonVersion.match(/(\d+)\.(\d+)/)
  if (!tm || !pm) return null
  const tMinor = Number(tm[2])
  const pyTag = `cp${pm[1]}${pm[2]}`
  // Torch 2.9.x + cu128 is the Captioer default; CUDA minor in torch tag is authoritative.
  const cuMajor = cm ? Number(cm[1]) : 12
  const cuMinor = cm ? Number(cm[2]) : 8
  let cuTag = 'cu128'
  if (cuMajor === 13) cuTag = 'cu130'
  else if (cuMinor <= 6) cuTag = 'cu126'
  else if (cuMinor === 9) cuTag = 'cu129'
  else cuTag = 'cu128'
  if (tMinor !== 9) {
    // Only auto-wire the known torch2.9 matrix for now.
    return null
  }
  const name = `flash_attn-2.8.3+${cuTag}torch2.9-${pyTag}-${pyTag}-win_amd64.whl`
  const encoded = name.replace(/\+/g, '%2B')
  return `https://github.com/PozzettiAndrea/cuda-wheels/releases/download/flash_attn-latest/${encoded}`
}

async function readPythonVersion(py: string): Promise<string> {
  try {
    const { stdout } = await execFileAsync(
      py,
      ['-c', 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'],
      { timeout: 30000, windowsHide: true, encoding: 'utf8' }
    )
    return (stdout || '').trim()
  } catch {
    return ''
  }
}

async function installFlashAttn(opts: {
  uv: string
  py: string
  installRoot: string
  onProgress: ProgressFn
}): Promise<{ ok: boolean; message: string }> {
  const { uv, py, installRoot, onProgress } = opts
  onProgress({
    stage: 'flash',
    message: 'Installing flash-attn (FA2 wheel)…',
    pct: 86
  })

  // Soft dep used by some flash_attn import paths.
  await runSpawn(uv, ['pip', 'install', '--python', py, 'einops', 'packaging', 'ninja'], {
    cwd: installRoot
  })

  if (process.platform === 'win32') {
    const torchVer = await readTorchVersion(py)
    const cudaVer = await readTorchCuda(py)
    const pyVer = await readPythonVersion(py)
    const wheelUrl = flashAttnWindowsWheelUrl(torchVer, cudaVer, pyVer || '3.11')
    if (!wheelUrl) {
      return {
        ok: false,
        message: `no flash-attn Windows wheel for torch ${torchVer || '?'} cuda ${cudaVer || '?'} py ${pyVer || '?'}`
      }
    }
    onProgress({
      stage: 'flash',
      message: `Downloading flash-attn wheel (torch ${torchVer})…`,
      pct: 86
    })
    const wheelName = decodeURIComponent(wheelUrl.split('/').pop() || 'flash_attn.whl')
    const wheelPath = join(installRoot, wheelName)
    try {
      await downloadFile(wheelUrl, wheelPath)
    } catch (err) {
      const detail = err instanceof Error ? err.message : String(err)
      return { ok: false, message: `flash-attn wheel download failed: ${detail}` }
    }
    const r = await runSpawn(
      uv,
      ['pip', 'install', '--python', py, '--force-reinstall', '--no-deps', wheelPath],
      { cwd: installRoot }
    )
    await rm(wheelPath, { force: true }).catch(() => undefined)
    if (r.code !== 0) {
      return {
        ok: false,
        message: `flash-attn wheel install failed: ${r.stderr || r.stdout}`
      }
    }
  } else {
    // Linux: try PyPI (often needs CUDA toolkit to build; soft-fail).
    const r = await runSpawn(
      uv,
      ['pip', 'install', '--python', py, 'flash-attn', '--no-build-isolation'],
      { cwd: installRoot }
    )
    if (r.code !== 0) {
      return {
        ok: false,
        message: `flash-attn install failed: ${r.stderr || r.stdout}`
      }
    }
  }

  try {
    const { stdout } = await execFileAsync(
      py,
      [
        '-c',
        'from flash_attn import flash_attn_func; import flash_attn; print(getattr(flash_attn, "__version__", "ok"))'
      ],
      { timeout: 120000, windowsHide: true, encoding: 'utf8' }
    )
    const ver = (stdout || '').trim() || 'ok'
    return { ok: true, message: `flash-attn ${ver} OK` }
  } catch (err) {
    const detail = err instanceof Error ? err.message : String(err)
    return { ok: false, message: `flash-attn import failed: ${detail}` }
  }
}

export async function installPythonEnv(opts: {
  installPath?: string
  trainerRoot: string
  onProgress: ProgressFn
}): Promise<{ ok: boolean; pythonPath?: string; message: string }> {
  installCancelled = false
  const installRoot = resolvePythonInstallPath(opts.installPath)
  const onProgress = opts.onProgress
  try {
    await mkdir(installRoot, { recursive: true })
    const uv = await ensureUv(installRoot, onProgress)

    onProgress({ stage: 'python', message: 'Installing Python 3.11…', pct: 20 })
    let r = await runSpawn(uv, ['python', 'install', '3.11'], { cwd: installRoot })
    if (r.code !== 0) {
      throw new Error(`uv python install failed: ${r.stderr || r.stdout}`)
    }

    const venvDir = join(installRoot, 'venv')
    onProgress({ stage: 'venv', message: 'Creating venv…', pct: 30 })
    r = await runSpawn(uv, ['venv', venvDir, '--python', '3.11'], { cwd: installRoot })
    if (r.code !== 0 && !existsSync(venvPythonPath(installRoot))) {
      throw new Error(`uv venv failed: ${r.stderr || r.stdout}`)
    }

    const py = venvPythonPath(installRoot)
    if (!existsSync(py)) {
      throw new Error(`venv python missing: ${py}`)
    }

    onProgress({ stage: 'torch', message: 'Installing CUDA torch 2.9.1 (cu128)…', pct: 40 })
    r = await runSpawn(
      uv,
      [
        'pip',
        'install',
        '--python',
        py,
        'torch==2.9.1',
        'torchvision==0.24.1',
        '--index-url',
        'https://download.pytorch.org/whl/cu128'
      ],
      { cwd: installRoot }
    )
    let torchMode = 'CUDA cu128 (torch 2.9.1)'
    if (r.code !== 0) {
      onProgress({ stage: 'torch', message: 'CUDA torch failed; installing CPU torch…', pct: 45 })
      r = await runSpawn(
        uv,
        [
          'pip',
          'install',
          '--python',
          py,
          'torch==2.9.1',
          'torchvision==0.24.1',
          '--index-url',
          'https://download.pytorch.org/whl/cpu'
        ],
        { cwd: installRoot }
      )
      if (r.code !== 0) {
        throw new Error(`torch install failed: ${r.stderr || r.stdout}`)
      }
      torchMode = 'CPU (torch 2.9.1)'
    }

    const reqTrain = join(opts.trainerRoot, 'requirements.txt')
    const reqWd14 = join(opts.trainerRoot, 'requirements-wd14.txt')
    const tmpReq = join(installRoot, 'requirements-no-torch.txt')
    if (!existsSync(reqTrain)) throw new Error(`Missing ${reqTrain}`)
    await writeRequirementsNoTorch(reqTrain, tmpReq)

    onProgress({ stage: 'reqs', message: 'Installing training requirements…', pct: 60 })
    r = await runSpawn(uv, ['pip', 'install', '--python', py, '-r', tmpReq], {
      cwd: installRoot
    })
    if (r.code !== 0) {
      throw new Error(`requirements.txt install failed: ${r.stderr || r.stdout}`)
    }

    // Keep torchao cpp ops matched to torch minor (mismatched versions skip kernels).
    const torchVer = await readTorchVersion(py)
    const torchaoSpec = torchaoSpecForTorch(torchVer || '2.9.1')
    onProgress({
      stage: 'torchao',
      message: `Pinning ${torchaoSpec} for torch ${torchVer || 'unknown'}…`,
      pct: 72
    })
    r = await runSpawn(uv, ['pip', 'install', '--python', py, torchaoSpec], {
      cwd: installRoot
    })
    if (r.code !== 0) {
      throw new Error(`torchao pin failed: ${r.stderr || r.stdout}`)
    }

    if (existsSync(reqWd14)) {
      onProgress({ stage: 'wd14', message: 'Installing WD14 requirements…', pct: 75 })
      r = await runSpawn(uv, ['pip', 'install', '--python', py, '-r', reqWd14], {
        cwd: installRoot
      })
      if (r.code !== 0) {
        throw new Error(`requirements-wd14.txt install failed: ${r.stderr || r.stdout}`)
      }
    }

    let tritonNote = ''
    let flashNote = ''
    if (process.platform === 'win32' && torchMode.startsWith('CUDA')) {
      const tritonResult = await installTritonWindows({ uv, py, installRoot, onProgress })
      if (!tritonResult.ok) {
        // Soft-fail: env still usable, but torchao Quantize kernels will be slow.
        tritonNote = ` Triton warning: ${tritonResult.message}`
        onProgress({
          stage: 'triton',
          message: `triton-windows skipped: ${tritonResult.message}`,
          pct: 86
        })
      } else {
        tritonNote = ` ${tritonResult.message}.`
      }
      const flashResult = await installFlashAttn({ uv, py, installRoot, onProgress })
      if (!flashResult.ok) {
        flashNote = ` flash-attn warning: ${flashResult.message}`
        onProgress({
          stage: 'flash',
          message: `flash-attn skipped: ${flashResult.message}`,
          pct: 86
        })
      } else {
        flashNote = ` ${flashResult.message}.`
      }
    } else if (process.platform === 'win32') {
      tritonNote = ' Triton skipped (CPU torch).'
    } else if (torchMode.startsWith('CUDA')) {
      const flashResult = await installFlashAttn({ uv, py, installRoot, onProgress })
      if (!flashResult.ok) {
        flashNote = ` flash-attn warning: ${flashResult.message}`
      } else {
        flashNote = ` ${flashResult.message}.`
      }
    }

    onProgress({ stage: 'krea', message: 'Checking Krea2Pipeline…', pct: 88 })
    const probe = await probePython(py)
    if (probe.missing?.includes('diffusers.Krea2Pipeline')) {
      onProgress({ stage: 'krea', message: 'Installing latest diffusers (git)…', pct: 90 })
      r = await runSpawn(
        uv,
        ['pip', 'install', '--python', py, 'git+https://github.com/huggingface/diffusers.git'],
        { cwd: installRoot }
      )
      if (r.code !== 0) {
        throw new Error(`diffusers git install failed: ${r.stderr || r.stdout}`)
      }
    }

    const finalProbe = await probePython(py)
    onProgress({ stage: 'done', message: 'Install complete', pct: 100 })
    return {
      ok: finalProbe.status === 'ready' || finalProbe.status === 'missingPackages',
      pythonPath: py,
      message:
        finalProbe.status === 'ready'
          ? `Installed (${torchMode}).${tritonNote}${flashNote} ${finalProbe.message}`
          : `Installed (${torchMode}) with warnings:${tritonNote}${flashNote} ${finalProbe.message}`
    }
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err)
    return { ok: false, message }
  } finally {
    installProc = null
    installCancelled = false
  }
}
