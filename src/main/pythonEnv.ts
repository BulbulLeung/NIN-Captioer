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
    'print("KREA:" + ("1" if ok_krea else "0"))'
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
    if (missing.length === 0) {
      return {
        status: 'ready',
        message: `OK (${py} ${ver})${cuda ? ' · CUDA' : ' · no CUDA'}${krea ? ' · Krea2' : ''}`,
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
      return !/^torch\b/i.test(t)
    })
    .join('\n')
  await writeFile(dest, filtered + '\n', 'utf8')
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

    onProgress({ stage: 'torch', message: 'Installing CUDA torch…', pct: 40 })
    r = await runSpawn(
      uv,
      [
        'pip',
        'install',
        '--python',
        py,
        'torch',
        'torchvision',
        '--index-url',
        'https://download.pytorch.org/whl/cu124'
      ],
      { cwd: installRoot }
    )
    let torchMode = 'CUDA cu124'
    if (r.code !== 0) {
      onProgress({ stage: 'torch', message: 'CUDA torch failed; installing CPU torch…', pct: 45 })
      r = await runSpawn(uv, ['pip', 'install', '--python', py, 'torch', 'torchvision'], {
        cwd: installRoot
      })
      if (r.code !== 0) {
        throw new Error(`torch install failed: ${r.stderr || r.stdout}`)
      }
      torchMode = 'CPU'
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

    if (existsSync(reqWd14)) {
      onProgress({ stage: 'wd14', message: 'Installing WD14 requirements…', pct: 75 })
      r = await runSpawn(uv, ['pip', 'install', '--python', py, '-r', reqWd14], {
        cwd: installRoot
      })
      if (r.code !== 0) {
        throw new Error(`requirements-wd14.txt install failed: ${r.stderr || r.stdout}`)
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
          ? `Installed (${torchMode}). ${finalProbe.message}`
          : `Installed (${torchMode}) with warnings: ${finalProbe.message}`
    }
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err)
    return { ok: false, message }
  } finally {
    installProc = null
    installCancelled = false
  }
}
