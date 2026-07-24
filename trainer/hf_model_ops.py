#!/usr/bin/env python3
"""Check / download Hugging Face models for Captioer LoRA Train."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SIDECAR_NAME = ".captioer_model.json"


def sanitize_repo_id(repo_id: str) -> str:
    return repo_id.strip().replace("/", "__").replace("\\", "__")


def is_hf_repo_id(value: str) -> bool:
    """True for Hub ids like 'org/name'. '/' is allowed (Windows altsep must not reject it)."""
    s = value.strip()
    if not s:
        return False
    # Absolute / relative filesystem paths — not Hub ids
    if "\\" in s or s.startswith("/") or s.startswith("//"):
        return False
    if len(s) >= 2 and s[1] == ":":
        return False
    if s.startswith(".") or s.startswith("~"):
        return False
    parts = s.split("/")
    return len(parts) == 2 and all(parts) and not any(" " in p for p in parts)


def local_dir_for(download_path: str, repo_id: str) -> Path:
    return Path(download_path).expanduser().resolve() / sanitize_repo_id(repo_id)


def read_sidecar(directory: Path) -> dict[str, Any] | None:
    path = directory / SIDECAR_NAME
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("repo_id"):
            return data
    except Exception:
        return None
    return None


def write_sidecar(directory: Path, repo_id: str, revision: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"repo_id": repo_id, "revision": revision}
    (directory / SIDECAR_NAME).write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def download_is_incomplete(directory: Path) -> bool:
    """True if huggingface_hub left resume markers from an interrupted snapshot_download."""
    cache_dl = directory / ".cache" / "huggingface" / "download"
    if not cache_dl.is_dir():
        return False
    try:
        for p in cache_dl.rglob("*"):
            if not p.is_file():
                continue
            name = p.name
            if name.endswith(".lock") or name.endswith(".incomplete"):
                return True
    except OSError:
        return False
    return False


def has_weight_files(directory: Path) -> bool:
    """True if a non-trivial weight file exists outside .cache."""
    try:
        for p in directory.rglob("*"):
            if not p.is_file():
                continue
            if ".cache" in p.parts:
                continue
            name = p.name.lower()
            if name.endswith((".safetensors", ".bin", ".pt", ".ckpt")):
                try:
                    if p.stat().st_size > 1_000_000:
                        return True
                except OSError:
                    continue
    except OSError:
        return False
    return False


def dir_has_model_files(directory: Path) -> bool:
    """True only for a usable snapshot — not a partial interrupted download.

    Our downloads write `.captioer_model.json` only after success. `model_index.json`
    alone is not enough: HF often fetches it early while large weights are still
    incomplete (leaving `.lock` / `.incomplete` under `.cache`).
    """
    if not directory.is_dir():
        return False
    if download_is_incomplete(directory):
        return False
    if (directory / SIDECAR_NAME).is_file():
        return True
    has_index = (directory / "model_index.json").is_file()
    if not has_index:
        try:
            for child in directory.rglob("model_index.json"):
                if child.is_file() and ".cache" not in child.parts:
                    has_index = True
                    break
        except OSError:
            return False
    return bool(has_index and has_weight_files(directory))


def format_hf_error(err: BaseException, repo_id: str = "") -> str:
    msg = str(err).replace("\n", " ").strip()
    low = msg.lower()
    page = f"https://huggingface.co/{repo_id}" if repo_id else "the model page on Hugging Face"
    # Token present but HF account has not agreed / been approved for this gated repo
    if (
        "not in the authorized list" in low
        or ("access to model" in low and "restricted" in low)
        or type(err).__name__ == "GatedRepoError"
    ):
        return (
            f"Gated model access not granted for your Hugging Face account. "
            f"Open {page} while logged in, click Agree / request access, "
            f"then retry Download. Token is fine — the license gate is blocking files."
        )
    if "no hugging face token" in low:
        return (
            f"Hugging Face token required for gated model {repo_id or 'download'}. "
            f"Set it in LoRA Train Settings, accept access at {page}, then retry."
        )
    if any(k in low for k in ("gated", "403", "401", "cannot access gated", "unauthorized")):
        return (
            f"Gated model — set Hugging Face token in LoRA Train Settings, "
            f"login/accept access at {page}, then retry. ({msg[:240]})"
        )
    return msg[:500]


def remote_revision(repo_id: str, token: str | None) -> str:
    from huggingface_hub import HfApi

    api = HfApi(token=token or None)
    info = api.model_info(repo_id, token=token or None)
    sha = getattr(info, "sha", None) or ""
    if not sha:
        raise RuntimeError(f"No revision for {repo_id}")
    return str(sha)


def check_one(
    role: str,
    path_value: str,
    download_path: str,
    token: str | None,
) -> dict[str, Any]:
    raw = (path_value or "").strip()
    base: dict[str, Any] = {
        "role": role,
        "path": raw,
        "repoId": None,
        "status": "error",
        "localPath": None,
        "localRevision": None,
        "remoteRevision": None,
        "message": None,
    }
    if not raw:
        base["status"] = "missing"
        base["message"] = "Empty model path"
        return base

    try:
        if is_hf_repo_id(raw):
            repo_id = raw
            local_path = local_dir_for(download_path, repo_id)
            base["repoId"] = repo_id
            base["localPath"] = str(local_path)
            sidecar = read_sidecar(local_path) if local_path.is_dir() else None
            has_files = dir_has_model_files(local_path)
            if not has_files:
                base["status"] = "missing"
                if local_path.is_dir() and download_is_incomplete(local_path):
                    base["message"] = "Download incomplete — resume with Download"
                else:
                    base["message"] = "Not downloaded"
                return base
            local_rev = (sidecar or {}).get("revision") if sidecar else None
            base["localRevision"] = local_rev
            try:
                remote = remote_revision(repo_id, token)
                base["remoteRevision"] = remote
                if local_rev and local_rev != remote:
                    base["status"] = "updateAvailable"
                    base["message"] = "Newer revision on Hugging Face"
                else:
                    base["status"] = "ready"
                    base["message"] = "Up to date" if local_rev else "Local copy present"
            except Exception as err:
                base["status"] = "ready"
                base["message"] = f"Local copy present (remote check failed: {format_hf_error(err, repo_id)})"
            return base

        # Local filesystem path
        local_path = Path(raw).expanduser()
        try:
            local_path = local_path.resolve()
        except OSError:
            pass
        base["localPath"] = str(local_path)
        if not dir_has_model_files(local_path):
            base["status"] = "missing"
            base["message"] = "Local path missing or empty"
            return base

        sidecar = read_sidecar(local_path)
        if not sidecar:
            base["status"] = "local"
            base["message"] = "Local path (no update metadata)"
            return base

        repo_id = str(sidecar.get("repo_id") or "")
        local_rev = sidecar.get("revision")
        base["repoId"] = repo_id or None
        base["localRevision"] = local_rev
        if not repo_id:
            base["status"] = "local"
            base["message"] = "Local path (no repo id in sidecar)"
            return base
        try:
            remote = remote_revision(repo_id, token)
            base["remoteRevision"] = remote
            if local_rev and local_rev != remote:
                base["status"] = "updateAvailable"
                base["message"] = "Newer revision on Hugging Face"
            else:
                base["status"] = "ready"
                base["message"] = "Up to date"
        except Exception as err:
            base["status"] = "ready"
            base["message"] = f"Local copy present (remote check failed: {err})"
        return base
    except Exception as err:
        base["status"] = "error"
        base["message"] = str(err)
        return base


def cmd_check(args: argparse.Namespace) -> int:
    token = (args.token or "").strip() or None
    download_path = (args.download_path or "").strip()
    if not download_path:
        print(json.dumps({"ok": False, "error": "download_path required"}), flush=True)
        return 1
    try:
        targets = json.loads(args.targets)
    except json.JSONDecodeError as err:
        print(json.dumps({"ok": False, "error": f"Invalid targets JSON: {err}"}), flush=True)
        return 1
    if not isinstance(targets, list):
        print(json.dumps({"ok": False, "error": "targets must be a list"}), flush=True)
        return 1

    results = []
    for item in targets:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "train")
        path_value = str(item.get("path") or "")
        results.append(check_one(role, path_value, download_path, token))

    print(json.dumps({"ok": True, "results": results}, ensure_ascii=False), flush=True)
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.utils import tqdm as hf_tqdm

    repo_id = (args.repo_id or "").strip()
    download_path = (args.download_path or "").strip()
    token = (args.token or "").strip() or None
    if not repo_id or not is_hf_repo_id(repo_id):
        print(f"CAPTIOER_MODEL_ERROR message=Invalid repo id: {repo_id}", flush=True)
        return 1
    if not download_path:
        print("CAPTIOER_MODEL_ERROR message=download_path required", flush=True)
        return 1

    local_dir = local_dir_for(download_path, repo_id)
    local_dir.mkdir(parents=True, exist_ok=True)

    repo_total = 0
    try:
        info = HfApi(token=token).model_info(repo_id, token=token, files_metadata=True)
        for sib in getattr(info, "siblings", None) or []:
            size = getattr(sib, "size", None)
            if isinstance(size, int) and size > 0:
                repo_total += size
    except Exception:
        repo_total = 0

    # Track multi-file snapshot progress across parallel byte bars.
    # Non-TTY spawns create bars with disable=True (self.n never advances); files
    # also download in parallel, so aggregate via active dict + nArg fallback.
    prog_state: dict[str, Any] = {
        "completed": 0,
        "active": {},  # tqdm id -> bytes so far
        "finished": set(),
        "last_key": None,
        "last_emit_at": 0.0,
    }

    def emit_progress(done: int, total: int, pct: int) -> None:
        parts = [f"CAPTIOER_MODEL_PROGRESS repo={repo_id} pct={pct}"]
        if done >= 0:
            parts.append(f"done={done}")
        if total > 0:
            parts.append(f"total={total}")
        print(" ".join(parts), flush=True)

    class ProgressTqdm(hf_tqdm):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[no-untyped-def]
            # Non-TTY spawns set disable=True; disabled bars do not advance self.n.
            kwargs["disable"] = False
            super().__init__(*args, **kwargs)

        def update(self, n: float | int = 1) -> None | bool:  # type: ignore[override]
            result = super().update(n)
            total = int(getattr(self, "total", None) or 0)
            n_inc = int(n or 0)
            n_done = int(getattr(self, "n", 0) or 0)
            unit = getattr(self, "unit", None)
            tid = id(self)

            # Outer "Fetching N files" counter is not byte progress.
            is_file_counter = unit == "it" or (
                isinstance(getattr(self, "desc", None), str)
                and str(self.desc).lower().startswith("fetching")
            )
            if is_file_counter:
                return result

            active: dict[int, int] = prog_state["active"]
            finished: set[int] = prog_state["finished"]

            # If disable still ate self.n, accumulate nArg manually.
            if n_inc > 0 and n_done == 0:
                n_done = int(active.get(tid, 0)) + n_inc
            elif n_done == 0 and n_inc == 0:
                n_done = int(active.get(tid, 0))

            if tid in finished:
                display_done = int(prog_state["completed"]) + sum(active.values())
            elif total > 0 and n_done >= total:
                active.pop(tid, None)
                if tid not in finished:
                    finished.add(tid)
                    prog_state["completed"] = int(prog_state["completed"]) + total
                display_done = int(prog_state["completed"]) + sum(active.values())
            else:
                active[tid] = n_done
                display_done = int(prog_state["completed"]) + sum(active.values())

            display_total = repo_total if repo_total > 0 else total
            if display_total > 0:
                pct = int(min(99, max(0, (100 * display_done) // display_total)))
            elif total > 0:
                pct = int(min(99, max(0, (100 * n_done) // total)))
            else:
                pct = 0

            key = (pct, display_done, display_total)
            now = __import__("time").monotonic()
            # Throttle emits: on pct/done change, or at least every 0.5s while moving.
            if key != prog_state["last_key"] or (
                display_done > 0 and now - float(prog_state["last_emit_at"]) >= 0.5
            ):
                prog_state["last_key"] = key
                prog_state["last_emit_at"] = now
                emit_progress(display_done, display_total, pct)
            return result

    emit_progress(0, repo_total, 0)
    try:
        if not token:
            print(
                "CAPTIOER_MODEL_ERROR message="
                + format_hf_error(
                    RuntimeError("No Hugging Face token provided"),
                    repo_id,
                ),
                flush=True,
            )
            return 1
        revision = remote_revision(repo_id, token)
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            revision=revision,
            token=token,
            tqdm_class=ProgressTqdm,
        )
        write_sidecar(local_dir, repo_id, revision)
        print(
            f"CAPTIOER_MODEL_DONE repo={repo_id} path={local_dir} revision={revision}",
            flush=True,
        )
        return 0
    except KeyboardInterrupt:
        print(f"CAPTIOER_MODEL_ERROR message=Cancelled repo={repo_id}", flush=True)
        return 130
    except Exception as err:
        print(
            f"CAPTIOER_MODEL_ERROR message={format_hf_error(err, repo_id)}",
            flush=True,
        )
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Captioer HF model check/download")
    sub = parser.add_subparsers(dest="command", required=True)

    p_check = sub.add_parser("check")
    p_check.add_argument("--download-path", required=True)
    p_check.add_argument("--token", default="")
    p_check.add_argument("--targets", required=True, help="JSON list of {role,path}")

    p_dl = sub.add_parser("download")
    p_dl.add_argument("--download-path", required=True)
    p_dl.add_argument("--token", default="")
    p_dl.add_argument("--repo-id", required=True)

    args = parser.parse_args()
    if args.command == "check":
        return cmd_check(args)
    if args.command == "download":
        return cmd_download(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
