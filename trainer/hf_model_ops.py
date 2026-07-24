#!/usr/bin/env python3
"""Check / download Hugging Face models for Captioer LoRA Train."""

from __future__ import annotations

import argparse
import json
import os
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


def dir_has_model_files(directory: Path) -> bool:
    """True only for a usable snapshot (sidecar or diffusers model_index.json).

    Partial / failed downloads often leave .cache crumbs; those must not count as ready.
    """
    if not directory.is_dir():
        return False
    if (directory / SIDECAR_NAME).is_file():
        return True
    if (directory / "model_index.json").is_file():
        return True
    try:
        for child in directory.rglob("model_index.json"):
            if child.is_file():
                return True
    except OSError:
        return False
    return False


def format_hf_error(err: BaseException, repo_id: str = "") -> str:
    msg = str(err).replace("\n", " ").strip()
    low = msg.lower()
    if any(k in low for k in ("gated", "403", "401", "cannot access gated", "unauthorized")):
        page = f"https://huggingface.co/{repo_id}" if repo_id else "the model page on Hugging Face"
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
    from huggingface_hub import snapshot_download
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

    last_pct = {"v": -1}

    class ProgressTqdm(hf_tqdm):  # type: ignore[misc, valid-type]
        def update(self, n: float | int = 1) -> None | bool:  # type: ignore[override]
            result = super().update(n)
            total = getattr(self, "total", None) or 0
            n_done = getattr(self, "n", 0) or 0
            if total and total > 0:
                pct = int(min(99, max(0, (100 * n_done) // total)))
                if pct != last_pct["v"]:
                    last_pct["v"] = pct
                    print(
                        f"CAPTIOER_MODEL_PROGRESS repo={repo_id} pct={pct}",
                        flush=True,
                    )
            return result

    print(f"CAPTIOER_MODEL_PROGRESS repo={repo_id} pct=0", flush=True)
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
