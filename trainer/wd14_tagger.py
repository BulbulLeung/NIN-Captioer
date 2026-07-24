#!/usr/bin/env python3
"""WD14 ONNX tagger for Captioer Auto Caption / reCaption."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import sys
from pathlib import Path
from typing import Any

MODEL_FILENAME = "model.onnx"
LABEL_FILENAME = "selected_tags.csv"
SIDECAR_NAME = ".captioer_model.json"


def sanitize_repo_id(repo_id: str) -> str:
    return repo_id.strip().replace("/", "__").replace("\\", "__")


def is_hf_repo_id(value: str) -> bool:
    s = value.strip()
    if not s:
        return False
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


def write_sidecar(directory: Path, repo_id: str, revision: str) -> None:
    path = directory / SIDECAR_NAME
    path.write_text(
        json.dumps({"repo_id": repo_id, "revision": revision}, indent=2),
        encoding="utf-8",
    )


def model_ready(model_dir: Path) -> bool:
    return (model_dir / MODEL_FILENAME).is_file() and (model_dir / LABEL_FILENAME).is_file()


def emit_error(message: str) -> None:
    print(f"CAPTIOER_TAG_ERROR message={message}", flush=True)


def ensure_model(download_path: str, repo_id: str, token: str | None) -> Path:
    from huggingface_hub import snapshot_download

    if not is_hf_repo_id(repo_id):
        raise RuntimeError(f"Invalid repo id: {repo_id}")
    local_dir = local_dir_for(download_path, repo_id)
    if model_ready(local_dir):
        print(
            f"CAPTIOER_TAG_MODEL path={local_dir} status=ready",
            flush=True,
        )
        return local_dir

    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"CAPTIOER_TAG_MODEL path={local_dir} status=downloading", flush=True)
    try:
        revision = snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            token=token or None,
            allow_patterns=[MODEL_FILENAME, LABEL_FILENAME],
        )
        rev = "unknown"
        try:
            from huggingface_hub import HfApi

            info = HfApi(token=token or None).model_info(repo_id)
            rev = getattr(info, "sha", None) or "unknown"
        except Exception:
            rev = "unknown"
        write_sidecar(local_dir, repo_id, str(rev))
        _ = revision  # local path returned by snapshot_download
    except Exception as err:
        hint = str(err)
        if "401" in hint or "Unauthorized" in hint:
            hint += " (public WD14 models usually need no token)"
        raise RuntimeError(f"Failed to download {repo_id}: {hint}") from err

    if not model_ready(local_dir):
        raise RuntimeError(
            f"Model download incomplete under {local_dir} "
            f"(need {MODEL_FILENAME} and {LABEL_FILENAME})"
        )
    print(f"CAPTIOER_TAG_MODEL path={local_dir} status=ready", flush=True)
    return local_dir


def load_labels(csv_path: Path) -> tuple[list[str], list[int], list[int], list[int]]:
    names: list[str] = []
    rating_idx: list[int] = []
    general_idx: list[int] = []
    character_idx: list[int] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            name = (row.get("name") or "").strip()
            names.append(name)
            try:
                cat = int(row.get("category") or -1)
            except ValueError:
                cat = -1
            if cat == 9:
                rating_idx.append(i)
            elif cat == 0:
                general_idx.append(i)
            elif cat == 4:
                character_idx.append(i)
    if not names:
        raise RuntimeError(f"No tags found in {csv_path}")
    return names, rating_idx, general_idx, character_idx


def prepare_image(path: Path, target_size: int):
    import numpy as np
    from PIL import Image

    image = Image.open(path).convert("RGBA")
    canvas = Image.new("RGBA", image.size, (255, 255, 255))
    canvas.alpha_composite(image)
    image = canvas.convert("RGB")

    max_dim = max(image.size)
    pad_left = (max_dim - image.size[0]) // 2
    pad_top = (max_dim - image.size[1]) // 2
    padded = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    padded.paste(image, (pad_left, pad_top))

    if max_dim != target_size:
        resample = getattr(Image, "Resampling", Image).BICUBIC
        padded = padded.resize((target_size, target_size), resample)

    arr = np.asarray(padded, dtype=np.float32)
    arr = arr[:, :, ::-1]  # RGB -> BGR
    return np.expand_dims(arr, axis=0)


def load_session(model_dir: Path):
    try:
        import onnxruntime as ort
    except ImportError as err:
        raise RuntimeError(
            "onnxruntime is not installed. "
            "Run: pip install -r trainer/requirements-wd14.txt"
        ) from err

    model_path = model_dir / MODEL_FILENAME
    labels_path = model_dir / LABEL_FILENAME
    if not model_path.is_file() or not labels_path.is_file():
        raise RuntimeError(f"Missing model files in {model_dir}")

    providers: list[str] = []
    available = ort.get_available_providers()
    for p in ("CUDAExecutionProvider", "CPUExecutionProvider"):
        if p in available:
            providers.append(p)
    if not providers:
        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(str(model_path), providers=providers)
    input_meta = session.get_inputs()[0]
    shape = input_meta.shape
    # NHWC: [batch, height, width, channels]
    if len(shape) == 4 and isinstance(shape[1], int) and shape[1] > 0:
        target_size = int(shape[1])
    else:
        target_size = 448

    names, rating_idx, general_idx, character_idx = load_labels(labels_path)
    return session, input_meta.name, target_size, names, rating_idx, general_idx, character_idx


def tag_image(
    session: Any,
    input_name: str,
    target_size: int,
    names: list[str],
    rating_idx: list[int],
    general_idx: list[int],
    character_idx: list[int],
    image_path: Path,
    threshold: float,
    character_threshold: float,
) -> str:
    import numpy as np

    tensor = prepare_image(image_path, target_size)
    outputs = session.run(None, {input_name: tensor})
    probs = np.asarray(outputs[0]).reshape(-1)

    ratings = [(names[i], float(probs[i])) for i in rating_idx]
    ratings.sort(key=lambda x: x[1], reverse=True)
    rating_tag = ratings[0][0] if ratings else None

    general = [
        (names[i], float(probs[i]))
        for i in general_idx
        if float(probs[i]) >= threshold and names[i]
    ]
    general.sort(key=lambda x: x[1], reverse=True)

    characters = [
        (names[i], float(probs[i]))
        for i in character_idx
        if float(probs[i]) >= character_threshold and names[i]
    ]
    characters.sort(key=lambda x: x[1], reverse=True)

    parts: list[str] = []
    if rating_tag:
        parts.append(rating_tag)
    parts.extend(name for name, _ in characters)
    parts.extend(name for name, _ in general)
    # CSV names already use underscores; keep for SD/XL training captions
    return ", ".join(parts)


def cmd_ensure(args: argparse.Namespace) -> int:
    try:
        ensure_model(args.download_path, args.repo_id, args.token or None)
        print("CAPTIOER_TAG_DONE", flush=True)
        return 0
    except Exception as err:
        emit_error(str(err))
        return 1


def emit_payload(kind: str, obj: dict[str, Any]) -> None:
    """Emit a protocol line as ASCII base64 JSON (safe across pipes / special paths)."""
    raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    b64 = base64.b64encode(raw).decode("ascii")
    print(f"{kind} {b64}", flush=True)


def cmd_tag(args: argparse.Namespace) -> int:
    try:
        model_dir = Path(args.model_dir).expanduser().resolve()
        if args.ensure and args.repo_id and args.download_path:
            model_dir = ensure_model(
                args.download_path, args.repo_id, args.token or None
            )

        if not model_ready(model_dir):
            raise RuntimeError(
                f"WD14 model not found at {model_dir}. "
                "Install deps and retry Auto Caption to download."
            )

        (
            session,
            input_name,
            target_size,
            names,
            rating_idx,
            general_idx,
            character_idx,
        ) = load_session(model_dir)

        images: list[str] = list(args.images or [])
        if args.images_file:
            text = Path(args.images_file).read_text(encoding="utf-8")
            for line in text.splitlines():
                line = line.strip()
                if line:
                    images.append(line)

        if not images:
            raise RuntimeError("No images provided")

        threshold = float(args.threshold)
        character_threshold = float(args.character_threshold)

        for raw_path in images:
            path = Path(raw_path)
            try:
                if not path.is_file():
                    raise RuntimeError(f"Image not found: {path}")
                tags = tag_image(
                    session,
                    input_name,
                    target_size,
                    names,
                    rating_idx,
                    general_idx,
                    character_idx,
                    path,
                    threshold,
                    character_threshold,
                )
                emit_payload("CAPTIOER_TAG_B64", {"path": str(path), "tags": tags})
            except Exception as err:
                emit_payload(
                    "CAPTIOER_TAG_ITEM_ERROR_B64",
                    {"path": str(path), "error": str(err)},
                )

        print("CAPTIOER_TAG_DONE", flush=True)
        return 0
    except Exception as err:
        emit_error(str(err))
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Captioer WD14 ONNX tagger")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ensure = sub.add_parser("ensure", help="Download WD14 model if missing")
    p_ensure.add_argument("--download-path", required=True)
    p_ensure.add_argument("--repo-id", required=True)
    p_ensure.add_argument("--token", default="")

    p_tag = sub.add_parser("tag", help="Tag one or more images")
    p_tag.add_argument("--model-dir", required=True)
    p_tag.add_argument("--threshold", type=float, default=0.35)
    p_tag.add_argument("--character-threshold", type=float, default=0.85)
    p_tag.add_argument("--images", nargs="*", default=[])
    p_tag.add_argument("--images-file", default="")
    p_tag.add_argument("--ensure", action="store_true")
    p_tag.add_argument("--download-path", default="")
    p_tag.add_argument("--repo-id", default="")
    p_tag.add_argument("--token", default="")

    args = parser.parse_args()
    if args.command == "ensure":
        return cmd_ensure(args)
    if args.command == "tag":
        return cmd_tag(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
