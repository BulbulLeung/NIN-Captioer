#!/usr/bin/env python3
"""
Captioer native Krea 2 LoRA trainer.
Train on Krea-2-Raw with image + .txt caption sidecars; optionally sample on Turbo.

Progress protocol (stdout):
  CAPTIOER_PROGRESS step=N total=T loss=X.XXXX
  CAPTIOER_DONE path=...
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def log(msg: str) -> None:
    print(msg, flush=True)


def progress(step: int, total: int, loss: float) -> None:
    print(f"CAPTIOER_PROGRESS step={step} total={total} loss={loss:.6f}", flush=True)


def done(path: str) -> None:
    print(f"CAPTIOER_DONE path={path}", flush=True)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_device(device: str) -> tuple[str, str | None]:
    """Return (torch_device, cuda_visible_index_or_None)."""
    d = (device or "cuda:0").strip().lower()
    if d.startswith("cuda:"):
        idx = d.split(":", 1)[1]
        return "cuda", idx
    if d == "cuda":
        return "cuda", "0"
    if d == "cpu":
        return "cpu", None
    return d, None


def collect_samples(folder: Path, caption_ext: str, trigger: str) -> list[tuple[Path, str]]:
    ext = caption_ext.lstrip(".")
    pairs: list[tuple[Path, str]] = []
    if not folder.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {folder}")
    for p in sorted(folder.iterdir()):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        cap = p.with_suffix(f".{ext}")
        text = ""
        if cap.is_file():
            text = cap.read_text(encoding="utf-8", errors="ignore").strip()
        if trigger and trigger not in text:
            text = f"{trigger}, {text}".strip(", ").strip()
        pairs.append((p, text))
    if not pairs:
        raise RuntimeError(f"No images found in {folder}")
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description="Captioer Krea 2 LoRA trainer")
    parser.add_argument("--config", required=True, help="Path to Captioer train JSON")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg.get("model") or {}
    train_cfg = cfg.get("train") or {}
    net_cfg = cfg.get("network") or {}
    save_cfg = cfg.get("save") or {}
    sample_cfg = cfg.get("sample") or {}
    datasets = cfg.get("datasets") or [{}]
    ds0 = datasets[0] if datasets else {}

    arch = model_cfg.get("arch") or "krea2"
    if arch != "krea2":
        log(f"ERROR: unsupported arch={arch}; only krea2 is supported")
        return 2

    train_path = (
        model_cfg.get("train_name_or_path")
        or model_cfg.get("name_or_path")
        or "krea/Krea-2-Raw"
    )
    sample_path = model_cfg.get("sample_name_or_path") or "krea/Krea-2-Turbo"
    name = cfg.get("name") or "krea2_lora"
    out_root = Path(cfg.get("training_folder") or "output")
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    folder = Path(ds0.get("folder_path") or "")
    caption_ext = ds0.get("caption_ext") or "txt"
    trigger = (cfg.get("trigger_word") or "").strip()
    dropout = float(ds0.get("caption_dropout_rate") or 0.0)
    resolutions = ds0.get("resolution") or [1024]
    res = int(resolutions[0] if isinstance(resolutions, list) else resolutions)

    steps = int(train_cfg.get("steps") or 1000)
    batch_size = int(train_cfg.get("batch_size") or 1)
    grad_accum = int(train_cfg.get("gradient_accumulation_steps") or 1)
    lr = float(train_cfg.get("lr") or 1e-4)
    dtype_name = (train_cfg.get("dtype") or "bf16").lower()
    grad_ckpt = bool(train_cfg.get("gradient_checkpointing", True))
    optimizer_name = (train_cfg.get("optimizer") or "adamw8bit").lower()
    disable_sampling = bool(train_cfg.get("disable_sampling", True))
    save_every = int(save_cfg.get("save_every") or 250)
    rank = int(net_cfg.get("linear") or 16)
    alpha = int(net_cfg.get("linear_alpha") or rank)

    torch_device, cuda_idx = parse_device(cfg.get("device") or "cuda:0")
    if cuda_idx is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_idx

    token = (cfg.get("huggingface_token") or os.environ.get("HF_TOKEN") or "").strip()
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token

    log(f"Captioer Krea2 trainer starting")
    log(f"train_base={train_path}")
    log(f"sample_base={sample_path}")
    log(f"dataset={folder}")
    log(f"output={out_dir}")
    log(f"steps={steps} batch={batch_size} lr={lr} rank={rank} alpha={alpha}")

    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from PIL import Image
        from torchvision import transforms
    except Exception as e:
        log(f"ERROR: failed to import torch/vision: {e}")
        return 3

    try:
        from diffusers import Krea2Pipeline
        from peft import LoraConfig, get_peft_model
        from peft.utils import get_peft_model_state_dict
        from safetensors.torch import save_file
    except Exception as e:
        log(
            "ERROR: need recent diffusers with Krea2Pipeline and peft. "
            "Try: pip install -r requirements.txt "
            'and/or pip install "git+https://github.com/huggingface/diffusers.git"'
        )
        log(f"detail: {e}")
        return 4

    samples = collect_samples(folder, caption_ext, trigger)
    log(f"loaded {len(samples)} image/caption pairs")

    if dtype_name in ("bf16", "bfloat16"):
        weight_dtype = torch.bfloat16
    elif dtype_name in ("fp16", "float16"):
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    device = torch.device("cuda" if torch_device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        log("WARNING: CUDA not available; training on CPU will be extremely slow")

    log("Loading Krea2Pipeline (train base / Raw)…")
    pipe = Krea2Pipeline.from_pretrained(train_path, torch_dtype=weight_dtype)
    transformer = pipe.transformer
    vae = pipe.vae
    text_encoder = getattr(pipe, "text_encoder", None) or getattr(pipe, "text_encoder_2", None)

    vae.requires_grad_(False)
    vae.to(device, dtype=weight_dtype)
    if text_encoder is not None:
        text_encoder.requires_grad_(False)
        text_encoder.to(device, dtype=weight_dtype)

    # Freeze transformer base; LoRA adapters will be trainable
    transformer.requires_grad_(False)
    if grad_ckpt and hasattr(transformer, "enable_gradient_checkpointing"):
        transformer.enable_gradient_checkpointing()

    target_modules = [
        "to_q",
        "to_k",
        "to_v",
        "to_out.0",
        "proj_out",
        "linear_1",
        "linear_2",
    ]
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    transformer = get_peft_model(transformer, lora_config)
    transformer.to(device, dtype=weight_dtype)
    transformer.train()

    class CaptionImageDataset(Dataset):
        def __init__(self, pairs: list[tuple[Path, str]], size: int, drop: float):
            self.pairs = pairs
            self.drop = drop
            self.tf = transforms.Compose(
                [
                    transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )

        def __len__(self) -> int:
            return len(self.pairs)

        def __getitem__(self, idx: int):
            path, caption = self.pairs[idx]
            img = Image.open(path).convert("RGB")
            pixel = self.tf(img)
            if self.drop > 0 and random.random() < self.drop:
                caption = ""
            return {"pixel_values": pixel, "caption": caption}

    dataset = CaptionImageDataset(samples, res, dropout)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    params = [p for p in transformer.parameters() if p.requires_grad]
    if optimizer_name in ("adamw8bit", "adam8bit"):
        try:
            import bitsandbytes as bnb

            optimizer = bnb.optim.AdamW8bit(params, lr=lr)
        except Exception:
            log("bitsandbytes unavailable; falling back to AdamW")
            optimizer = torch.optim.AdamW(params, lr=lr)
    else:
        optimizer = torch.optim.AdamW(params, lr=lr)

    # Persist config snapshot
    (out_dir / "captioer_train_config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    def encode_prompt(captions: list[str]):
        # Prefer pipeline encode helpers when present
        if hasattr(pipe, "encode_prompt"):
            try:
                out = pipe.encode_prompt(
                    prompt=captions,
                    device=device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                )
                if isinstance(out, tuple):
                    return out[0]
                return out
            except TypeError:
                pass
            try:
                out = pipe.encode_prompt(captions, device=device)
                if isinstance(out, tuple):
                    return out[0]
                return out
            except Exception as e:
                log(f"encode_prompt fallback warning: {e}")
        raise RuntimeError(
            "Could not encode prompts with this Krea2Pipeline build. "
            "Upgrade diffusers to a version that exposes encode_prompt for Krea2."
        )

    def save_lora(step: int | None = None) -> Path:
        target = out_dir if step is None else out_dir / f"checkpoint-{step}"
        target.mkdir(parents=True, exist_ok=True)
        state = get_peft_model_state_dict(transformer)
        # Cast to fp16/bf16 for smaller files when requested
        save_dtype = torch.float16 if (save_cfg.get("dtype") or "").startswith("float16") else None
        tensors = {}
        for k, v in state.items():
            t = v.detach().cpu()
            if save_dtype is not None and t.is_floating_point():
                t = t.to(save_dtype)
            tensors[k] = t
        path = target / "pytorch_lora_weights.safetensors"
        save_file(tensors, str(path))
        log(f"saved LoRA -> {path}")
        return path

    global_step = 0
    data_iter = iter(loader)
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0

    log("Training loop…")
    while global_step < steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        pixel_values = batch["pixel_values"].to(device, dtype=weight_dtype)
        captions = list(batch["caption"])

        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample()
            if hasattr(vae.config, "scaling_factor"):
                latents = latents * vae.config.scaling_factor
            prompt_embeds = encode_prompt(captions)

        # Flow-matching style: noise latents and predict velocity / noise
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.rand(bsz, device=device, dtype=weight_dtype)
        # x_t = (1 - t) * x0 + t * noise  (simple linear path)
        t = timesteps.view(-1, *([1] * (latents.ndim - 1)))
        noisy = (1.0 - t) * latents + t * noise
        target = noise - latents

        try:
            pred = transformer(
                hidden_states=noisy,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]
        except TypeError:
            pred = transformer(
                noisy,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]

        loss = torch.nn.functional.mse_loss(pred.float(), target.float())
        loss = loss / grad_accum
        loss.backward()
        running_loss += loss.item() * grad_accum

        if (global_step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        global_step += 1
        avg = running_loss / global_step
        progress(global_step, steps, float(loss.item() * grad_accum))

        if save_every > 0 and global_step % save_every == 0:
            save_lora(global_step)

        if (
            not disable_sampling
            and sample_cfg.get("sample_every")
            and global_step % int(sample_cfg["sample_every"]) == 0
        ):
            log(
                f"sample skipped/lightweight at step {global_step} "
                f"(sample_base={sample_path}; full sample pipeline optional in later builds)"
            )

    final_path = save_lora(None)
    done(str(final_path))
    log("Training finished")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("ERROR: interrupted")
        raise SystemExit(130)
    except Exception as e:
        log(f"ERROR: {e}")
        raise SystemExit(1)
