#!/usr/bin/env python3
"""
Captioer native Krea 2 LoRA trainer.

Train strategy mirrors AI-Toolkit (24GB path):
  1) Cache VAE latents to disk (only VAE on GPU)
  2) Cache text embeddings to disk (only text encoder on GPU), then unload TE
  3) Train with transformer (+LoRA) resident on GPU — never swap whole models per step

Progress protocol (stdout):
  CAPTIOER_PROGRESS step=N total=T loss=X.XXXX
  CAPTIOER_DONE path=...
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import re
import sys
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

# Diffusers Krea2Transformer2DModel names → original krea-ai / ComfyUI names.
# Inverse of diffusers.loaders.lora_conversion_utils._convert_non_diffusers_krea2_lora_to_diffusers.
_KREA2_ATTN_TO_COMFY = {
    "to_q": "wq",
    "to_k": "wk",
    "to_v": "wv",
    "to_out.0": "wo",
    "to_gate": "gate",
}
_KREA2_FF_TO_COMFY = {"ff.gate": "gate", "ff.up": "up", "ff.down": "down"}
_KREA2_STANDALONE_TO_COMFY = {
    "img_in": "first",
    "final_layer.linear": "last.linear",
    "time_embed.linear_1": "tmlp.0",
    "time_embed.linear_2": "tmlp.2",
    "time_mod_proj": "tproj.1",
    "txt_in.linear_1": "txtmlp.1",
    "txt_in.linear_2": "txtmlp.3",
    "text_fusion.projector": "txtfusion.projector",
}


def log(msg: str) -> None:
    print(msg, flush=True)


def convert_diffusers_krea2_lora_to_comfy(state_dict: dict) -> dict:
    """Map PEFT/Diffusers Krea2 LoRA keys to ComfyUI `diffusion_model.*` keys."""
    out: dict = {}
    skipped: list[str] = []

    for key, tensor in state_dict.items():
        k = key
        for prefix in ("base_model.model.", "transformer.", "diffusion_model."):
            if k.startswith(prefix):
                k = k[len(prefix) :]
        m = re.search(r"\.(lora_[AB])\.weight$", k)
        if m is None:
            # Accept already-converted Diffusers lora.down / lora.up naming.
            m_du = re.search(r"\.lora\.(down|up)\.weight$", k)
            if m_du is None:
                skipped.append(key)
                continue
            side = "lora_A" if m_du.group(1) == "down" else "lora_B"
            module = k[: m_du.start()]
            suffix = f".{side}.weight"
        else:
            module = k[: m.start()]
            suffix = k[m.start() :]

        comfy_module: str | None = None
        m_attn = re.match(r"transformer_blocks\.(\d+)\.attn\.(.+)$", module)
        if m_attn and m_attn.group(2) in _KREA2_ATTN_TO_COMFY:
            comfy_module = f"blocks.{m_attn.group(1)}.attn.{_KREA2_ATTN_TO_COMFY[m_attn.group(2)]}"
        if comfy_module is None:
            m_ff = re.match(r"transformer_blocks\.(\d+)\.(ff\.(?:gate|up|down))$", module)
            if m_ff and m_ff.group(2) in _KREA2_FF_TO_COMFY:
                comfy_module = f"blocks.{m_ff.group(1)}.mlp.{_KREA2_FF_TO_COMFY[m_ff.group(2)]}"
        if comfy_module is None:
            m_tf = re.match(
                r"text_fusion\.(layerwise_blocks|refiner_blocks)\.(\d+)\.attn\.(.+)$",
                module,
            )
            if m_tf and m_tf.group(3) in _KREA2_ATTN_TO_COMFY:
                comfy_module = (
                    f"txtfusion.{m_tf.group(1)}.{m_tf.group(2)}.attn."
                    f"{_KREA2_ATTN_TO_COMFY[m_tf.group(3)]}"
                )
        if comfy_module is None:
            m_tf_ff = re.match(
                r"text_fusion\.(layerwise_blocks|refiner_blocks)\.(\d+)\.(ff\.(?:gate|up|down))$",
                module,
            )
            if m_tf_ff and m_tf_ff.group(3) in _KREA2_FF_TO_COMFY:
                comfy_module = (
                    f"txtfusion.{m_tf_ff.group(1)}.{m_tf_ff.group(2)}.mlp."
                    f"{_KREA2_FF_TO_COMFY[m_tf_ff.group(3)]}"
                )
        if comfy_module is None:
            comfy_module = _KREA2_STANDALONE_TO_COMFY.get(module)

        if comfy_module is None:
            skipped.append(key)
            continue
        out[f"diffusion_model.{comfy_module}{suffix}"] = tensor

    if skipped:
        raise ValueError(
            f"Could not map {len(skipped)} LoRA key(s) to ComfyUI format; "
            f"examples: {skipped[:5]}"
        )
    return out


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


def flush() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def safe_job_name(name: str) -> str:
    import re

    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", (name or "job").strip()) or "job"
    return cleaned.rstrip(" .")


def format_step(step: int) -> str:
    return f"{int(step):06d}"


def cache_key(*parts: object) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode("utf-8", errors="replace"))
        h.update(b"\0")
    return h.hexdigest()[:20]


def normalize_sample_prompts(
    raw: object,
    *,
    default_width: int,
    default_height: int,
    default_seed: int,
) -> list[dict]:
    """Accept [{prompt,width,height,seed}] or legacy string[]."""
    if not isinstance(raw, list) or not raw:
        return []
    out: list[dict] = []
    for i, item in enumerate(raw):
        if isinstance(item, str):
            text = item.strip()
            if not text:
                continue
            out.append(
                {
                    "prompt": text,
                    "width": default_width,
                    "height": default_height,
                    "seed": default_seed + i,
                }
            )
            continue
        if not isinstance(item, dict):
            continue
        text = str(item.get("prompt") or "").strip()
        if not text:
            continue
        out.append(
            {
                "prompt": text,
                "width": int(item.get("width") or default_width),
                "height": int(item.get("height") or default_height),
                "seed": int(item.get("seed") if item.get("seed") is not None else default_seed + i),
            }
        )
    return out


def run_midtrain_samples(
    *,
    step: int,
    prompts: list[dict],
    out_dir: Path,
    job_name: str,
    pipe,
    train_transformer,
    vae,
    text_encoder,
    train_path: str,
    device,
    weight_dtype,
    guidance_scale: float,
    num_inference_steps: int,
    neg: str,
    trigger: str,
    low_vram: bool = True,
) -> list[str]:
    """Sample with in-memory Raw+LoRA.

    low_vram=True: sequential GPU residency (24GB-safe) — never keep DiT + TE + VAE
    on GPU together; DiT is moved to CPU during encode/decode.
    low_vram=False: keep DiT on GPU; only stage TE/VAE on/off (faster, more VRAM).
    """
    import time as _time

    import torch
    from diffusers import Krea2Pipeline
    from diffusers.image_processor import VaeImageProcessor

    if not prompts:
        log("sample: no prompts configured; skipping")
        return []

    sample_dir = out_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    safe_name = safe_job_name(job_name)
    step_tag = format_step(step)

    def _vram(tag: str) -> dict:
        if device.type != "cuda":
            return {"tag": tag}
        return {
            "tag": tag,
            "alloc_gb": round(torch.cuda.memory_allocated() / 1e9, 3),
            "reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 3),
        }

    was_training = bool(getattr(train_transformer, "training", False))
    train_transformer.eval()
    pipe.transformer = train_transformer

    te = text_encoder
    te_reloaded = False
    if te is None:
        log("Reloading text encoder from train base for sampling…")
        tmp = Krea2Pipeline.from_pretrained(train_path, torch_dtype=weight_dtype)
        te = getattr(tmp, "text_encoder", None) or getattr(tmp, "text_encoder_2", None)
        try:
            tmp.transformer = None
            tmp.vae = None
        except Exception:
            pass
        del tmp
        flush()
        te_reloaded = True
        if te is None:
            raise RuntimeError("No text encoder available for mid-train sampling")

    mode = "staged offload" if low_vram else "DiT resident"
    log(f"Sampling at step {step} with train base (Raw+LoRA; {mode})…")

    try:
        # Keep VAE/TE off GPU until their stage. DiT may start on GPU from training.
        vae.to("cpu")
        te.to("cpu")
        pipe.vae = vae
        pipe.text_encoder = te
        if not low_vram:
            train_transformer.to(device)
        flush()

        for i, item in enumerate(prompts):
            prompt = item["prompt"]
            if trigger and trigger not in prompt:
                prompt = f"{trigger}, {prompt}".strip(", ").strip()
            w = int(item["width"])
            h = int(item["height"])
            seed = int(item["seed"])
            gen = torch.Generator(device="cpu").manual_seed(seed)
            log(f"  sample {i + 1}/{len(prompts)} seed={seed} {w}x{h}")
            step_t0 = _time.perf_counter()

            # ---- Stage A: text encode ----
            if low_vram:
                log("  stage encode: TE on GPU, DiT off…")
                train_transformer.to("cpu")
                flush()
            else:
                log("  stage encode: TE on GPU, DiT resident…")
            te.to(device, dtype=weight_dtype)
            pipe.text_encoder = te
            with torch.inference_mode():
                enc = pipe.encode_prompt(
                    prompt=[prompt],
                    device=device,
                    num_images_per_prompt=1,
                )
                prompt_embeds, prompt_embeds_mask = enc[0], enc[1]
                neg_embeds = neg_mask = None
                if float(guidance_scale) > 0:
                    nenc = pipe.encode_prompt(
                        prompt=[neg or ""],
                        device=device,
                        num_images_per_prompt=1,
                    )
                    neg_embeds, neg_mask = nenc[0], nenc[1]
            te.to("cpu")
            flush()
            log(f"  encode done ({_time.perf_counter() - step_t0:.1f}s) {_vram('after_encode')}")

            # ---- Stage B: denoise to latents (DiT on GPU) ----
            log("  stage denoise: DiT on GPU, TE/VAE off…")
            train_transformer.to(device)
            pipe.transformer = train_transformer
            # Prevent pipeline from expecting TE/VAE on device during denoise.
            pipe.text_encoder = None
            flush()

            def _on_step_end(pipe_, step_idx, timestep, callback_kwargs):
                if step_idx == 0 or (step_idx + 1) % 4 == 0 or step_idx + 1 == num_inference_steps:
                    log(
                        f"  denoise {step_idx + 1}/{num_inference_steps} "
                        f"({_time.perf_counter() - step_t0:.1f}s)"
                    )
                return callback_kwargs

            call_kwargs = dict(
                prompt=None,
                prompt_embeds=prompt_embeds,
                prompt_embeds_mask=prompt_embeds_mask,
                width=w,
                height=h,
                num_inference_steps=int(num_inference_steps),
                guidance_scale=float(guidance_scale),
                generator=gen,
                output_type="latent",
            )
            if neg_embeds is not None:
                call_kwargs["negative_prompt_embeds"] = neg_embeds
                call_kwargs["negative_prompt_embeds_mask"] = neg_mask

            with torch.inference_mode():
                try:
                    latents_out = pipe(
                        **call_kwargs,
                        callback_on_step_end=_on_step_end,
                    )
                except TypeError:
                    latents_out = pipe(**call_kwargs)
            latents = latents_out.images if hasattr(latents_out, "images") else latents_out[0]
            log(f"  denoise done ({_time.perf_counter() - step_t0:.1f}s) {_vram('after_denoise')}")

            # ---- Stage C: VAE decode ----
            if low_vram:
                log("  stage decode: VAE on GPU, DiT off…")
                train_transformer.to("cpu")
                flush()
            else:
                log("  stage decode: VAE on GPU, DiT resident…")
            vae.to(device, dtype=weight_dtype)
            pipe.vae = vae
            with torch.inference_mode():
                packed = latents
                if not torch.is_tensor(packed):
                    packed = packed[0]
                unpacked = pipe._unpack_latents(packed, h, w)
                unpacked = unpacked.to(device=device, dtype=vae.dtype)
                latents_mean = (
                    torch.tensor(vae.config.latents_mean)
                    .view(1, vae.config.z_dim, 1, 1, 1)
                    .to(unpacked.device, unpacked.dtype)
                )
                latents_std = (
                    1.0
                    / torch.tensor(vae.config.latents_std)
                    .view(1, vae.config.z_dim, 1, 1, 1)
                    .to(unpacked.device, unpacked.dtype)
                )
                decoded = unpacked / latents_std + latents_mean
                image = vae.decode(decoded, return_dict=False)[0][:, :, 0]
                processor = getattr(pipe, "image_processor", None) or VaeImageProcessor()
                pil_list = processor.postprocess(image, output_type="pil")
            img = pil_list[0]
            vae.to("cpu")
            flush()

            out_name = f"{safe_name}_Sampling_{step_tag}.png"
            if len(prompts) > 1:
                out_name = f"{safe_name}_Sampling_{step_tag}_{i + 1}.png"
            out_path = sample_dir / out_name
            img.save(out_path)
            saved.append(str(out_path))
            log(f"  saved {out_path} ({_time.perf_counter() - step_t0:.1f}s)")

            # Ensure DiT on GPU between prompts / before training resumes
            train_transformer.to(device)
            flush()
    finally:
        try:
            vae.to("cpu")
        except Exception:
            pass
        try:
            te.to("cpu")
        except Exception:
            pass
        try:
            pipe.text_encoder = te if not te_reloaded else None
        except Exception:
            pass
        if te_reloaded:
            try:
                del te
            except Exception:
                pass
        try:
            pipe.transformer = train_transformer
            train_transformer.to(device)
        except Exception:
            pass
        flush()
        if was_training:
            train_transformer.train()
        log(f"Sample cleanup done; train VRAM {_vram('restored')}")

    return saved


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
    job_name = cfg.get("name") or "krea2_lora"
    out_root = Path(cfg.get("training_folder") or "output")
    out_dir = out_root / job_name
    out_dir.mkdir(parents=True, exist_ok=True)
    latent_cache_dir = out_dir / "_latent_cache"
    text_cache_dir = out_dir / "_text_cache"
    latent_cache_dir.mkdir(parents=True, exist_ok=True)
    text_cache_dir.mkdir(parents=True, exist_ok=True)

    folder = Path(ds0.get("folder_path") or "")
    caption_ext = ds0.get("caption_ext") or "txt"
    trigger = (cfg.get("trigger_word") or "").strip()
    dropout = float(ds0.get("caption_dropout_rate") or 0.0)
    resolutions = ds0.get("resolution") or [1024]
    res = int(resolutions[0] if isinstance(resolutions, list) else resolutions)

    # AI-Toolkit 24GB defaults: always cache unless explicitly disabled
    cache_latents = bool(ds0.get("cache_latents_to_disk", True))
    cache_text = bool(train_cfg.get("cache_text_embeddings", True))

    steps = int(train_cfg.get("steps") or 1000)
    batch_size = int(train_cfg.get("batch_size") or 1)
    grad_accum = int(train_cfg.get("gradient_accumulation_steps") or 1)
    lr = float(train_cfg.get("lr") or 1e-4)
    dtype_name = (train_cfg.get("dtype") or "bf16").lower()
    optimizer_name = (train_cfg.get("optimizer") or "adamw8bit").lower()
    disable_sampling = bool(train_cfg.get("disable_sampling", True))
    skip_first_sample = bool(train_cfg.get("skip_first_sample", False))
    sample_every = int(sample_cfg.get("sample_every") or 0)
    sample_start_step = int(sample_cfg.get("sample_start_step") or 0)
    save_every = int(save_cfg.get("save_every") or 250)
    rank = int(net_cfg.get("linear") or 16)
    alpha = int(net_cfg.get("linear_alpha") or rank)
    low_vram = bool(model_cfg.get("low_vram", True))
    # Low VRAM forces gradient checkpointing; otherwise respect the train toggle.
    grad_ckpt = bool(train_cfg.get("gradient_checkpointing", True)) or low_vram

    prompts_raw = sample_cfg.get("prompts") or []
    sample_prompts = normalize_sample_prompts(
        prompts_raw,
        default_width=int(sample_cfg.get("width") or 1024),
        default_height=int(sample_cfg.get("height") or 1024),
        default_seed=int(sample_cfg.get("seed") or 42),
    )
    sample_guidance = float(sample_cfg.get("guidance_scale") or 0.0)
    sample_steps_n = int(sample_cfg.get("sample_steps") or 8)
    sample_neg = str(sample_cfg.get("neg") or "")

    torch_device, cuda_idx = parse_device(cfg.get("device") or "cuda:0")
    if cuda_idx is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_idx

    token = (cfg.get("huggingface_token") or os.environ.get("HF_TOKEN") or "").strip()
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token

    log("Captioer Krea2 trainer starting (AI-Toolkit-style cache strategy)")
    log(f"train_base={train_path}")
    log(f"dataset={folder}")
    log(f"output={out_dir}")
    log(
        f"steps={steps} batch={batch_size} lr={lr} rank={rank} alpha={alpha} "
        f"cache_latents={cache_latents} cache_text={cache_text} "
        f"low_vram={low_vram} offload={'staged' if low_vram else 'dit_resident'} "
        f"checkpoint={grad_ckpt}"
    )

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
        from safetensors.torch import load_file, save_file
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

    device = torch.device(
        "cuda" if torch_device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
    if device.type != "cuda":
        log("WARNING: CUDA not available; training on CPU will be extremely slow")

    total_vram_gb = (
        torch.cuda.get_device_properties(0).total_memory / 1e9 if device.type == "cuda" else 0.0
    )

    log("Loading Krea2Pipeline (components stay on CPU until needed)…")
    pipe = Krea2Pipeline.from_pretrained(train_path, torch_dtype=weight_dtype)
    transformer = pipe.transformer
    vae = pipe.vae
    text_encoder = getattr(pipe, "text_encoder", None) or getattr(pipe, "text_encoder_2", None)

    # Keep everything on CPU after load (AI-Toolkit low_vram load pattern).
    transformer.requires_grad_(False)
    transformer.to("cpu")
    vae.requires_grad_(False)
    vae.to("cpu")
    if text_encoder is not None:
        text_encoder.requires_grad_(False)
        text_encoder.to("cpu")
    flush()

    def normalize_latents(raw: torch.Tensor) -> torch.Tensor:
        c = getattr(vae, "config", None)
        if c is not None and hasattr(c, "latents_mean") and hasattr(c, "latents_std"):
            mean = torch.tensor(c.latents_mean).view(1, c.z_dim, 1, 1, 1).to(raw.device, raw.dtype)
            std = torch.tensor(c.latents_std).view(1, c.z_dim, 1, 1, 1).to(raw.device, raw.dtype)
            return (raw - mean) / std
        if c is not None and hasattr(c, "scaling_factor"):
            return raw * c.scaling_factor
        return raw

    def pack_latents(latents_5d: torch.Tensor) -> torch.Tensor:
        if latents_5d.ndim == 5:
            latents_4d = latents_5d.squeeze(2)
        else:
            latents_4d = latents_5d
        bsz, channels, height, width = latents_4d.shape
        return pipe._pack_latents(latents_4d, bsz, channels, height, width)

    def encode_prompt_on(dev: torch.device, captions: list[str]):
        out = pipe.encode_prompt(prompt=captions, device=dev, num_images_per_prompt=1)
        if not isinstance(out, tuple) or len(out) < 2:
            raise RuntimeError("Krea2 encode_prompt must return (embeds, mask)")
        return out[0], out[1]

    tf_img = transforms.Compose(
        [
            transforms.Resize(res, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(res),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    # ------------------------------------------------------------------
    # Phase A: cache latents (only VAE on GPU) — AI-Toolkit pattern
    # ------------------------------------------------------------------
    latent_index: list[dict] = []
    if cache_latents:
        log("Caching latents to disk (VAE on GPU only)…")
        vae.to(device, dtype=weight_dtype)
        if hasattr(vae, "enable_slicing"):
            vae.enable_slicing()
        if hasattr(vae, "enable_tiling"):
            vae.enable_tiling()
        flush()
        with torch.no_grad():
            for i, (path, _cap) in enumerate(samples):
                try:
                    mtime = path.stat().st_mtime_ns
                except OSError:
                    mtime = 0
                key = cache_key(str(path.resolve()), res, mtime, "lat_v1")
                out_file = latent_cache_dir / f"{key}.safetensors"
                if not out_file.is_file():
                    img = Image.open(path).convert("RGB")
                    pixel = tf_img(img).unsqueeze(0).to(device, dtype=weight_dtype)
                    if pixel.ndim == 4:
                        pixel = pixel.unsqueeze(2)
                    lat = vae.encode(pixel).latent_dist.sample()
                    lat = normalize_latents(lat).squeeze(0).cpu()
                    save_file({"latents": lat.contiguous()}, str(out_file))
                latent_index.append({"path": str(path), "caption": _cap, "latent": str(out_file)})
                if (i + 1) % 10 == 0 or i + 1 == len(samples):
                    log(f"  latent cache {i + 1}/{len(samples)}")
        vae.to("cpu")
        flush()
        log("Latent cache complete; VAE moved to CPU")
    else:
        for path, cap in samples:
            latent_index.append({"path": str(path), "caption": cap, "latent": None})

    # ------------------------------------------------------------------
    # Phase B: cache text embeddings (only TE on GPU), then unload
    # ------------------------------------------------------------------
    text_paths: dict[str, str] = {}
    if cache_text:
        if text_encoder is None:
            raise RuntimeError("cache_text_embeddings requires a text encoder")
        log("Caching text embeddings to disk (text encoder on GPU only)…")
        text_encoder.to(device, dtype=weight_dtype)
        flush()
        unique_caps = sorted({item["caption"] for item in latent_index} | {""})
        with torch.no_grad():
            for i, cap in enumerate(unique_caps):
                key = cache_key(cap, "txt_v1")
                out_file = text_cache_dir / f"{key}.safetensors"
                if not out_file.is_file():
                    embeds, mask = encode_prompt_on(device, [cap])
                    save_file(
                        {
                            "prompt_embeds": embeds.squeeze(0).cpu().contiguous(),
                            "prompt_embeds_mask": mask.squeeze(0).cpu().contiguous(),
                        },
                        str(out_file),
                    )
                text_paths[cap] = str(out_file)
                if (i + 1) % 20 == 0 or i + 1 == len(unique_caps):
                    log(f"  text cache {i + 1}/{len(unique_caps)}")
        text_encoder.to("cpu")
        flush()
        # Keep TE on CPU for mid-train Raw sampling (do not load Turbo).
        try:
            pipe.text_encoder = text_encoder
        except Exception:
            pass
        flush()
        log("Text embedding cache complete; text encoder kept on CPU for sampling")

    # ------------------------------------------------------------------
    # Phase C: PEFT LoRA + torchao weight-only quant on base layers
    # (AI-Toolkit-style WOQ; NOT bitsandbytes Linear8bitLt — that path is
    # pathologically slow on DiT and shows ~100% util / ~110W / minutes/step.)
    # ------------------------------------------------------------------
    log("Preparing transformer + LoRA…")
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

    # peft>=0.17 requires torchao>=0.16 which needs torch>=2.11; our venv is
    # torch 2.6 + torchao 0.9. Disable peft's torchao dispatcher so standard
    # LoRA Linear wrappers are used, then we WOQ the base_layer ourselves.
    try:
        import peft.import_utils as _peft_iu
        import peft.tuners.lora.torchao as _peft_lora_tao

        if hasattr(_peft_iu.is_torchao_available, "cache_clear"):
            _peft_iu.is_torchao_available.cache_clear()
        _peft_iu.is_torchao_available = lambda: False  # type: ignore[assignment]
        _peft_lora_tao.is_torchao_available = lambda: False  # type: ignore[assignment]
    except Exception:
        pass

    transformer = get_peft_model(transformer, lora_config)

    use_quant = bool(model_cfg.get("quantize", True))
    if low_vram:
        use_quant = use_quant or total_vram_gb < 40.0
    if use_quant and device.type == "cuda":
        try:
            from torchao.quantization.quant_api import Int8WeightOnlyConfig, quantize_

            # Int8 WOQ: uint4/int4 TensorCore layouts often cannot .cpu() on this
            # torchao/torch combo; int8 still halves weight VRAM vs bf16 and uses
            # fast kernels (unlike bitsandbytes Linear8bitLt).
            ao_cfg = Int8WeightOnlyConfig()
            n_q = 0
            skipped_lora = 0
            log("Quantizing transformer Linear weights with torchao Int8WeightOnly…")

            def _woq_linear(lin: torch.nn.Linear) -> bool:
                if lin.weight.numel() < 64 * 64:
                    return False
                if "torchao" in type(lin.weight).__module__:
                    return False
                lin.to(device)
                quantize_(lin, ao_cfg)
                for p in lin.parameters(recurse=False):
                    p.requires_grad_(False)
                lin.to("cpu")
                return True

            for mod_name, mod in transformer.named_modules():
                # Never touch PEFT adapter linears (lora_A / lora_B) — they must stay trainable.
                if "lora_" in mod_name:
                    if isinstance(mod, torch.nn.Linear):
                        skipped_lora += 1
                    continue
                base = getattr(mod, "base_layer", None)
                if base is not None and isinstance(base, torch.nn.Linear):
                    if _woq_linear(base):
                        n_q += 1
                elif isinstance(mod, torch.nn.Linear):
                    if _woq_linear(mod):
                        n_q += 1
            flush()
            log(f"torchao WOQ applied to {n_q} Linear layers (skipped {skipped_lora} lora linears)")
        except Exception as e:
            log(f"WARNING: torchao quantize failed ({e}); training in bf16 may OOM on 24GB")
            use_quant = False

    # Ensure LoRA adapters remain trainable after WOQ / device moves.
    for n, p in transformer.named_parameters():
        if "lora_" in n:
            p.requires_grad_(True)

    if device.type == "cuda":
        transformer.to(device)
    else:
        transformer.to(device)

    transformer.train()
    flush()

    n_train = sum(1 for p in transformer.parameters() if p.requires_grad)
    if n_train == 0:
        raise RuntimeError(
            "No trainable parameters after LoRA/WOQ setup "
            "(LoRA adapters were likely quantized). Check target_modules / lora_ skip."
        )

    if device.type == "cuda":
        log(
            f"Transformer ready on GPU: "
            f"{torch.cuda.memory_allocated() / 1e9:.2f}GB alloc / "
            f"{torch.cuda.memory_reserved() / 1e9:.2f}GB reserved "
            f"({n_train} trainable params)"
        )

    # ------------------------------------------------------------------
    # Preload disk caches into CPU RAM (avoid per-step safetensors I/O)
    # ------------------------------------------------------------------
    log("Preloading latent/text caches into RAM…")
    ram_items: list[dict] = []
    for item in latent_index:
        if item["latent"] is None:
            raise RuntimeError("cache_latents_to_disk is required for this trainer path")
        lat = load_file(item["latent"])["latents"]
        cap = item["caption"]
        if cache_text:
            # Store both real caption embeds and empty-caption embeds for dropout
            t_real = load_file(text_paths[cap] if cap in text_paths else text_paths[""])
            t_empty = load_file(text_paths[""])
            ram_items.append(
                {
                    "latents": lat,
                    "prompt_embeds": t_real["prompt_embeds"],
                    "prompt_embeds_mask": t_real["prompt_embeds_mask"],
                    "empty_embeds": t_empty["prompt_embeds"],
                    "empty_mask": t_empty["prompt_embeds_mask"],
                }
            )
        else:
            ram_items.append({"latents": lat, "caption": cap})
    log(f"RAM cache ready: {len(ram_items)} samples")

    class CachedTrainDataset(Dataset):
        def __init__(self, items: list[dict], drop: float, use_text_cache: bool):
            self.items = items
            self.drop = drop
            self.use_text_cache = use_text_cache

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, idx: int):
            item = self.items[idx]
            if self.use_text_cache:
                use_empty = self.drop > 0 and random.random() < self.drop
                return {
                    "latents": item["latents"],
                    "prompt_embeds": item["empty_embeds"] if use_empty else item["prompt_embeds"],
                    "prompt_embeds_mask": item["empty_mask"] if use_empty else item["prompt_embeds_mask"],
                }
            caption = item["caption"]
            if self.drop > 0 and random.random() < self.drop:
                caption = ""
            return {"latents": item["latents"], "caption": caption}

    dataset = CachedTrainDataset(ram_items, dropout, cache_text)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    params = [p for p in transformer.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("optimizer got an empty parameter list (no requires_grad tensors)")
    if optimizer_name in ("adamw8bit", "adam8bit"):
        try:
            import bitsandbytes as bnb

            optimizer = bnb.optim.AdamW8bit(params, lr=lr)
        except Exception:
            log("bitsandbytes unavailable; falling back to AdamW")
            optimizer = torch.optim.AdamW(params, lr=lr)
    else:
        optimizer = torch.optim.AdamW(params, lr=lr)

    (out_dir / "captioer_train_config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    def save_lora(step: int | None = None) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        step_n = int(steps if step is None else step)
        path = out_dir / f"{safe_job_name(job_name)}_{format_step(step_n)}.safetensors"
        state = get_peft_model_state_dict(transformer)
        save_dtype = torch.float16 if (save_cfg.get("dtype") or "").startswith("float16") else None
        peft_tensors = {}
        for k, v in state.items():
            t = v.detach().cpu()
            if save_dtype is not None and t.is_floating_point():
                t = t.to(save_dtype)
            peft_tensors[k] = t
        # ComfyUI / StabilityMatrix expect diffusion_model.blocks.*.attn.wq keys.
        # Raw PEFT Diffusers keys (base_model.model.transformer_blocks.*.to_q) load as no-ops.
        tensors = convert_diffusers_krea2_lora_to_comfy(peft_tensors)
        save_file(tensors, str(path))
        log(f"saved LoRA (ComfyUI/diffusion_model keys) -> {path} ({len(tensors)} tensors)")
        return path

    global_step = 0
    data_iter = iter(loader)
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    import time as _time

    def do_sample(at_step: int) -> None:
        try:
            saved = run_midtrain_samples(
                step=at_step,
                prompts=sample_prompts,
                out_dir=out_dir,
                job_name=job_name,
                pipe=pipe,
                train_transformer=transformer,
                vae=vae,
                text_encoder=text_encoder,
                train_path=train_path,
                device=device,
                weight_dtype=weight_dtype,
                guidance_scale=sample_guidance,
                num_inference_steps=sample_steps_n,
                neg=sample_neg,
                trigger=trigger,
                low_vram=low_vram,
            )
            log(f"sample complete at step {at_step}: {len(saved)} image(s)")
        except Exception as e:
            log(f"WARNING: sampling failed at step {at_step}: {e}")
            import traceback

            log(traceback.format_exc())
            try:
                vae.to("cpu")
            except Exception:
                pass
            try:
                if text_encoder is not None:
                    text_encoder.to("cpu")
            except Exception:
                pass
            try:
                transformer.to(device)
                transformer.train()
            except Exception:
                pass
            flush()

    # Baseline sample at step 0 (before any training), unless skip_first_sample.
    if not disable_sampling and sample_prompts:
        if skip_first_sample:
            log("Skipping first sample at step 0 (skip_first_sample=true)")
        else:
            log("Generating baseline sample at step 0…")
            do_sample(0)

    log("Training loop (RAM caches; torchao WOQ base; transformer stays on GPU)…")
    step_t0 = _time.perf_counter()
    while global_step < steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        latents = batch["latents"].to(device=device, dtype=weight_dtype)
        # Cached as C F H W; DataLoader stacks → B C F H W
        if latents.ndim == 4:
            latents = latents.unsqueeze(2)

        if cache_text:
            prompt_embeds = batch["prompt_embeds"].to(device=device, dtype=weight_dtype)
            prompt_embeds_mask = batch["prompt_embeds_mask"].to(device=device)
        else:
            raise RuntimeError("cache_text_embeddings=true is required on <=24GB GPUs")

        latents = pack_latents(latents)
        image_seq = latents.shape[1]
        grid = int(math.isqrt(image_seq))
        if grid * grid != image_seq:
            raise RuntimeError(f"Non-square packed seq_len={image_seq}; expected square crop")
        position_ids = pipe.prepare_position_ids(
            int(prompt_embeds.shape[1]), grid, grid, device
        )

        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.rand(bsz, device=device, dtype=weight_dtype)
        t = timesteps.view(-1, *([1] * (latents.ndim - 1)))
        noisy = (1.0 - t) * latents + t * noise
        target = noise - latents

        pred = transformer(
            hidden_states=noisy,
            encoder_hidden_states=prompt_embeds,
            timestep=timesteps,
            position_ids=position_ids,
            encoder_attention_mask=prompt_embeds_mask,
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
        loss_v = float(loss.item() * grad_accum)
        progress(global_step, steps, loss_v)
        if global_step == 1 or global_step % 10 == 0:
            dt = _time.perf_counter() - step_t0
            per = dt / global_step
            log(f"step {global_step}/{steps} loss={loss_v:.6f} ({per:.2f}s/step avg)")

        if save_every > 0 and global_step % save_every == 0:
            save_lora(global_step)

        should_sample = (
            not disable_sampling
            and sample_every > 0
            and global_step >= max(sample_start_step, sample_every)
            and global_step % sample_every == 0
        )
        if should_sample:
            do_sample(global_step)

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
        import traceback

        err_text = f"{type(e).__name__}: {e}"
        tb = traceback.format_exc()
        log(f"ERROR: {err_text}")
        log(tb)
        print(f"CAPTIOER_TRAIN_ERROR message={err_text}", flush=True)
        raise SystemExit(1)
