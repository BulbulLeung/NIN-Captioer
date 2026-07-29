#!/usr/bin/env python3
"""
Captioer native Krea 2 LoRA trainer.

Train strategy mirrors AI-Toolkit (24GB path):
  1) Cache VAE latents to disk (only VAE on GPU)
  2) Cache text embeddings to disk (only text encoder on GPU), then unload TE
  3) Train DiT+LoRA on GPU; optional Layer offload streams transformer blocks
     CPU<->GPU by % (auto from free VRAM or manual). Low VRAM only stages TE/VAE.

Progress protocol (stdout):
  CAPTIOER_PROGRESS step=N total=T loss=X.XXXX
  CAPTIOER_LOSS_SPIKE step=N loss=X.XXXXXX path=...
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

# High-loss spike: current loss >= 1.0, OR >= 5× previous step loss.
LOSS_SPIKE_MULT = 5.0
LOSS_SPIKE_FLOOR = 1.0

# Must be set before the first CUDA allocator init.
# expandable_segments is unsupported on Windows and can worsen shared-RAM spill.
# Torch 2.9+ prefers PYTORCH_ALLOC_CONF; keep legacy name for older builds.
_alloc = (
    "max_split_size_mb:128"
    if sys.platform == "win32"
    else "expandable_segments:True"
)
if "PYTORCH_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_ALLOC_CONF"] = _alloc
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = _alloc

# Diffusers Krea2Transformer2DModel names -> original krea-ai / ComfyUI names.
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
_KREA2_ATTN_FROM_COMFY = {v: k for k, v in _KREA2_ATTN_TO_COMFY.items()}
_KREA2_FF_FROM_COMFY = {v: k for k, v in _KREA2_FF_TO_COMFY.items()}
_KREA2_STANDALONE_FROM_COMFY = {v: k for k, v in _KREA2_STANDALONE_TO_COMFY.items()}


def log(msg: str) -> None:
    # Windows consoles often use cp1252/charmap; avoid crashing on Unicode.
    text = str(msg)
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        safe = text.encode(enc, errors="replace").decode(enc, errors="replace")
        print(safe, flush=True)


def convert_diffusers_krea2_lora_to_comfy(state_dict: dict) -> dict:
    """Map PEFT/Diffusers Krea2 adapter keys to ComfyUI `diffusion_model.*` keys.

    Supports LoRA (`lora_A`/`lora_B` / `lora.down`/`lora.up`) and LyCORIS-style
    LoHa/LoKr suffixes (`hada_*`, `lokr_*`).
    """
    out: dict = {}
    skipped: list[str] = []

    adapter_re = re.compile(
        r"\.(?P<suf>"
        r"lora_[AB]\.weight|"
        r"lora\.(?:down|up)\.weight|"
        r"hada_[^.]+\.weight|"
        r"lokr_[^.]+\.weight|"
        r"alpha"
        r")$"
    )

    for key, tensor in state_dict.items():
        k = key
        for prefix in ("base_model.model.", "transformer.", "diffusion_model."):
            if k.startswith(prefix):
                k = k[len(prefix) :]

        m = adapter_re.search(k)
        if m is None:
            skipped.append(key)
            continue

        module = k[: m.start()]
        suf = m.group("suf")
        if suf == "lora.down.weight":
            suffix = ".lora_A.weight"
        elif suf == "lora.up.weight":
            suffix = ".lora_B.weight"
        elif suf == "alpha":
            suffix = ".alpha"
        else:
            suffix = f".{suf}"

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

    if not out:
        raise ValueError(
            f"Could not map any adapter keys to ComfyUI format; "
            f"examples: {skipped[:5]}"
        )
    if skipped:
        log(f"WARNING: skipped {len(skipped)} unmapped adapter key(s); examples: {skipped[:5]}")
    return out


def convert_comfy_krea2_lora_to_diffusers(state_dict: dict) -> dict:
    """Map ComfyUI `diffusion_model.*` adapter keys back to Diffusers module paths."""
    out: dict = {}
    skipped: list[str] = []

    adapter_re = re.compile(
        r"\.(?P<suf>"
        r"lora_[AB]\.weight|"
        r"lora\.(?:down|up)\.weight|"
        r"hada_[^.]+\.weight|"
        r"lokr_[^.]+\.weight|"
        r"alpha"
        r")$"
    )

    for key, tensor in state_dict.items():
        k = key
        if k.startswith("diffusion_model."):
            k = k[len("diffusion_model.") :]

        m = adapter_re.search(k)
        if m is None:
            skipped.append(key)
            continue

        module = k[: m.start()]
        suf = m.group("suf")
        if suf == "lora.down.weight":
            suffix = ".lora_A.weight"
        elif suf == "lora.up.weight":
            suffix = ".lora_B.weight"
        elif suf == "alpha":
            suffix = ".alpha"
        else:
            suffix = f".{suf}"

        diff_module: str | None = None
        m_attn = re.match(r"blocks\.(\d+)\.attn\.(.+)$", module)
        if m_attn and m_attn.group(2) in _KREA2_ATTN_FROM_COMFY:
            diff_module = (
                f"transformer_blocks.{m_attn.group(1)}.attn."
                f"{_KREA2_ATTN_FROM_COMFY[m_attn.group(2)]}"
            )
        if diff_module is None:
            m_ff = re.match(r"blocks\.(\d+)\.mlp\.(.+)$", module)
            if m_ff and m_ff.group(2) in _KREA2_FF_FROM_COMFY:
                diff_module = (
                    f"transformer_blocks.{m_ff.group(1)}."
                    f"{_KREA2_FF_FROM_COMFY[m_ff.group(2)]}"
                )
        if diff_module is None:
            m_tf = re.match(
                r"txtfusion\.(layerwise_blocks|refiner_blocks)\.(\d+)\.attn\.(.+)$",
                module,
            )
            if m_tf and m_tf.group(3) in _KREA2_ATTN_FROM_COMFY:
                diff_module = (
                    f"text_fusion.{m_tf.group(1)}.{m_tf.group(2)}.attn."
                    f"{_KREA2_ATTN_FROM_COMFY[m_tf.group(3)]}"
                )
        if diff_module is None:
            m_tf_ff = re.match(
                r"txtfusion\.(layerwise_blocks|refiner_blocks)\.(\d+)\.mlp\.(.+)$",
                module,
            )
            if m_tf_ff and m_tf_ff.group(3) in _KREA2_FF_FROM_COMFY:
                diff_module = (
                    f"text_fusion.{m_tf_ff.group(1)}.{m_tf_ff.group(2)}."
                    f"{_KREA2_FF_FROM_COMFY[m_tf_ff.group(3)]}"
                )
        if diff_module is None:
            diff_module = _KREA2_STANDALONE_FROM_COMFY.get(module)

        if diff_module is None:
            skipped.append(key)
            continue
        out[f"{diff_module}{suffix}"] = tensor

    if not out:
        raise ValueError(
            f"Could not map any ComfyUI adapter keys to Diffusers format; "
            f"examples: {skipped[:5]}"
        )
    if skipped:
        log(f"WARNING: skipped {len(skipped)} unmapped Comfy key(s); examples: {skipped[:5]}")
    return out


def parse_resume_step_from_path(path: Path, job_name: str) -> int | None:
    """Parse `{job}_{NNNNNN}.safetensors` step from a checkpoint path."""
    name = path.name
    prefix = f"{safe_job_name(job_name)}_"
    suffix = ".safetensors"
    if not (name.startswith(prefix) and name.endswith(suffix)):
        # Fallback: any trailing _######.safetensors
        m = re.search(r"_(\d{6})\.safetensors$", name)
        return int(m.group(1)) if m else None
    step_part = name[len(prefix) : -len(suffix)]
    if not re.fullmatch(r"\d{6}", step_part):
        return None
    return int(step_part)


def strip_adapter_key_prefixes(key: str) -> str:
    k = key
    for prefix in ("base_model.model.", "base_model.", "transformer.", "diffusion_model."):
        if k.startswith(prefix):
            k = k[len(prefix) :]
    return k


def align_adapter_state_to_peft(loaded: dict, peft_ref: dict) -> dict:
    """Match loaded adapter tensors onto current PEFT state_dict key names."""
    by_core: dict[str, object] = {}
    for k, v in loaded.items():
        by_core[strip_adapter_key_prefixes(k)] = v

    aligned: dict = {}
    missing: list[str] = []
    for ref_key in peft_ref.keys():
        core = strip_adapter_key_prefixes(ref_key)
        if core in by_core:
            aligned[ref_key] = by_core[core]
        else:
            missing.append(ref_key)

    if not aligned:
        raise ValueError(
            f"No overlapping adapter keys after resume load; "
            f"loaded={len(loaded)} peft={len(peft_ref)} "
            f"examples_loaded={[strip_adapter_key_prefixes(k) for k in list(loaded)[:3]]} "
            f"examples_peft={[strip_adapter_key_prefixes(k) for k in list(peft_ref)[:3]]}"
        )
    if missing:
        log(
            f"WARNING: resume missing {len(missing)}/{len(peft_ref)} adapter key(s); "
            f"examples: {missing[:5]}"
        )
    return aligned


def load_adapter_weights_for_resume(
    path: Path,
    transformer,
    *,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    load_file,
) -> int:
    """
    Load LoRA/LoHa/LoKr weights from a step checkpoint into the PEFT model.
    Returns number of tensors loaded.
    """
    raw = load_file(str(path))
    if not raw:
        raise ValueError(f"empty checkpoint: {path}")

    sample_key = next(iter(raw.keys()))
    if sample_key.startswith("diffusion_model.") or any(
        k.startswith("diffusion_model.") for k in raw.keys()
    ):
        mapped = convert_comfy_krea2_lora_to_diffusers(raw)
    else:
        mapped = {strip_adapter_key_prefixes(k): v for k, v in raw.items()}

    peft_ref = get_peft_model_state_dict(transformer)
    aligned = align_adapter_state_to_peft(mapped, peft_ref)
    set_peft_model_state_dict(transformer, aligned)
    return len(aligned)


def progress(step: int, total: int, loss: float) -> None:
    print(f"CAPTIOER_PROGRESS step={step} total={total} loss={loss:.6f}", flush=True)


def loss_spike(step: int, loss: float, path: Path | str) -> None:
    print(
        f"CAPTIOER_LOSS_SPIKE step={int(step)} loss={float(loss):.6f} path={path}",
        flush=True,
    )


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


def shuffle_caption_tokens(text: str) -> str:
    """Shuffle comma-separated tags, else whitespace tokens."""
    text = (text or "").strip()
    if not text:
        return text
    if "," in text:
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if len(parts) <= 1:
            return text
        random.shuffle(parts)
        return ", ".join(parts)
    parts = text.split()
    if len(parts) <= 1:
        return text
    random.shuffle(parts)
    return " ".join(parts)


def parse_resolutions(raw: object) -> list[int]:
    if isinstance(raw, list) and raw:
        out = [int(x) for x in raw if isinstance(x, (int, float)) and int(x) > 0]
        return out or [1024]
    if isinstance(raw, (int, float)) and int(raw) > 0:
        return [int(raw)]
    return [1024]


def parse_save_dtype(name: str):
    import torch

    n = (name or "fp16").lower()
    if n in ("bf16", "bfloat16"):
        return torch.bfloat16
    if n in ("fp32", "float32"):
        return torch.float32
    return torch.float16


def build_noise_pair(latents, scheduler_name: str, device, weight_dtype):
    """Return (noisy, timesteps_for_model, target) for the chosen training scheduler."""
    import torch

    noise = torch.randn_like(latents)
    bsz = latents.shape[0]
    name = (scheduler_name or "flowmatch").lower()

    if name == "ddpm":
        # Cosine-ish alpha_bar via t in (0,1); epsilon prediction.
        t = torch.rand(bsz, device=device, dtype=weight_dtype).clamp(1e-4, 1.0 - 1e-4)
        alpha_bar = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        shape = (-1, *([1] * (latents.ndim - 1)))
        a = alpha_bar.view(*shape)
        noisy = a.sqrt() * latents + (1.0 - a).sqrt() * noise
        return noisy, t, noise

    if name == "euler":
        # Flow in sigma space: x_t = (1-sigma)*x + sigma*noise, sigma in (0,1).
        sigma = torch.rand(bsz, device=device, dtype=weight_dtype).clamp(1e-4, 1.0 - 1e-4)
        shape = (-1, *([1] * (latents.ndim - 1)))
        s = sigma.view(*shape)
        noisy = (1.0 - s) * latents + s * noise
        target = noise - latents
        return noisy, sigma, target

    if name not in ("flowmatch", "flow_match", "flow-matching"):
        log(f"WARNING: unknown noise_scheduler={scheduler_name!r}; using flowmatch")

    timesteps = torch.rand(bsz, device=device, dtype=weight_dtype)
    t = timesteps.view(-1, *([1] * (latents.ndim - 1)))
    noisy = (1.0 - t) * latents + t * noise
    target = noise - latents
    return noisy, timesteps, target


class AdapterEma:
    """EMA over trainable adapter parameters."""

    def __init__(self, parameters, decay: float):
        import torch

        self.decay = float(decay)
        self.params = [p for p in parameters if p.requires_grad]
        self.shadow: dict[int, "torch.Tensor"] = {
            id(p): p.detach().float().clone() for p in self.params
        }
        self._backup: dict[int, "torch.Tensor"] = {}

    def update(self) -> None:
        d = self.decay
        for p in self.params:
            s = self.shadow[id(p)]
            s.mul_(d).add_(p.detach().float(), alpha=1.0 - d)

    def store(self) -> None:
        import torch

        self._backup = {id(p): p.data.detach().clone() for p in self.params}

    def copy_to(self) -> None:
        for p in self.params:
            p.data.copy_(self.shadow[id(p)].to(device=p.device, dtype=p.dtype))

    def restore(self) -> None:
        for p in self.params:
            p.data.copy_(self._backup[id(p)])
        self._backup = {}


def flush() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


# Safety margin for batch activations when deciding if VAE/TE can share VRAM with DiT.
VRAM_SAFETY_MARGIN_BYTES = 2 * 1024**3


def vram_free_bytes() -> int | None:
    """Return free VRAM bytes, or None if CUDA unavailable."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, _total = torch.cuda.mem_get_info()
        return int(free)
    except Exception:
        return None


def vram_log(tag: str) -> str:
    """Human-readable VRAM snapshot for logs."""
    try:
        import torch

        if not torch.cuda.is_available():
            return f"{tag}: cuda=unavailable"
        free, total = torch.cuda.mem_get_info()
        alloc = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        return (
            f"{tag}: free={free / 1e9:.2f}GB total={total / 1e9:.2f}GB "
            f"alloc={alloc / 1e9:.2f}GB reserved={reserved / 1e9:.2f}GB"
        )
    except Exception as e:
        return f"{tag}: vram_log_failed ({e})"


def estimate_module_bytes(mod) -> int:
    """Approximate parameter + buffer footprint of a module.

    torchao weight-only quantized tensors often still report floating
    ``element_size()`` (e.g. bf16=2) even though storage is int8 - detect
    that and count ~1 byte/weight plus scale overhead so VRAM budget checks
    stay meaningful after Quantize.
    """
    if mod is None:
        return 0
    total = 0
    try:
        for p in mod.parameters(recurse=True):
            total += _tensor_nbytes(p)
        for b in mod.buffers(recurse=True):
            total += _tensor_nbytes(b)
    except Exception:
        pass
    return total


def _tensor_nbytes(t) -> int:
    """Best-effort byte size for a tensor, including torchao quantized weights."""
    if t is None:
        return 0
    try:
        modname = type(t).__module__ or ""
        typename = type(t).__name__ or ""
        # torchao AffineQuantizedTensor / similar: logical dtype may stay bf16.
        if "torchao" in modname or "AffineQuantized" in typename or "QuantizedTensor" in typename:
            n = int(t.numel())
            # int8 WOQ ~= 1 byte/elem + small scale/zero-point overhead
            return n + max(n // 64, 256)
        if hasattr(t, "nbytes"):
            try:
                return int(t.nbytes)
            except Exception:
                pass
        return int(t.numel()) * int(t.element_size())
    except Exception:
        try:
            return int(t.numel()) * int(t.element_size())
        except Exception:
            return 0


def can_place_on_gpu(needed_bytes: int, margin_bytes: int = VRAM_SAFETY_MARGIN_BYTES) -> bool:
    free = vram_free_bytes()
    if free is None:
        return False
    return free >= int(needed_bytes) + int(margin_bytes)


def apply_weight_only_quantization(
    model,
    *,
    device,
    quant_mode: str,
    label: str,
    skip_adapter_names: bool = True,
) -> tuple[int, int, int, bool]:
    """Apply the selected torchao weight-only quantization to Linear layers."""
    import torch

    if model is None:
        return 0, 0, 0, False
    if quant_mode == "none":
        return estimate_module_bytes(model), estimate_module_bytes(model), 0, False

    before = estimate_module_bytes(model)
    n_q = 0
    skipped_adapter = 0
    changed = False

    try:
        from torchao.quantization.quant_api import (
            Float8WeightOnlyConfig,
            Int8WeightOnlyConfig,
            quantize_,
        )

        quant_label = "Int8WeightOnly"
        if quant_mode == "int8":
            try:
                # version=2 avoids deprecation warning; older torchao may lack it.
                ao_cfg = Int8WeightOnlyConfig(version=2)
            except TypeError:
                ao_cfg = Int8WeightOnlyConfig()
        elif quant_mode == "float8":
            try:
                ao_cfg = Float8WeightOnlyConfig(
                    weight_dtype=torch.float8_e4m3fn,
                    version=2,
                )
            except TypeError:
                ao_cfg = Float8WeightOnlyConfig(weight_dtype=torch.float8_e4m3fn)
            quant_label = "Float8WeightOnly[e4m3fn]"
        elif quant_mode == "qfloat8":
            qfloat8_dtype = getattr(torch, "float8_e4m3fnuz", None)
            if qfloat8_dtype is None:
                raise RuntimeError("torch.float8_e4m3fnuz is not available in this torch build")
            try:
                ao_cfg = Float8WeightOnlyConfig(weight_dtype=qfloat8_dtype, version=2)
            except TypeError:
                ao_cfg = Float8WeightOnlyConfig(weight_dtype=qfloat8_dtype)
            quant_label = "Float8WeightOnly[e4m3fnuz]"
        else:
            raise RuntimeError(f"unsupported quant_mode={quant_mode!r}")

        log(
            f"Quantizing {label} Linear weights with torchao {quant_label} "
            f"(mode={quant_mode}) "
            f"(pre-quant estimate={before / 1e9:.2f}GB)"
        )

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

        for mod_name, mod in model.named_modules():
            lower = mod_name.lower()
            if skip_adapter_names and any(x in lower for x in ("lora_", "hada_", "lokr_")):
                if isinstance(mod, torch.nn.Linear):
                    skipped_adapter += 1
                continue
            base = getattr(mod, "base_layer", None)
            if base is not None and isinstance(base, torch.nn.Linear):
                if _woq_linear(base):
                    n_q += 1
                    changed = True
            elif isinstance(mod, torch.nn.Linear):
                if _woq_linear(mod):
                    n_q += 1
                    changed = True
        flush()
    except Exception as e:
        log(
            f"WARNING: torchao quantize failed for {label} "
            f"(mode={quant_mode}, {e}); continuing without WOQ"
        )
        return before, before, 0, False

    after = estimate_module_bytes(model)
    log(
        f"torchao quant applied to {label}: {n_q} Linear layers "
        f"(mode={quant_mode}; skipped {skipped_adapter} adapter linears); "
        f"post-quant estimate={after / 1e9:.2f}GB "
        f"(was {before / 1e9:.2f}GB)"
    )
    if n_q == 0:
        log(
            f"WARNING: Quantize enabled for {label} (mode={quant_mode}) but 0 layers quantized; "
            "VRAM estimate will still look like full bf16"
        )
    return before, after, n_q, changed


# Soft hint for Auto offload % only (never overrides manual %).
def activation_reserve_bytes(max_res: int) -> int:
    if max_res >= 1536:
        return 20 * 1024**3
    if max_res >= 1024:
        return 16 * 1024**3
    if max_res >= 768:
        return 12 * 1024**3
    return 8 * 1024**3


def find_krea2_dit(root):
    """Return the real Krea2Transformer2DModel (not Peft proxies)."""
    for mod in root.modules():
        if type(mod).__name__ == "Krea2Transformer2DModel":
            return mod
    return None


def normalize_quantize_mode(raw) -> str:
    if isinstance(raw, bool):
        return "int8" if raw else "none"
    text = str(raw or "").strip().lower()
    if text in ("", "none", "-none-", "off", "false", "0"):
        return "none"
    if text in ("qfloat8", "float8", "int8"):
        return text
    if text in ("true", "1"):
        return "int8"
    return "none"


def enable_krea_gradient_checkpointing(root) -> tuple[bool, str]:
    """
    Bind checkpointing on the real Krea2 DiT.

    PeftModel.__getattr__ forwards attrs, so naive owner detection can think
    PeftModel owns the blocks while Krea2.forward never gets a working
    `_gradient_checkpointing_func`.
    """
    import torch
    from torch.utils.checkpoint import checkpoint as torch_checkpoint

    def _ckpt_func(module, *args):
        return torch_checkpoint(module.__call__, *args, use_reentrant=False)

    krea = find_krea2_dit(root)
    if krea is None:
        return False, "Krea2Transformer2DModel not found"

    if hasattr(krea, "enable_gradient_checkpointing"):
        try:
            krea.enable_gradient_checkpointing()
        except Exception:
            pass

    krea.gradient_checkpointing = True
    krea._gradient_checkpointing_func = _ckpt_func

    ok = bool(krea.gradient_checkpointing) and callable(
        getattr(krea, "_gradient_checkpointing_func", None)
    )
    n_blocks = len(getattr(krea, "transformer_blocks", []) or [])
    detail = (
        f"owner=Krea2Transformer2DModel flag={krea.gradient_checkpointing} "
        f"has_func={callable(getattr(krea, '_gradient_checkpointing_func', None))} "
        f"blocks={n_blocks}"
    )
    return ok, detail


_ATTN_CHUNK = 256


def probe_flash_attn() -> tuple[bool, str]:
    """Return (ok, detail) if flash-attn package can run a tiny CUDA op."""
    try:
        import torch
        from flash_attn import flash_attn_func

        if not torch.cuda.is_available():
            return False, "cuda unavailable"
        q = torch.randn(1, 32, 2, 64, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, 32, 2, 64, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(1, 32, 2, 64, device="cuda", dtype=torch.bfloat16)
        out = flash_attn_func(q, k, v, causal=False)
        if out.shape != q.shape:
            return False, f"unexpected shape {tuple(out.shape)}"
        ver = "unknown"
        try:
            import flash_attn

            ver = getattr(flash_attn, "__version__", None) or ver
        except Exception:
            pass
        if ver == "unknown":
            try:
                from importlib.metadata import version as _pkg_ver

                ver = _pkg_ver("flash-attn")
            except Exception:
                ver = "2.x"
        return True, f"flash-attn {ver}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _chunked_sdpa(query, key, value, attn_mask=None, *, chunk_size: int = _ATTN_CHUNK):
    """Q-chunked scaled_dot_product_attention (B, H, L, D). Caps score VRAM at O(chunk*L)."""
    import torch
    import torch.nn.functional as F

    enable_gqa = int(query.shape[1]) != int(key.shape[1])
    seq_q = int(query.shape[2])
    chunk = max(1, int(chunk_size))
    sdpa_kwargs = {"enable_gqa": enable_gqa}

    def _one(q_c, mask_c, use_gqa: bool):
        if use_gqa:
            return F.scaled_dot_product_attention(
                q_c, key, value, attn_mask=mask_c, **sdpa_kwargs
            )
        return F.scaled_dot_product_attention(q_c, key, value, attn_mask=mask_c)

    try:
        use_gqa = True
        if seq_q <= chunk:
            return _one(query, attn_mask, use_gqa)
        parts = []
        for i in range(0, seq_q, chunk):
            q_c = query[:, :, i : i + chunk, :]
            mask_c = attn_mask
            if attn_mask is not None and attn_mask.dim() == 4 and attn_mask.shape[-2] == seq_q:
                mask_c = attn_mask[:, :, i : i + chunk, :]
            parts.append(_one(q_c, mask_c, use_gqa))
        return torch.cat(parts, dim=2)
    except TypeError:
        use_gqa = False
        if seq_q <= chunk:
            return _one(query, attn_mask, use_gqa)
        parts = []
        for i in range(0, seq_q, chunk):
            q_c = query[:, :, i : i + chunk, :]
            mask_c = attn_mask
            if attn_mask is not None and attn_mask.dim() == 4 and attn_mask.shape[-2] == seq_q:
                mask_c = attn_mask[:, :, i : i + chunk, :]
            parts.append(_one(q_c, mask_c, use_gqa))
        return torch.cat(parts, dim=2)


class CaptioerFlashAttnProcessor:
    """Krea2 attention via flash-attn package (real FlashAttention-2 + autograd).

    PyTorch Windows wheels often report flash_built=False for SDPA; the separate
    ``flash-attn`` wheel still provides memory-efficient fwd/bwd for training.
    Layout: (B, L, H, D). GQA supported when H_q != H_kv.
    """

    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn,
        hidden_states,
        attention_mask=None,
        image_rotary_emb=None,
    ):
        import torch
        from diffusers.models.embeddings import apply_rotary_emb
        from flash_attn import flash_attn_func

        query = attn.to_q(hidden_states).unflatten(-1, (attn.num_heads, attn.head_dim))
        key = attn.to_k(hidden_states).unflatten(-1, (attn.num_kv_heads, attn.head_dim))
        value = attn.to_v(hidden_states).unflatten(-1, (attn.num_kv_heads, attn.head_dim))
        gate = attn.to_gate(hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        # flash_attn wants (B, L, H, D) contiguous.
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()

        if attention_mask is not None:
            # flash_attn_func has limited mask support; fall back to chunked SDPA.
            q_h = query.permute(0, 2, 1, 3)
            k_h = key.permute(0, 2, 1, 3)
            v_h = value.permute(0, 2, 1, 3)
            hidden = _chunked_sdpa(q_h, k_h, v_h, attention_mask)
            hidden = hidden.permute(0, 2, 1, 3)
        else:
            hidden = flash_attn_func(query, key, value, causal=False)

        hidden = hidden.flatten(2, 3)
        hidden = hidden * torch.sigmoid(gate)
        return attn.to_out[0](hidden)


class CaptioerChunkedAttnProcessor:
    """Q-chunked SDPA for Windows wheels without Flash Attention.

    Diffusers Krea2 passes Q/K/V as (B, L, H, D); SDPA needs (B, H, L, D).
    Chunking along Lq caps attention score memory at O(chunk * L) instead of O(L^2).
    """

    _attention_backend = None
    _parallel_config = None
    chunk_size = _ATTN_CHUNK

    def __call__(
        self,
        attn,
        hidden_states,
        attention_mask=None,
        image_rotary_emb=None,
    ):
        import torch
        from diffusers.models.embeddings import apply_rotary_emb

        query = attn.to_q(hidden_states).unflatten(-1, (attn.num_heads, attn.head_dim))
        key = attn.to_k(hidden_states).unflatten(-1, (attn.num_kv_heads, attn.head_dim))
        value = attn.to_v(hidden_states).unflatten(-1, (attn.num_kv_heads, attn.head_dim))
        gate = attn.to_gate(hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        # (B, L, H, D) -> (B, H, L, D)
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        hidden = _chunked_sdpa(
            query, key, value, attention_mask, chunk_size=int(self.chunk_size)
        )

        hidden = hidden.permute(0, 2, 1, 3).flatten(2, 3)
        hidden = hidden * torch.sigmoid(gate)
        return attn.to_out[0](hidden)


def apply_efficient_attention(root) -> tuple[str, int]:
    """
    Prefer flash-attn (real FA2 fwd+bwd), else Q-chunked SDPA.

    Returns (backend_name, modules_replaced).
    """
    flash_ok, flash_detail = probe_flash_attn()
    n = 0
    if flash_ok:
        for mod in root.modules():
            if type(mod).__name__ != "Krea2Attention":
                continue
            if hasattr(mod, "set_processor"):
                mod.set_processor(CaptioerFlashAttnProcessor())
                n += 1
        return f"flash-attn ({flash_detail})", n

    for mod in root.modules():
        if type(mod).__name__ != "Krea2Attention":
            continue
        if hasattr(mod, "set_processor"):
            mod.set_processor(CaptioerChunkedAttnProcessor())
            n += 1
    return f"q-chunked-sdpa (flash-attn unavailable: {flash_detail})", n

def apply_chunked_attention(root) -> int:
    """Replace Krea2 attn processors with Q-chunked SDPA. Returns count replaced."""
    _name, n = apply_efficient_attention(root)
    return n


def find_transformer_blocks(model) -> tuple[list, str]:
    """Return (blocks, attr_name) for Krea2 / Peft-wrapped DiT."""
    import torch

    krea = find_krea2_dit(model)
    if krea is not None:
        blocks = getattr(krea, "transformer_blocks", None)
        if isinstance(blocks, torch.nn.ModuleList) and len(blocks) >= 2:
            return list(blocks), "transformer_blocks"

    roots = [model]
    if hasattr(model, "get_base_model"):
        try:
            roots.insert(0, model.get_base_model())
        except Exception:
            pass
    for root in list(roots):
        if root is None:
            continue
        inner = getattr(root, "model", None)
        if inner is not None:
            roots.append(inner)
    for root in roots:
        if root is None:
            continue
        # Avoid Peft __getattr__ false positives: only use own ModuleList attrs.
        for attr in ("transformer_blocks", "blocks", "layers"):
            if attr not in getattr(type(root), "__dict__", {}):
                continue
            mod = getattr(root, attr, None)
            if isinstance(mod, torch.nn.ModuleList) and len(mod) >= 2:
                return list(mod), attr
    return [], ""


def compute_auto_offload_percent(
    dit_est: int,
    free_bytes: int | None,
    *,
    max_res: int = 1024,
    no_flash: bool = False,
    has_flash_attn: bool = False,
) -> int:
    """Suggest offload % from free VRAM. No forced floor."""
    reserve = auto_offload_reserve_bytes(
        max_res,
        no_flash=no_flash,
        has_flash_attn=has_flash_attn,
    )
    if not dit_est or free_bytes is None:
        return 0
    budget = max(0, int(free_bytes) - reserve)
    if budget >= dit_est:
        return 0
    return int(min(100, max(0, math.ceil((1.0 - budget / float(dit_est)) * 100.0))))


def auto_offload_reserve_bytes(
    max_res: int,
    no_flash: bool = False,
    has_flash_attn: bool = False,
) -> int:
    reserve = activation_reserve_bytes(max_res)
    # flash-attn / torch Flash keep activation peaks lower than MATH/chunked.
    if has_flash_attn or not no_flash:
        return reserve
    if max_res >= 1024:
        reserve = max(reserve, 20 * 1024**3)
    elif max_res >= 768:
        reserve = max(reserve, 15 * 1024**3)
    return reserve


def _install_train_block_swap(
    blocks: list, device, *, offload_after_forward: bool = False
) -> list:
    """
    Training-safe block swap (not accelerate inference hooks).

    - At most one swapped block on GPU at a time
    - Caller must move trainable params back to GPU before optimizer.step()
    """
    handles: list = []
    if not blocks:
        return handles

    def _others_to_cpu(except_idx: int) -> None:
        for j, other in enumerate(blocks):
            if j == except_idx:
                continue
            try:
                other.to("cpu", non_blocking=False)
            except Exception:
                pass

    for i, block in enumerate(blocks):
        block.to("cpu")

        def _pre_fwd(module, _inputs, idx=i):
            _others_to_cpu(idx)
            module.to(device, non_blocking=False)
            return None

        def _post_fwd(module, _inputs, output, idx=i):
            if offload_after_forward:
                module.to("cpu", non_blocking=False)
            return output

        def _pre_bwd(module, _grad_output, idx=i):
            _others_to_cpu(idx)
            module.to(device, non_blocking=False)
            return None

        def _post_bwd(module, _grad_input, _grad_output, idx=i):
            module.to("cpu", non_blocking=False)
            return None

        handles.append(block.register_forward_pre_hook(_pre_fwd))
        if offload_after_forward:
            handles.append(block.register_forward_hook(_post_fwd))
        if hasattr(block, "register_full_backward_pre_hook"):
            handles.append(block.register_full_backward_pre_hook(_pre_bwd))
        handles.append(block.register_full_backward_hook(_post_bwd))

    flush()
    return handles


def ensure_trainable_on_device(parameters, device) -> None:
    """Move trainable params (+grads) to device before optimizer.step()."""
    for p in parameters:
        if not p.requires_grad:
            continue
        if p.device != device:
            p.data = p.data.to(device, non_blocking=False)
        if p.grad is not None and p.grad.device != device:
            p.grad = p.grad.to(device, non_blocking=False)


def apply_layer_offload(
    transformer,
    *,
    percent: int,
    device,
    offload_after_forward: bool = False,
) -> dict:
    """
    Place DiT for training with optional block CPU offload.

    percent=0: full GPU residency
    percent>=1: offload exactly that fraction of blocks (no auto-bump)
    """
    pct = int(min(100, max(0, percent)))
    blocks, attr = find_transformer_blocks(transformer)
    n_blocks = len(blocks)
    import torch

    if pct <= 0 or n_blocks == 0:
        transformer.to(device)
        return {
            "mode": "resident",
            "percent": 0,
            "blocks_total": n_blocks,
            "blocks_offloaded": 0,
            "stream_all": False,
            "hook_handles": [],
        }

    n_off = int(math.ceil(n_blocks * pct / 100.0))
    n_off = min(max(n_off, 1), n_blocks)

    try:
        transformer.to("cpu")
    except Exception:
        pass
    flush()
    # Full move fits after WOQ (~13GB); then park swapped blocks on CPU.
    transformer.to(device)
    off = blocks[n_blocks - n_off :]
    for b in off:
        b.to("cpu")
    flush()
    handles = _install_train_block_swap(
        off, device, offload_after_forward=offload_after_forward
    )

    return {
        "mode": "block_swap",
        "percent": int(round(100.0 * n_off / n_blocks)),
        "blocks_total": n_blocks,
        "blocks_offloaded": n_off,
        "blocks_attr": attr,
        "stream_all": n_off >= n_blocks,
        "hook_handles": handles,
    }


def run_with_encoder_on_gpu(
    encoder,
    *,
    dit,
    device,
    weight_dtype,
    fn,
    force_stage: bool = False,
    label: str = "encoder",
):
    """
    Run `fn()` with `encoder` on GPU under a VRAM budget.

    If free VRAM can fit encoder + safety margin (and not force_stage), keep DiT resident.
    Otherwise: DiT->CPU -> encoder->GPU -> fn -> encoder->CPU -> DiT->GPU.
    Always returns encoder to CPU afterward when device is CUDA.
    """
    import torch

    if device.type != "cuda" or encoder is None:
        return fn()

    needed = estimate_module_bytes(encoder)
    free = vram_free_bytes() or 0
    place_with_dit = (not force_stage) and can_place_on_gpu(needed)

    if place_with_dit:
        log(
            f"VRAM budget: {label} with DiT resident "
            f"(need={needed / 1e9:.2f}GB free={free / 1e9:.2f}GB)"
        )
        encoder.to(device, dtype=weight_dtype)
        try:
            return fn()
        finally:
            encoder.to("cpu")
            flush()

    reason = "force_stage" if force_stage else "insufficient free VRAM"
    log(
        f"VRAM budget: staging offload for {label} ({reason}; "
        f"need={needed / 1e9:.2f}GB free={free / 1e9:.2f}GB margin="
        f"{VRAM_SAFETY_MARGIN_BYTES / 1e9:.1f}GB)"
    )
    was_training = bool(getattr(dit, "training", False)) if dit is not None else False
    if dit is not None:
        dit.to("cpu")
        flush()
    encoder.to(device, dtype=weight_dtype)
    try:
        return fn()
    finally:
        encoder.to("cpu")
        flush()
        if dit is not None:
            dit.to(device)
            if was_training:
                dit.train()
            flush()


def safe_job_name(name: str) -> str:
    import re

    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", (name or "job").strip()) or "job"
    return cleaned.rstrip(" .")


def format_step(step: int) -> str:
    return f"{int(step):06d}"


def list_step_lora_saves(out_dir: Path, job_name: str) -> list[Path]:
    """Step checkpoints named `{job}_{NNNNNN}.safetensors` (newest last)."""
    prefix = f"{safe_job_name(job_name)}_"
    suffix = ".safetensors"
    found: list[tuple[int, float, Path]] = []
    if not out_dir.is_dir():
        return []
    for p in out_dir.iterdir():
        if not p.is_file():
            continue
        name = p.name
        if not (name.startswith(prefix) and name.endswith(suffix)):
            continue
        step_part = name[len(prefix) : -len(suffix)]
        if not re.fullmatch(r"\d{6}", step_part):
            continue
        try:
            mtime = p.stat().st_mtime
        except OSError:
            mtime = 0.0
        found.append((int(step_part), mtime, p))
    found.sort(key=lambda t: (t[0], t[1]))
    return [t[2] for t in found]


def cleanup_old_step_saves(out_dir: Path, job_name: str, max_keep: int) -> list[Path]:
    """
    Keep only the newest `max_keep` step LoRA checkpoints under out_dir.
    Returns paths that were removed. max_keep <= 0 disables cleanup.
    """
    if max_keep <= 0:
        return []
    saves = list_step_lora_saves(out_dir, job_name)
    if len(saves) <= max_keep:
        return []
    to_remove = saves[:-max_keep]
    removed: list[Path] = []
    for path in to_remove:
        try:
            path.unlink()
            removed.append(path)
            log(f"removed old checkpoint -> {path}")
        except OSError as e:
            log(f"WARN: failed to remove old checkpoint {path}: {e}")
    return removed


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
    quant_mode: str = "none",
    low_vram: bool = True,
    auto_stage: bool = False,
    stream_offload: bool = False,
) -> list[str]:
    """Sample with in-memory Raw+LoRA.

    low_vram=True or auto_stage=True: sequential GPU residency - never keep DiT + TE + VAE
    on GPU together; DiT is moved to CPU during encode/decode.
    stream_offload=True: DiT uses accelerate cpu_offload hooks - do not bulk .to(cuda).
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

    stage_offload = bool(low_vram or auto_stage or stream_offload)
    if not stage_offload and device.type == "cuda":
        te_est = estimate_module_bytes(text_encoder)
        vae_est = estimate_module_bytes(vae)
        if not can_place_on_gpu(max(te_est, vae_est)):
            stage_offload = True
            log(
                f"VRAM budget: forcing sample staging "
                f"(TE={te_est / 1e9:.2f}GB VAE={vae_est / 1e9:.2f}GB; {vram_log('pre_sample')})"
            )

    was_training = bool(getattr(train_transformer, "training", False))
    train_transformer.eval()
    pipe.transformer = train_transformer

    te = text_encoder
    te_reloaded = False
    if te is None:
        log("Reloading text encoder from train base for sampling...")
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
        if quant_mode != "none" and device.type == "cuda":
            _te_before, _te_after, te_q, te_changed = apply_weight_only_quantization(
                te,
                device=device,
                quant_mode=quant_mode,
                label="text_encoder_reload",
                skip_adapter_names=True,
            )
            if not te_changed and te_q == 0:
                log(
                    f"Sampling reload: text encoder quantize produced no changes "
                    f"(mode={quant_mode})"
                )

    mode = (
        "stream offload"
        if stream_offload
        else ("staged offload" if stage_offload else "DiT resident")
    )
    log(f"Sampling at step {step} with train base (Raw+LoRA; {mode})...")
    log(vram_log("sample_start"))

    try:
        vae.to("cpu")
        te.to("cpu")
        pipe.vae = vae
        pipe.text_encoder = te
        if not stage_offload and not stream_offload:
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
            if stage_offload and not stream_offload:
                log("  stage encode: TE on GPU, DiT off...")
                train_transformer.to("cpu")
                flush()
            else:
                log("  stage encode: TE on GPU...")
            te.to(device, dtype=weight_dtype)
            pipe.text_encoder = te
            # torchao Float8WeightOnly .t() fails under inference_mode
            # ("Cannot set version_counter for inference tensor"); use no_grad.
            with torch.no_grad():
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
            log(f"  encode done ({_time.perf_counter() - step_t0:.1f}s) {vram_log('after_encode')}")

            # ---- Stage B: denoise ----
            log("  stage denoise: DiT...")
            if not stream_offload:
                train_transformer.to(device)
            pipe.transformer = train_transformer
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

            with torch.no_grad():
                try:
                    latents_out = pipe(
                        **call_kwargs,
                        callback_on_step_end=_on_step_end,
                    )
                except TypeError:
                    latents_out = pipe(**call_kwargs)
            latents = latents_out.images if hasattr(latents_out, "images") else latents_out[0]
            log(f"  denoise done ({_time.perf_counter() - step_t0:.1f}s) {vram_log('after_denoise')}")

            # ---- Stage C: VAE decode ----
            if stage_offload and not stream_offload:
                log("  stage decode: VAE on GPU, DiT off...")
                train_transformer.to("cpu")
                flush()
            else:
                log("  stage decode: VAE on GPU...")
            vae.to(device, dtype=weight_dtype)
            pipe.vae = vae
            with torch.no_grad():
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

            if not stream_offload:
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
            if not stream_offload:
                train_transformer.to(device)
        except Exception:
            pass
        flush()
        if was_training:
            train_transformer.train()
        log(f"Sample cleanup done; {vram_log('restored')}")

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
    if not isinstance(datasets, list) or not datasets:
        datasets = [{}]
    ema_cfg = train_cfg.get("ema_config") or {}

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

    resume_from_raw = str(cfg.get("resume_from") or "").strip()
    resume_path: Path | None = Path(resume_from_raw) if resume_from_raw else None
    resume_step: int | None = None
    if resume_path is not None:
        if not resume_path.is_file():
            log(f"ERROR: resume_from not found: {resume_path}")
            return 3
        resume_step = parse_resume_step_from_path(resume_path, job_name)
        if resume_step is None:
            log(f"ERROR: cannot parse step from resume_from filename: {resume_path.name}")
            return 3
    latent_cache_dir.mkdir(parents=True, exist_ok=True)
    text_cache_dir.mkdir(parents=True, exist_ok=True)

    # Shared dataset settings from the first entry with a folder path (else first entry).
    shared_ds = next(
        (d for d in datasets if isinstance(d, dict) and str(d.get("folder_path") or "").strip()),
        datasets[0] if isinstance(datasets[0], dict) else {},
    )
    if not isinstance(shared_ds, dict):
        shared_ds = {}

    caption_ext = shared_ds.get("caption_ext") or "txt"
    trigger = (cfg.get("trigger_word") or "").strip()
    dropout = float(shared_ds.get("caption_dropout_rate") or 0.0)
    shuffle_tokens = bool(shared_ds.get("shuffle_tokens", False))
    resolutions = parse_resolutions(shared_ds.get("resolution"))
    from ar_buckets import BUCKET_STEP, buckets_fingerprint, normalize_resolutions

    resolutions = normalize_resolutions(resolutions)
    bucket_step = BUCKET_STEP
    bucket_min_res = min(resolutions)
    max_res = max(resolutions)

    cache_latents = bool(shared_ds.get("cache_latents_to_disk", True))
    cache_text = bool(train_cfg.get("cache_text_embeddings", True))

    steps = int(train_cfg.get("steps") or 1000)
    batch_size = int(train_cfg.get("batch_size") or 1)
    grad_accum = int(train_cfg.get("gradient_accumulation_steps") or 1)
    lr = float(train_cfg.get("lr") or 1e-4)
    lr_scheduler_name = (train_cfg.get("lr_scheduler") or "constant_with_warmup").lower()
    if lr_scheduler_name in ("constant", "constant_with_warmup"):
        lr_scheduler_name = "constant_with_warmup"
    elif lr_scheduler_name != "cosine":
        log(f"WARNING: unknown lr_scheduler={lr_scheduler_name!r}; using constant_with_warmup")
        lr_scheduler_name = "constant_with_warmup"
    if "warmup_percent" in train_cfg:
        try:
            warmup_percent = float(train_cfg.get("warmup_percent"))
        except (TypeError, ValueError):
            warmup_percent = 5.0
    else:
        # Legacy absolute warmup_steps → treat 0 as off, else ~5%.
        try:
            legacy_wu = int(train_cfg.get("warmup_steps") or 0)
        except (TypeError, ValueError):
            legacy_wu = 0
        warmup_percent = 0.0 if legacy_wu <= 0 else 5.0
    warmup_percent = max(0.0, min(100.0, warmup_percent))
    dtype_name = (train_cfg.get("dtype") or "bf16").lower()
    optimizer_name = (train_cfg.get("optimizer") or "adamw8bit").lower()
    noise_scheduler = (train_cfg.get("noise_scheduler") or "flowmatch").lower()
    disable_sampling = bool(train_cfg.get("disable_sampling", True))
    skip_first_sample = bool(train_cfg.get("skip_first_sample", False))
    sample_every = int(sample_cfg.get("sample_every") or 0)
    sample_start_step = int(sample_cfg.get("sample_start_step") or 0)
    save_every = int(save_cfg.get("save_every") or 250)
    max_step_saves_to_keep = int(save_cfg.get("max_step_saves_to_keep") or 4)
    rank = int(net_cfg.get("linear") or 16)
    alpha = int(net_cfg.get("linear_alpha") or rank)
    network_type = (net_cfg.get("type") or "lora").lower()
    if network_type not in ("lora", "locon", "lokr"):
        log(f"WARNING: unknown network.type={network_type!r}; using lora")
        network_type = "lora"
    low_vram = bool(model_cfg.get("low_vram", False))
    quant_mode = normalize_quantize_mode(model_cfg.get("quantize", "none"))
    use_quant = quant_mode != "none"
    layer_offload = bool(model_cfg.get("layer_offload", False))
    try:
        layer_offload_percent_cfg = float(model_cfg.get("layer_offload_percent", 0))
    except (TypeError, ValueError):
        layer_offload_percent_cfg = 0.0
    layer_offload_percent_cfg = max(0.0, min(100.0, layer_offload_percent_cfg))
    # Legacy configs may still send layer_offload_mode; percent<=0 means Auto.
    legacy_mode = str(model_cfg.get("layer_offload_mode") or "").lower()
    if legacy_mode == "auto":
        layer_offload_percent_cfg = 0.0
    grad_ckpt = bool(train_cfg.get("gradient_checkpointing", True)) or low_vram
    use_ema = bool(ema_cfg.get("use_ema", False))
    ema_decay = float(ema_cfg.get("ema_decay") or 0.99)

    if resume_step is not None and resume_step >= steps:
        log(
            f"ERROR: resume step {resume_step} >= train.steps {steps}; "
            f"increase Steps before resuming"
        )
        return 3

    # Always-on AR buckets under enabled resolution tiers.
    ar_assign: list[tuple[int, int]] = []
    ar_counts: dict[tuple[int, int], int] = {}
    ar_tier_counts: dict[int, int] = {}
    ar_allow_up: list[bool] = []
    ar_forced_up = 0
    ar_fp = buckets_fingerprint(resolutions, bucket_step)

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

    log("Captioer Krea2 trainer starting")
    log(f"train_base={train_path}")
    log(f"output={out_dir}")
    if resume_path is not None and resume_step is not None:
        log(f"resume_from={resume_path} step={resume_step}")
    log(
        f"steps={steps} batch={batch_size} lr={lr} lr_scheduler={lr_scheduler_name} "
        f"warmup_percent={warmup_percent:g} rank={rank} alpha={alpha} "
        f"network={network_type} scheduler={noise_scheduler} optimizer={optimizer_name} "
        f"resolutions={resolutions} shuffle_tokens={shuffle_tokens} "
        f"ar_bucketing=always step={bucket_step} min_res={bucket_min_res} "
        f"cache_latents={cache_latents} cache_text={cache_text} "
        f"quantize={quant_mode} low_vram={low_vram} "
        f"layer_offload={layer_offload} "
        f"offload_pct={'auto' if (not layer_offload or layer_offload_percent_cfg <= 0) else int(round(layer_offload_percent_cfg))} "
        f"checkpoint={grad_ckpt} ema={use_ema} save_every={save_every} "
        f"max_step_saves_to_keep={max_step_saves_to_keep}"
    )
    if not cache_latents or not cache_text:
        log("WARNING: cache disabled - high VRAM path (encode every step)")

    try:
        import torch
        from PIL import Image
    except Exception as e:
        log(f"ERROR: failed to import torch/vision: {e}")
        return 3

    try:
        from diffusers import Krea2Pipeline
        from peft import LoraConfig, LoHaConfig, LoKrConfig, get_peft_model
        from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
        from safetensors.torch import load_file, save_file
    except Exception as e:
        log(
            "ERROR: need recent diffusers with Krea2Pipeline and peft. "
            "Try: pip install -r requirements.txt "
            'and/or pip install "git+https://github.com/huggingface/diffusers.git"'
        )
        log(f"detail: {e}")
        return 4

    samples: list[tuple[Path, str]] = []
    dataset_summaries: list[dict] = []
    for di, ds in enumerate(datasets):
        if not isinstance(ds, dict):
            continue
        folder_raw = str(ds.get("folder_path") or "").strip()
        if not folder_raw:
            continue
        folder = Path(folder_raw)
        try:
            reps = int(ds.get("num_repeats") or 1)
        except (TypeError, ValueError):
            reps = 1
        reps = max(1, reps)
        try:
            pairs = collect_samples(folder, caption_ext, trigger)
        except Exception as e:
            log(f"WARNING: dataset[{di}] {folder}: {e}")
            continue
        unique_n = len(pairs)
        for _ in range(reps):
            samples.extend(pairs)
        dataset_summaries.append(
            {
                "index": di,
                "folder": str(folder),
                "unique_images": unique_n,
                "num_repeats": reps,
                "effective_images": unique_n * reps,
            }
        )
        log(
            f"dataset[{di}] folder={folder} unique={unique_n} "
            f"repeats={reps} effective={unique_n * reps}"
        )

    log(
        f"loaded {len(samples)} effective image/caption pairs "
        f"from {len(dataset_summaries)} dataset(s)"
    )
    if not samples:
        log("ERROR: no training images found")
        return 3

    # Assign AR buckets after we know image list (need PIL sizes).
    bucket_indices: dict[tuple[int, int], list[int]] = {}
    from ar_buckets import assign_with_resolution_tiers, format_bucket_counts
    from PIL import Image as _PILImage

    sizes: list[tuple[int, int]] = []
    for path, _cap in samples:
        try:
            with _PILImage.open(path) as im:
                sizes.append(im.size)
        except Exception as e:
            log(f"WARNING: cannot read size for {path.name}: {e}; using 1024x1024")
            sizes.append((1024, 1024))
    (
        ar_assign,
        ar_counts,
        ar_tier_counts,
        ar_forced_up,
        ar_allow_up,
    ) = assign_with_resolution_tiers(sizes, resolutions, step=bucket_step)
    for i, b in enumerate(ar_assign):
        bucket_indices.setdefault(b, []).append(i)
    log(
        f"AR bucketing: resolutions={resolutions} min_res={bucket_min_res} "
        f"step={bucket_step} used_buckets={sum(1 for n in ar_counts.values() if n)} "
        f"forced_upscale={ar_forced_up} fingerprint={ar_fp}"
    )
    for t in sorted(ar_tier_counts):
        log(f"  tier {t}: {ar_tier_counts[t]}")
    for label, n in format_bucket_counts({b: c for b, c in ar_counts.items() if c}):
        log(f"  bucket {label}: {n}")
    try:
        stats_path = out_dir / "_bucket_stats.json"
        import json as _json

        stats_path.write_text(
            _json.dumps(
                {
                    "ok": True,
                    "resolutions": resolutions,
                    "min_res": bucket_min_res,
                    "step": bucket_step,
                    "fingerprint": ar_fp,
                    "image_count": len(samples),
                    "datasets": dataset_summaries,
                    "forced_upscale": ar_forced_up,
                    "tier_counts": {str(k): v for k, v in ar_tier_counts.items()},
                    "counts": {
                        f"{w}x{h}": n for (w, h), n in ar_counts.items() if n
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception as e:
        log(f"WARNING: failed to write _bucket_stats.json: {e}")
    # VRAM reserve uses longest side among used buckets
    used_sides = [max(w, h) for (w, h), n in ar_counts.items() if n > 0]
    if used_sides:
        max_res = max(used_sides)

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

    log("Loading Krea2Pipeline (components stay on CPU until needed)...")
    pipe = Krea2Pipeline.from_pretrained(train_path, torch_dtype=weight_dtype)
    transformer = pipe.transformer
    vae = pipe.vae
    text_encoder = getattr(pipe, "text_encoder", None) or getattr(pipe, "text_encoder_2", None)

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

    def encode_image_latents_wh(
        path: Path,
        tw: int,
        th: int,
        *,
        no_upscale: bool = True,
        random_crop: bool = False,
    ) -> torch.Tensor:
        """Encode one image to exact (tw, th) via AR cover+crop."""
        from ar_buckets import prepare_image_for_bucket
        from torchvision import transforms as _T

        img = Image.open(path).convert("RGB")
        prepared = prepare_image_for_bucket(
            img,
            tw,
            th,
            no_upscale=no_upscale,
            random_crop=random_crop,
        )
        pixel = _T.Compose(
            [
                _T.ToTensor(),
                _T.Normalize([0.5], [0.5]),
            ]
        )(prepared).unsqueeze(0).to(device, dtype=weight_dtype)
        if pixel.ndim == 4:
            pixel = pixel.unsqueeze(2)
        lat = vae.encode(pixel).latent_dist.sample()
        return normalize_latents(lat).squeeze(0).cpu()

    # Updated after DiT placement (encode helpers close over these).
    auto_stage = False
    dit_for_encode = None  # type: ignore[assignment]

    def ensure_text_cached(caption: str) -> tuple[torch.Tensor, torch.Tensor]:
        key = cache_key(caption, "txt_v1")
        out_file = text_cache_dir / f"{key}.safetensors"
        if out_file.is_file():
            data = load_file(str(out_file))
            return data["prompt_embeds"], data["prompt_embeds_mask"]
        if text_encoder is None:
            raise RuntimeError("text encoder required to encode captions")

        result: dict = {}

        def _encode():
            with torch.no_grad():
                embeds, mask = encode_prompt_on(device, [caption])
            result["pe"] = embeds.squeeze(0).cpu().contiguous()
            result["pm"] = mask.squeeze(0).cpu().contiguous()

        run_with_encoder_on_gpu(
            text_encoder,
            dit=dit_for_encode,
            device=device,
            weight_dtype=weight_dtype,
            fn=_encode,
            force_stage=auto_stage and dit_for_encode is not None,
            label="text_encoder",
        )
        pe, pm = result["pe"], result["pm"]
        save_file({"prompt_embeds": pe, "prompt_embeds_mask": pm}, str(out_file))
        return pe, pm

    # ------------------------------------------------------------------
    # Phase A: optional latent cache (always AR buckets)
    # ------------------------------------------------------------------
    latent_by_index: list = [None] * len(samples)
    if cache_latents:
        log(
            f"Caching AR latents to disk (fingerprint={ar_fp}, "
            f"{len(bucket_indices)} used buckets)..."
        )
        log(vram_log("pre_latent_cache"))
        vae_bytes = estimate_module_bytes(vae)
        if device.type == "cuda" and not can_place_on_gpu(vae_bytes):
            log(
                f"WARNING: VAE ({vae_bytes / 1e9:.2f}GB) + margin may exceed free VRAM; "
                f"attempting cache anyway ({vram_log('vae_fit')})"
            )

        def _latent_cache_body():
            if hasattr(vae, "enable_slicing"):
                vae.enable_slicing()
            if hasattr(vae, "enable_tiling"):
                vae.enable_tiling()
            with torch.no_grad():
                for i, (path, _cap) in enumerate(samples):
                    tw, th = ar_assign[i]
                    try:
                        mtime = path.stat().st_mtime_ns
                    except OSError:
                        mtime = 0
                    key = cache_key(
                        str(path.resolve()), tw, th, mtime, ar_fp, "lat_ar_v2"
                    )
                    out_file = latent_cache_dir / f"{key}.safetensors"
                    if not out_file.is_file():
                        lat = encode_image_latents_wh(
                            path,
                            tw,
                            th,
                            no_upscale=not ar_allow_up[i],
                            random_crop=False,
                        )
                        save_file({"latents": lat.contiguous()}, str(out_file))
                    else:
                        lat = load_file(str(out_file))["latents"]
                    latent_by_index[i] = lat
                    if (i + 1) % 10 == 0 or i + 1 == len(samples):
                        log(
                            f"  AR latent cache {i + 1}/{len(samples)} "
                            f"last={tw}x{th}"
                        )

        run_with_encoder_on_gpu(
            vae,
            dit=None,
            device=device,
            weight_dtype=weight_dtype,
            fn=_latent_cache_body,
            force_stage=False,
            label="vae_cache",
        )
        log("Latent cache complete; VAE moved to CPU")
        log(vram_log("post_latent_cache"))
    else:
        log("Skipping latent cache (on-the-fly VAE encode)")

    # ------------------------------------------------------------------
    # Phase B: optional text embedding cache (base captions + empty)
    # ------------------------------------------------------------------
    text_cache_ram: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    if cache_text:
        if text_encoder is None:
            raise RuntimeError("cache_text_embeddings requires a text encoder")
        log("Caching base text embeddings to disk...")
        log(vram_log("pre_text_cache"))

        def _text_cache_body():
            unique_caps = sorted({cap for _p, cap in samples} | {""})
            with torch.no_grad():
                for i, cap in enumerate(unique_caps):
                    # Disk hit path inside ensure_text_cached; miss encodes with TE already on GPU
                    # via nested run_with - avoid double-move by encoding inline here.
                    key = cache_key(cap, "txt_v1")
                    out_file = text_cache_dir / f"{key}.safetensors"
                    if out_file.is_file():
                        data = load_file(str(out_file))
                        pe, pm = data["prompt_embeds"], data["prompt_embeds_mask"]
                    else:
                        embeds, mask = encode_prompt_on(device, [cap])
                        pe = embeds.squeeze(0).cpu().contiguous()
                        pm = mask.squeeze(0).cpu().contiguous()
                        save_file(
                            {"prompt_embeds": pe, "prompt_embeds_mask": pm},
                            str(out_file),
                        )
                    text_cache_ram[cap] = (pe, pm)
                    if (i + 1) % 20 == 0 or i + 1 == len(unique_caps):
                        log(f"  text cache {i + 1}/{len(unique_caps)}")

        run_with_encoder_on_gpu(
            text_encoder,
            dit=None,
            device=device,
            weight_dtype=weight_dtype,
            fn=_text_cache_body,
            force_stage=False,
            label="text_encoder_cache",
        )
        try:
            pipe.text_encoder = text_encoder
        except Exception:
            pass
        flush()
        log("Text embedding cache complete; text encoder kept on CPU for sampling")
        log(vram_log("post_text_cache"))
    else:
        log("Skipping text cache (on-the-fly encode; TE stays on CPU until needed)")
        if text_encoder is not None:
            text_encoder.to("cpu")
        flush()

    # ------------------------------------------------------------------
    # Phase C: PEFT adapter + optional WOQ
    # ------------------------------------------------------------------
    log(f"Preparing transformer + {network_type}...")

    target_modules = [
        "to_q",
        "to_k",
        "to_v",
        "to_out.0",
        "proj_out",
        "linear_1",
        "linear_2",
    ]
    if network_type == "lokr":
        adapter_config = LoKrConfig(
            r=rank,
            alpha=alpha,
            target_modules=target_modules,
        )
    elif network_type == "locon":
        adapter_config = LoHaConfig(
            r=rank,
            alpha=alpha,
            target_modules=target_modules,
        )
    else:
        adapter_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )

    try:
        import peft.import_utils as _peft_iu
        import peft.tuners.lora.torchao as _peft_lora_tao

        if hasattr(_peft_iu.is_torchao_available, "cache_clear"):
            _peft_iu.is_torchao_available.cache_clear()
        _peft_iu.is_torchao_available = lambda: False  # type: ignore[assignment]
        _peft_lora_tao.is_torchao_available = lambda: False  # type: ignore[assignment]
    except Exception:
        pass

    transformer = get_peft_model(transformer, adapter_config)

    te_n_q = 0
    dit_est_before_quant = estimate_module_bytes(transformer)
    n_q = 0
    if use_quant and device.type == "cuda":
        te_before, te_after, te_n_q, te_changed = apply_weight_only_quantization(
            text_encoder,
            device=device,
            quant_mode=quant_mode,
            label="text_encoder",
            skip_adapter_names=True,
        )
        dit_est_before_quant, dit_est_after, n_q, dit_changed = apply_weight_only_quantization(
            transformer,
            device=device,
            quant_mode=quant_mode,
            label="transformer",
            skip_adapter_names=True,
        )
        if not te_changed and not dit_changed:
            use_quant = False
            te_n_q = 0
            n_q = 0
            dit_est_after = dit_est_before_quant
        elif te_changed and te_before > 0:
            log(
                f"VRAM budget: text encoder estimate={te_after / 1e9:.2f}GB "
                f"(was {te_before / 1e9:.2f}GB; layers={te_n_q})"
            )
    elif not use_quant:
        log("Quantize disabled (mode=none)")
        dit_est_after = dit_est_before_quant
    else:
        dit_est_after = dit_est_before_quant

    active_quant_mode = quant_mode if use_quant else "none"

    for n, p in transformer.named_parameters():
        lower = n.lower()
        if any(x in lower for x in ("lora_", "hada_", "lokr_")):
            p.requires_grad_(True)

    if resume_path is not None and resume_step is not None:
        try:
            n_loaded = load_adapter_weights_for_resume(
                resume_path,
                transformer,
                get_peft_model_state_dict=get_peft_model_state_dict,
                set_peft_model_state_dict=set_peft_model_state_dict,
                load_file=load_file,
            )
            log(
                f"Resuming from step {resume_step} -> {resume_path} "
                f"({n_loaded} tensors)"
            )
        except Exception as e:
            log(f"ERROR: failed to load resume checkpoint: {e}")
            import traceback

            log(traceback.format_exc())
            return 3

    dit_est = estimate_module_bytes(transformer)
    # Fallback: if weight-only quant ran but estimate barely moved (torchao may still
    # report logical bf16/float sizes), use a 1-byte-per-weight-style approximation.
    if use_quant and n_q > 0 and dit_est > dit_est_before_quant * 0.85:
        # Most DiT mass is Linear weights; int8/float8 weight-only formats are close
        # to 1 byte per stored weight plus small scale metadata.
        dit_est_adj = int(dit_est_before_quant * 0.55)
        log(
            f"VRAM budget: post-quant estimate still {dit_est / 1e9:.2f}GB "
            f"(torchao may misreport dtype for mode={quant_mode}); "
            f"using adjusted~={dit_est_adj / 1e9:.2f}GB"
        )
        dit_est = dit_est_adj

    log(vram_log("pre_dit_to_gpu"))
    log(
        f"VRAM budget: DiT estimate={dit_est / 1e9:.2f}GB "
        f"(quantize={active_quant_mode} layers={n_q}; TE layers={te_n_q})"
    )
    if device.type == "cuda":
        free_pre = vram_free_bytes() or 0
        if free_pre < dit_est:
            if use_quant and n_q > 0:
                log(
                    f"WARNING: DiT estimate ({dit_est / 1e9:.2f}GB) > free VRAM "
                    f"({free_pre / 1e9:.2f}GB) even after Quantize; attempting load anyway"
                )
            else:
                log(
                    f"ERROR: DiT (~{dit_est / 1e9:.2f}GB) does not fit free VRAM "
                    f"({free_pre / 1e9:.2f}GB). Enable Quantize in Advanced settings "
                    f"(bf16 alone is ~26GB for this DiT). Not auto-enabling quantize."
                )
                return 6

    try:
        transformer.to("cpu")
        flush()
    except Exception:
        pass

    # Probe Flash / flash-attn before Auto offload % (no-Flash needs more free VRAM at 1024).
    flash_built = None
    has_flash_attn = False
    flash_attn_detail = ""
    if device.type == "cuda":
        try:
            flash_built = getattr(
                torch.backends.cuda, "is_flash_attention_available", lambda: None
            )()
        except Exception:
            flash_built = None
        has_flash_attn, flash_attn_detail = probe_flash_attn()
        if has_flash_attn:
            log(f"flash-attn probe OK: {flash_attn_detail}")
        else:
            log(f"flash-attn probe skipped: {flash_attn_detail}")

    dit_stream_offload = False
    offload_info: dict = {
        "mode": "resident",
        "percent": 0,
        "blocks_total": 0,
        "blocks_offloaded": 0,
        "stream_all": False,
    }
    if device.type == "cuda" and layer_offload:
        free_for_off = vram_free_bytes()
        if layer_offload_percent_cfg <= 0:
            off_pct = compute_auto_offload_percent(
                dit_est,
                free_for_off,
                max_res=max_res,
                no_flash=(flash_built is False),
                has_flash_attn=has_flash_attn,
            )
            log(
                f"Layer offload: auto percent={off_pct} "
                f"(dit~={dit_est / 1e9:.2f}GB free={((free_for_off or 0) / 1e9):.2f}GB "
                f"reserve={auto_offload_reserve_bytes(max_res, flash_built is False, has_flash_attn=has_flash_attn) / 1e9:.1f}GB "
                f"no_flash={flash_built is False} flash_attn={has_flash_attn})"
            )
        else:
            off_pct = int(round(layer_offload_percent_cfg))
            log(f"Layer offload: manual percent={off_pct}")
        try:
            offload_info = apply_layer_offload(
                transformer,
                percent=off_pct,
                device=device,
                offload_after_forward=False,
            )
            dit_stream_offload = bool(offload_info.get("stream_all")) or (
                int(offload_info.get("blocks_offloaded") or 0) > 0
            )
            log(
                f"Layer offload applied: mode={offload_info.get('mode')} "
                f"percent={offload_info.get('percent')} "
                f"blocks={offload_info.get('blocks_offloaded')}/"
                f"{offload_info.get('blocks_total')} "
                f"stream_all={offload_info.get('stream_all')}"
            )
            log(vram_log("post_layer_offload"))
        except Exception as e:
            log(
                f"WARNING: layer offload failed ({e}); "
                f"falling back to full GPU DiT (may OOM)"
            )
            try:
                transformer.to(device)
            except torch.cuda.OutOfMemoryError as e2:
                log(
                    f"ERROR: CUDA OOM moving DiT to GPU ({e2}). "
                    f"{vram_log('oom')}"
                )
                return 6
            dit_stream_offload = False
            offload_info = {
                "mode": "resident_fallback",
                "percent": 0,
                "stream_all": False,
            }
    else:
        try:
            transformer.to(device)
        except torch.cuda.OutOfMemoryError as e:
            log(
                f"ERROR: CUDA OOM moving DiT to GPU ({e}). "
                f"Enable Quantize and/or Layer offload; estimate was {dit_est / 1e9:.2f}GB, "
                f"{vram_log('oom')}"
            )
            return 6

    transformer.train()
    flush()
    # When any blocks stream, do not bulk-move DiT for TE/VAE staging.
    dit_for_encode = None if dit_stream_offload else transformer

    # Gradient checkpointing AFTER Peft wrap — bind on real Krea2 DiT only.
    if grad_ckpt:
        try:
            if hasattr(transformer, "enable_input_require_grads"):
                transformer.enable_input_require_grads()
            ok, detail = enable_krea_gradient_checkpointing(transformer)
            log(f"Gradient checkpointing={'ON' if ok else 'FAILED'} ({detail})")
            if not ok:
                log(
                    "ERROR: gradient checkpointing did not bind on Krea2 DiT. "
                    "768/1024 will OOM from activation memory. Aborting."
                )
                return 6
        except Exception as e:
            log(f"ERROR: gradient checkpointing setup failed: {e}")
            return 6

    # Prefer memory-efficient attention; Windows torch wheels often lack Flash.
    if device.type == "cuda":
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
            flash_ok = bool(torch.backends.cuda.flash_sdp_enabled())
            mem_ok = bool(torch.backends.cuda.mem_efficient_sdp_enabled())
            if flash_built is None:
                flash_built = getattr(
                    torch.backends.cuda, "is_flash_attention_available", lambda: None
                )()
            log(
                f"SDPA: flash_enabled={flash_ok} mem_efficient={mem_ok} "
                f"flash_built={flash_built}"
            )
            # Prefer flash-attn / Q-chunked when torch SDPA Flash is missing.
            if flash_built is False or has_flash_attn:
                attn_name, n_attn = apply_efficient_attention(transformer)
                extra = " (FA2 fwd+bwd)" if attn_name.startswith("flash-attn") else ""
                log(
                    f"Attention backend: {attn_name} on {n_attn} Krea2Attention modules"
                    f"{extra}"
                )
        except Exception as e:
            log(f"WARNING: SDPA backend tweak failed: {e}")

    n_train = sum(1 for p in transformer.parameters() if p.requires_grad)
    if n_train == 0:
        raise RuntimeError(
            "No trainable parameters after adapter/WOQ setup. "
            "Check target_modules / adapter skip."
        )

    vae_est = estimate_module_bytes(vae)
    te_est = estimate_module_bytes(text_encoder)
    free_after = vram_free_bytes()
    if device.type == "cuda":
        log(
            f"Transformer ready: "
            f"{torch.cuda.memory_allocated() / 1e9:.2f}GB alloc / "
            f"{torch.cuda.memory_reserved() / 1e9:.2f}GB reserved "
            f"({n_train} trainable params; stream_offload={dit_stream_offload})"
        )
        log(vram_log("post_dit_place"))
        log(
            f"VRAM budget: VAE~={vae_est / 1e9:.2f}GB TE~={te_est / 1e9:.2f}GB "
            f"margin={VRAM_SAFETY_MARGIN_BYTES / 1e9:.1f}GB"
        )
        auto_stage = (not can_place_on_gpu(max(vae_est, te_est))) or (
            free_after is not None and free_after < VRAM_SAFETY_MARGIN_BYTES
        )
        if low_vram or dit_stream_offload:
            auto_stage = True
        log(
            f"VRAM budget: auto_stage={auto_stage} "
            f"(low_vram={low_vram}; layer_offload={dit_stream_offload} "
            f"pct={offload_info.get('percent', 0)})"
        )
    else:
        auto_stage = False

    params = [p for p in transformer.parameters() if p.requires_grad]
    # Block-swap pins trainable (LoRA) weights on GPU; adamw8bit is OK.
    if optimizer_name in ("adamw8bit", "adam8bit"):
        try:
            import bitsandbytes as bnb

            optimizer = bnb.optim.AdamW8bit(params, lr=lr)
        except Exception as e:
            log(f"ERROR: adamw8bit requires bitsandbytes ({e})")
            return 5
    elif optimizer_name in ("adamw", "adam"):
        optimizer = torch.optim.AdamW(params, lr=lr)
    elif optimizer_name == "prodigy":
        try:
            from prodigyopt import Prodigy

            optimizer = Prodigy(params, lr=lr)
        except Exception as e:
            log(f"ERROR: prodigy requires prodigyopt ({e})")
            return 5
    elif optimizer_name == "adafactor":
        try:
            from transformers.optimization import Adafactor

            optimizer = Adafactor(
                params,
                lr=lr,
                scale_parameter=False,
                relative_step=False,
            )
        except Exception as e:
            log(f"ERROR: adafactor unavailable ({e})")
            return 5
    else:
        log(f"ERROR: unsupported optimizer={optimizer_name!r}")
        return 5

    try:
        from transformers import get_scheduler
    except Exception as e:
        log(f"ERROR: transformers.get_scheduler unavailable ({e})")
        return 5

    num_training_steps = max(1, (steps + grad_accum - 1) // grad_accum)
    if warmup_percent <= 0:
        num_warmup = 0
    else:
        num_warmup = int(round(num_training_steps * (warmup_percent / 100.0)))
        num_warmup = min(num_warmup, max(0, num_training_steps - 1))
    lr_scheduler = get_scheduler(
        name=lr_scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=num_warmup,
        num_training_steps=num_training_steps,
    )
    if resume_step is not None and resume_step > 0:
        resume_updates = resume_step // grad_accum
        for _ in range(resume_updates):
            lr_scheduler.step()
        log(
            f"LR scheduler={lr_scheduler_name} warmup={num_warmup} "
            f"({warmup_percent:g}% of {num_training_steps} updates) "
            f"resume_fast_forward={resume_updates}"
        )
    else:
        log(
            f"LR scheduler={lr_scheduler_name} warmup={num_warmup} "
            f"({warmup_percent:g}% of {num_training_steps} updates)"
        )

    ema: AdapterEma | None = None
    if use_ema:
        ema = AdapterEma(params, ema_decay)
        log(f"EMA enabled decay={ema_decay}")

    (out_dir / "captioer_train_config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    def prepare_caption(raw_cap: str) -> str:
        cap = raw_cap
        if shuffle_tokens:
            cap = shuffle_caption_tokens(cap)
        if dropout > 0 and random.random() < dropout:
            cap = ""
        return cap

    def get_text_embeds(caption: str) -> tuple[torch.Tensor, torch.Tensor]:
        if cache_text and not shuffle_tokens and caption in text_cache_ram:
            return text_cache_ram[caption]
        if cache_text:
            if caption in text_cache_ram:
                return text_cache_ram[caption]
            pe, pm = ensure_text_cached(caption)
            text_cache_ram[caption] = (pe, pm)
            return pe, pm
        # live encode - never leave TE resident on GPU
        if text_encoder is None:
            raise RuntimeError("text encoder required when cache_text_embeddings=false")
        result: dict = {}

        def _live():
            with torch.no_grad():
                embeds, mask = encode_prompt_on(device, [caption])
            result["pe"] = embeds.squeeze(0).cpu()
            result["pm"] = mask.squeeze(0).cpu()

        run_with_encoder_on_gpu(
            text_encoder,
            dit=dit_for_encode,
            device=device,
            weight_dtype=weight_dtype,
            fn=_live,
            force_stage=auto_stage and dit_for_encode is not None,
            label="text_encoder_live",
        )
        return result["pe"], result["pm"]

    def sample_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, list[Path]]:
        """Return latents, embeds, mask, latent_h, latent_w, batch image paths."""
        n = min(batch_size, len(samples))

        eligible = [
            (b, idxs) for b, idxs in bucket_indices.items() if len(idxs) >= n
        ]
        if not eligible:
            raise RuntimeError(
                f"AR bucketing: no bucket has >= batch_size ({n}) images. "
                f"Lower batch size or add more images."
            )
        weights = [len(idxs) for _b, idxs in eligible]
        pick_i = random.choices(range(len(eligible)), weights=weights, k=1)[0]
        (tw, th), pool = eligible[pick_i]
        idxs = random.sample(pool, n) if n < len(pool) else list(pool[:n])
        batch_paths = [samples[i][0] for i in idxs]

        lats = []
        embeds_list = []
        masks_list = []

        def _encode_ar():
            if hasattr(vae, "enable_slicing"):
                vae.enable_slicing()
            if hasattr(vae, "enable_tiling"):
                vae.enable_tiling()
            out = []
            for i in idxs:
                path, _raw_cap = samples[i]
                out.append(
                    encode_image_latents_wh(
                        path,
                        tw,
                        th,
                        no_upscale=not ar_allow_up[i],
                        random_crop=True,
                    )
                )
            return out

        with torch.no_grad():
            if cache_latents:
                for i in idxs:
                    lats.append(latent_by_index[i])
            else:
                batch_lats = run_with_encoder_on_gpu(
                    vae,
                    dit=dit_for_encode,
                    device=device,
                    weight_dtype=weight_dtype,
                    fn=_encode_ar,
                    force_stage=auto_stage and dit_for_encode is not None,
                    label="vae_live_ar",
                )
                lats.extend(batch_lats)

            for i in idxs:
                _path, raw_cap = samples[i]
                cap = prepare_caption(raw_cap)
                pe, pm = get_text_embeds(cap)
                embeds_list.append(pe)
                masks_list.append(pm)

        latents = torch.stack(lats, dim=0)
        lat0 = latents[0]
        lh, lw = int(lat0.shape[-2]), int(lat0.shape[-1])
        prompt_embeds = torch.stack(embeds_list, dim=0)
        prompt_embeds_mask = torch.stack(masks_list, dim=0)
        return latents, prompt_embeds, prompt_embeds_mask, lh, lw, batch_paths

    def save_adapter(step: int | None = None) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        step_n = int(steps if step is None else step)
        path = out_dir / f"{safe_job_name(job_name)}_{format_step(step_n)}.safetensors"

        if ema is not None:
            ema.store()
            ema.copy_to()
            log(f"saving EMA adapter weights at step {step_n}")

        state = get_peft_model_state_dict(transformer)
        save_dtype = parse_save_dtype(str(save_cfg.get("dtype") or "fp16"))
        peft_tensors = {}
        for k, v in state.items():
            t = v.detach().cpu()
            if t.is_floating_point():
                t = t.to(save_dtype)
            peft_tensors[k] = t

        try:
            tensors = convert_diffusers_krea2_lora_to_comfy(peft_tensors)
        except Exception as e:
            log(f"WARNING: Comfy key convert failed ({e}); saving PEFT keys")
            tensors = peft_tensors

        save_file(tensors, str(path))
        log(
            f"saved {network_type} ({save_dtype}) -> {path} "
            f"({len(tensors)} tensors)"
        )
        cleanup_old_step_saves(out_dir, job_name, max_step_saves_to_keep)

        if ema is not None:
            ema.restore()
        return path

    global_step = int(resume_step or 0)
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    prev_loss_v: float | None = None
    import time as _time

    if global_step > 0:
        progress(global_step, steps, 0.0)
        log(f"training loop starts at step {global_step}/{steps}")

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
                quant_mode=active_quant_mode,
                low_vram=low_vram,
                auto_stage=auto_stage,
                stream_offload=dit_stream_offload,
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
                if not dit_stream_offload:
                    transformer.to(device)
                transformer.train()
            except Exception:
                pass
            flush()

    if not disable_sampling and sample_prompts:
        if skip_first_sample:
            log("Skipping first sample at step 0 (skip_first_sample=true)")
        else:
            log("Generating baseline sample at step 0...")
            do_sample(0)

    log("Training loop...")
    step_t0 = _time.perf_counter()
    if dit_stream_offload:
        log(
            f"Note: layer offload active (block_swap percent={offload_info.get('percent')}); "
            "slower than full GPU residency but lower peak VRAM/RAM"
        )

    while global_step < steps:
        if device.type == "cuda" and (low_vram or auto_stage or dit_stream_offload):
            flush()

        try:
            (
                latents,
                prompt_embeds,
                prompt_embeds_mask,
                lat_h,
                lat_w,
                batch_paths,
            ) = sample_batch()
            latents = latents.to(device=device, dtype=weight_dtype)
            if latents.ndim == 4:
                latents = latents.unsqueeze(2)
            prompt_embeds = prompt_embeds.to(device=device, dtype=weight_dtype)
            prompt_embeds_mask = prompt_embeds_mask.to(device=device)

            latents = pack_latents(latents)
            patch = int(getattr(pipe, "patch_size", None) or 2)
            grid_h = int(lat_h) // patch
            grid_w = int(lat_w) // patch
            image_seq = latents.shape[1]
            if grid_h * grid_w != image_seq:
                raise RuntimeError(
                    f"Packed seq_len={image_seq} != grid {grid_h}x{grid_w} "
                    f"(latent {lat_h}x{lat_w}, patch={patch})"
                )

            # batch=1: trim text padding BEFORE position_ids so rotary/seq lengths match.
            attn_mask = prompt_embeds_mask
            if (
                prompt_embeds_mask is not None
                and prompt_embeds_mask.shape[0] == 1
                and prompt_embeds_mask.dtype == torch.bool
            ):
                n_tok = int(prompt_embeds_mask[0].sum().item())
                if 0 < n_tok < int(prompt_embeds.shape[1]):
                    prompt_embeds = prompt_embeds[:, :n_tok].contiguous()
                    attn_mask = None
                elif n_tok == int(prompt_embeds.shape[1]):
                    attn_mask = None

            position_ids = pipe.prepare_position_ids(
                int(prompt_embeds.shape[1]), grid_h, grid_w, device
            )
            noisy, timesteps, target = build_noise_pair(
                latents, noise_scheduler, device, weight_dtype
            )
            # Free pre-noise latents graph input alias if safe — keep noisy/target only.
            del latents

            if device.type == "cuda" and global_step == 0:
                torch.cuda.reset_peak_memory_stats()

            pred = transformer(
                hidden_states=noisy,
                encoder_hidden_states=prompt_embeds,
                timestep=timesteps,
                position_ids=position_ids,
                encoder_attention_mask=attn_mask,
                return_dict=False,
            )[0]

            # Per-sample MSE (equiv. to mean reduction for the training scalar).
            per = torch.nn.functional.mse_loss(pred, target, reduction="none")
            per_sample = per.reshape(pred.shape[0], -1).mean(dim=1)
            loss = per_sample.mean()
            loss = loss / grad_accum
            loss.backward()
            running_loss += loss.item() * grad_accum

            per_sample_cpu = per_sample.detach().float().cpu()
            del pred, noisy, target, prompt_embeds, prompt_embeds_mask, position_ids, timesteps
            del per, per_sample
            if attn_mask is not None:
                del attn_mask

            if (global_step + 1) % grad_accum == 0:
                if dit_stream_offload:
                    ensure_trainable_on_device(params, device)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update()
                n_off_keep = int(offload_info.get("blocks_offloaded") or 0)
                if dit_stream_offload and n_off_keep > 0:
                    # After stepping LoRA on GPU, park swapped blocks fully on CPU again.
                    swap_blocks = find_transformer_blocks(transformer)[0][-n_off_keep:]
                    for b in swap_blocks:
                        try:
                            b.to("cpu")
                        except Exception:
                            pass

            if device.type == "cuda" and dit_stream_offload:
                flush()

            global_step += 1
            loss_v = float(loss.item() * grad_accum)
            # loss tensor may still be alive; item() already taken above — re-get from loss_v
            progress(global_step, steps, loss_v)

            is_spike = False
            if loss_v >= LOSS_SPIKE_FLOOR:
                is_spike = True
            elif (
                prev_loss_v is not None
                and prev_loss_v > 0
                and loss_v >= prev_loss_v * LOSS_SPIKE_MULT
            ):
                is_spike = True

            if is_spike and batch_paths:
                worst_i = int(per_sample_cpu.argmax().item())
                worst_i = max(0, min(worst_i, len(batch_paths) - 1))
                loss_spike(global_step, loss_v, batch_paths[worst_i])

            prev_loss_v = loss_v

            if global_step == 1 or global_step % 10 == 0:
                dt = _time.perf_counter() - step_t0
                per = dt / global_step
                peak = ""
                if device.type == "cuda":
                    try:
                        peak = (
                            f" peak={torch.cuda.max_memory_allocated() / 1e9:.2f}GB"
                        )
                    except Exception:
                        peak = ""
                log(
                    f"step {global_step}/{steps} loss={loss_v:.6f} "
                    f"({per:.2f}s/step avg) {vram_log('step')}{peak}"
                )

            if save_every > 0 and global_step % save_every == 0:
                save_adapter(global_step)

            should_sample = (
                not disable_sampling
                and sample_every > 0
                and global_step >= max(sample_start_step, sample_every)
                and global_step % sample_every == 0
            )
            if should_sample:
                do_sample(global_step)

        except torch.cuda.OutOfMemoryError as e:
            log(f"ERROR: CUDA OOM during training step {global_step + 1}: {e}")
            log(vram_log("oom_step"))
            log(
                "Hints: ensure Gradient checkpointing is ON (activation memory, not "
                "Quantize/offload). Try batch_size=1, disable mid-train sampling."
            )
            flush()
            return 7

    final_path = save_adapter(None)
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
