"""INT8 ConvRot checkpoints using Comfy Kitchen's official W8A8 API.

Only DiT transformer-block Linear layers are quantized. Attention semantics,
normalization, embeddings, the LQ projector and decoders remain unchanged.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Callable

import torch
from torch import nn

FORMAT = "flashvsr_plus_int8_convrot_v1"
LINEAR_ROW_CHUNK = 8192
_COMMIT_LOCK = threading.Lock()


def kitchen():
    try:
        import comfy_kitchen as ck
    except (ImportError, OSError) as exc:
        raise RuntimeError("INT8 ConvRot needs comfy-kitchen>=0.2.35 in the ComfyUI interpreter") from exc
    if not callable(getattr(ck, "int8_linear", None)):
        raise RuntimeError("This comfy-kitchen version does not provide int8_linear")
    return ck


class ConvRotLinear(nn.Module):
    def __init__(self, weight: torch.Tensor, scale: torch.Tensor,
                 bias: torch.Tensor | None, group_size: int = 256):
        super().__init__()
        if weight.dtype != torch.int8 or weight.ndim != 2:
            raise ValueError("ConvRot weight must be an INT8 matrix")
        if type(group_size) is not int or group_size not in (4, 16, 64, 256) or weight.shape[1] % group_size:
            raise ValueError("ConvRot group size must divide input features")
        if not scale.is_floating_point():
            raise ValueError("ConvRot scales must be floating-point values")
        scale = scale.float()
        if scale.numel() not in (1, weight.shape[0]) or not torch.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError("ConvRot scales must be positive finite scalar/per-output-channel values")
        if bias is not None and (bias.shape != (weight.shape[0],) or bias.dtype not in (torch.float16, torch.bfloat16, torch.float32) or not torch.isfinite(bias).all()):
            raise ValueError("ConvRot bias must contain one finite floating-point value per output")
        self.in_features, self.out_features = weight.shape[1], weight.shape[0]
        self.group_size = group_size
        self.register_buffer("weight", weight.contiguous())
        self.register_buffer("weight_scale", scale.float().reshape(-1).contiguous())
        self.register_buffer("bias", bias)

    def _apply(self, fn, recurse=True):
        # Keep scales in FP32 even when the containing DiT is cast to BF16.
        scale = self._buffers.pop("weight_scale")
        try:
            result = super()._apply(fn, recurse=recurse)
            self._buffers["weight_scale"] = scale.to(device=self.weight.device, dtype=torch.float32)
        except BaseException:
            self._buffers["weight_scale"] = scale
            raise
        return result

    def forward(self, x):
        ck = kitchen()
        flat = x.reshape(-1, self.in_features)
        if flat.shape[0] == 0:
            return x.new_empty((*x.shape[:-1], self.out_features))
        if flat.shape[0] <= LINEAR_ROW_CHUNK:
            return ck.int8_linear(x, self.weight, self.weight_scale, self.bias, x.dtype,
                                  convrot=True, convrot_groupsize=self.group_size)
        # The eager CK fallback accumulates a full MxN INT32 matrix. Bound M
        # for high-resolution videos; row-wise rotation/quantization is independent.
        output = torch.empty((flat.shape[0], self.out_features), device=x.device, dtype=x.dtype)
        for start in range(0, flat.shape[0], LINEAR_ROW_CHUNK):
            stop = min(start + LINEAR_ROW_CHUNK, flat.shape[0])
            output[start:stop] = ck.int8_linear(flat[start:stop], self.weight, self.weight_scale,
                self.bias, x.dtype, convrot=True, convrot_groupsize=self.group_size)
        return output.reshape(*x.shape[:-1], self.out_features)


def read_quantization(path: str | Path) -> dict | None:
    from safetensors import safe_open
    with safe_open(str(path), framework="pt", device="cpu") as source:
        meta = source.metadata() or {}
    if meta.get("flashvsr_plus_format") != FORMAT:
        return None
    spec = json.loads(meta["flashvsr_plus_quantization"])
    if (not isinstance(spec, dict) or spec.get("format") != FORMAT
            or spec.get("mode") not in ("turbo", "base", "v1")
            or not isinstance(spec.get("source_sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", spec["source_sha256"]) is None):
        raise ValueError("Invalid FlashVSR+ INT8 ConvRot quantization metadata")
    if not isinstance(spec.get("layers"), dict) or not spec["layers"]:
        raise ValueError("Invalid FlashVSR+ INT8 ConvRot layer manifest")
    for name, conf in spec["layers"].items():
        if (not isinstance(name, str) or not name.startswith("blocks.") or not isinstance(conf, dict)
                or conf.get("format") != "int8_tensorwise" or conf.get("convrot") is not True
                or type(conf.get("convrot_groupsize")) is not int
                or conf.get("convrot_groupsize") not in (4, 16, 64, 256)):
            raise ValueError(f"Invalid FlashVSR+ INT8 ConvRot quantization layer: {name}")
    return spec


def apply_quantized_checkpoint(model: nn.Module, path: str | Path) -> dict:
    from safetensors.torch import load_file
    spec = read_quantization(path)
    if spec is None:
        raise ValueError("This is not a FlashVSR+ INT8 ConvRot checkpoint")
    kitchen()
    state = load_file(str(path), device="cpu")
    for name, conf in spec["layers"].items():
        parent_name, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_name)
        old = getattr(parent, leaf)
        weight, scale = state.pop(name + ".weight"), state.pop(name + ".weight_scale")
        bias = state.pop(name + ".bias", None)
        if not isinstance(old, nn.Linear) or tuple(old.weight.shape) != tuple(weight.shape):
            raise ValueError(f"Quantized layer shape mismatch: {name}")
        if (old.bias is None) != (bias is None) or (bias is not None and bias.shape != old.bias.shape):
            raise ValueError(f"Quantized layer bias mismatch: {name}")
        setattr(parent, leaf, ConvRotLinear(weight, scale, bias, conf["convrot_groupsize"]))
    expected = {k for k in model.state_dict() if not k.startswith("LQ_proj_in.")
                and not any(k.startswith(n + ".") for n in spec["layers"])}
    if set(state) != expected:
        raise ValueError(f"Quantized checkpoint keys mismatch: missing={sorted(expected-set(state))[:5]}, "
                         f"unexpected={sorted(set(state)-expected)[:5]}")
    result = model.load_state_dict(state, strict=False, assign=True)
    if result.unexpected_keys:
        raise ValueError(f"Unexpected checkpoint keys: {result.unexpected_keys[:5]}")
    return spec


def convert_checkpoint(source: str | Path, destination: str | Path, *, mode: str,
                       device: str = "cpu", progress: Callable[[str], None] | None = None) -> dict:
    from safetensors import safe_open
    from safetensors.torch import save_file
    from .model_io import normalize_key
    from comfy_kitchen.backends.eager.quantization import quantize_int8_convrot_weight
    source, destination = Path(source).expanduser(), Path(destination).expanduser()
    if source.resolve() == destination.resolve():
        raise ValueError("Quantization must not overwrite the original model")
    if mode not in ("turbo", "base", "v1"):
        raise ValueError("Invalid model mode")
    destination.parent.mkdir(parents=True, exist_ok=True)
    output, layers, errors = {}, {}, []
    h = hashlib.sha256()
    with source.open("rb") as f:
        for block in iter(lambda: f.read(16*1024*1024), b""):
            h.update(block)
    with safe_open(str(source), framework="pt", device="cpu") as st:
        for source_key in st.keys():
            value = st.get_tensor(source_key)
            key = normalize_key(source_key)
            if key.startswith("blocks.") and key.endswith(".weight") and value.ndim == 2:
                group = next((g for g in (256, 64, 16, 4) if value.shape[1] % g == 0), None)
            else:
                group = None
            if group is None:
                output[key] = value.to(torch.bfloat16) if value.is_floating_point() else value
                continue
            name = key[:-7]
            if progress:
                progress(f"INT8 ConvRot: {name}")
            weight = value.to(device=device, dtype=torch.float32)
            q, scale = quantize_int8_convrot_weight(weight, group)
            output[key], output[name+".weight_scale"] = q.cpu(), scale.float().cpu()
            layers[name] = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": group}
            del weight, q, scale, value
    spec = {"format": FORMAT, "mode": mode, "source_sha256": h.hexdigest(), "layers": layers}
    if not layers:
        raise ValueError("No eligible transformer Linear weights were found")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        save_file(output, str(temporary), metadata={"flashvsr_plus_format": FORMAT,
                  "flashvsr_plus_quantization": json.dumps(spec, sort_keys=True),
                  "source_sha256": spec["source_sha256"]})
        # Windows rejects overlapping replacements of the same destination.
        with _COMMIT_LOCK:
            os.replace(temporary, destination)
            spec["output_bytes"] = destination.stat().st_size
    finally:
        temporary.unlink(missing_ok=True)
    return spec
