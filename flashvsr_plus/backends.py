"""Capability checks for the three attention implementations supported upstream."""
from __future__ import annotations

import importlib
import re
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class BackendStatus:
    name: str
    available: bool
    reason: str


def backend_status(device: str = "cuda:0") -> list[BackendStatus]:
    target = torch.device(device)
    if target.type != "cuda" or not torch.cuda.is_available():
        return [BackendStatus(x, False, "FlashVSR+ requires an NVIDIA CUDA device")
                for x in ("sdpa", "triton_block_sparse", "block_sparse")]
    result = [BackendStatus("sdpa", True, "PyTorch SDPA; subject to compact-mask expansion limits")]
    try:
        triton = importlib.import_module("triton")
        version = tuple(int(x) for x in re.findall(r"\d+", triton.__version__)[:2])
        sm = torch.cuda.get_device_capability(target)
        if sm[0] >= 12 and version < (3, 3):
            raise RuntimeError("SM120 requires Triton >=3.3; match Triton to your PyTorch version")
        importlib.import_module(__package__ + ".attention_ops")
        result.append(BackendStatus("triton_block_sparse", True, f"bundled official operator; Triton {triton.__version__}"))
    except (ImportError, OSError, RuntimeError) as exc:
        result.append(BackendStatus("triton_block_sparse", False, str(exc)))
    try:
        module = importlib.import_module("block_sparse_attn")
        if not callable(getattr(module, "block_sparse_attn_func", None)):
            raise ImportError("block_sparse_attn_func is missing")
        result.append(BackendStatus("block_sparse", True, "official Block-Sparse-Attention extension imported"))
    except (ImportError, OSError, RuntimeError) as exc:
        result.append(BackendStatus("block_sparse", False, str(exc)))
    return result


def resolve_backend(requested: str, device: str = "cuda:0") -> str:
    states = {x.name: x for x in backend_status(device)}
    if requested == "auto":
        for name in ("block_sparse", "triton_block_sparse", "sdpa"):
            if states[name].available:
                return name
        raise RuntimeError("No CUDA attention backend is available")
    if requested not in states:
        raise ValueError(f"Unsupported backend {requested!r}; choose auto, block_sparse, triton_block_sparse, or sdpa")
    if not states[requested].available:
        raise RuntimeError(f"{requested} is unavailable: {states[requested].reason}")
    return requested


def validate_attention_size(height: int, width: int, backend: str) -> None:
    if min(height, width) <= 0 or height % 128 or width % 128:
        raise ValueError("The model canvas must have positive height/width divisible by 128")
    if backend == "sdpa":
        spatial = (height // 128) * (width // 128)
        entries = 12 * (2 * spatial * 128) * (5 * spatial * 128)
        if entries > 2_000_000_000:
            raise RuntimeError(
                f"SDPA cannot process this canvas at full cache: its bool mask alone would need "
                f"{entries / 1024**3:.2f} GiB. Install a compatible Triton or official BSA backend, "
                "or choose a smaller output. Reducing video duration does not fix the full-cache limit."
            )


def diagnostics(device: str = "cuda:0") -> list[dict]:
    return [asdict(x) for x in backend_status(device)]
