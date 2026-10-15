"""Host-neutral service used by the ComfyUI nodes and integration tests."""
from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import math
import random
import threading
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from .backends import diagnostics, resolve_backend, validate_attention_size
from .inference import InferenceRequest, build_inference_pipeline
from .inference.presets import stream_config_for_mode
from .inference.preprocessing.input import _resize_tensor_video
from .quantization import convert_checkpoint, read_quantization
from .weights import ROOT, FILES, model_paths

log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class Options:
    temporal_compression: int = 4
    temporal_merge: str = "gaussian"
    temporal_gaussian_sigma: float = 1.0
    decode_every: int = 28
    cache_offload_mode: str = "auto"
    offload_original_cache: bool = False
    tiny_decoder_mode: str = "auto"
    tiny_decoder_channels_last: bool = True
    color_correction: bool = False
    color_correction_levels: int = 5
    color_correction_chunk_size: int = 16
    ffn_chunk_tokens: int = 8192
    vae_tiled: bool = True
    vae_tile_height: int = 30
    vae_tile_width: int = 52
    vae_overlap: int = 15
    video_preset: str = "veryfast"

    def validate(self):
        if self.temporal_compression not in (1, 2, 4):
            raise ValueError("temporal_compression must be 1, 2 or 4")
        if self.decode_every < 0 or self.decode_every % 4:
            raise ValueError("decode_every must be 0 (auto) or a positive multiple of 4")
        if self.temporal_merge not in ("gaussian", "first") or not math.isfinite(self.temporal_gaussian_sigma) or self.temporal_gaussian_sigma <= 0:
            raise ValueError("Invalid temporal merge configuration")
        if self.cache_offload_mode not in ("auto", "never", "always"):
            raise ValueError("Invalid cache offload mode")
        if self.tiny_decoder_mode not in ("auto", "serial", "parallel"):
            raise ValueError("Invalid Tiny decoder mode")
        if min(self.color_correction_levels, self.color_correction_chunk_size,
               self.vae_tile_height, self.vae_tile_width) <= 0 or self.ffn_chunk_tokens < 0:
            raise ValueError("Invalid decoder/color/FFN configuration")
        if self.vae_overlap < 0 or self.vae_overlap >= min(self.vae_tile_height, self.vae_tile_width):
            raise ValueError("VAE overlap must be smaller than each tile dimension")
        if self.video_preset not in ("ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"):
            raise ValueError("Invalid x264 encoding preset")
        return self


def geometry(source_width, source_height, scale, width=0, height=0):
    if min(source_width, source_height) <= 0 or not math.isfinite(scale) or scale <= 0:
        raise ValueError("Image dimensions and scale must be positive")
    if width < 0 or height < 0:
        raise ValueError("Target dimensions cannot be negative")
    if width and not height:
        height = max(1, round(source_height*width/source_width))
    elif height and not width:
        width = max(1, round(source_width*height/source_height))
    elif not width and not height:
        width, height = max(1, round(source_width*scale)), max(1, round(source_height*scale))
    return int(width), int(height), math.ceil(width/128)*128, math.ceil(height/128)*128


def image_to_model(images: torch.Tensor, width: int, height: int,
                   canvas_width: int, canvas_height: int) -> torch.Tensor:
    if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[-1] != 3 or images.shape[0] < 1:
        raise ValueError("IMAGE must be a nonempty [frames,height,width,3] RGB tensor")
    if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
        raise ValueError("IMAGE values must be finite and within [0,1]")
    # Keep the full clip in CPU memory; only bounded runner windows reach CUDA.
    value = images.detach().to(device="cpu", dtype=torch.float32).permute(3, 0, 1, 2).unsqueeze(0)
    value = _resize_tensor_video(value, height, width)[0].permute(1, 0, 2, 3)
    value = F.pad(value, (0, canvas_width-width, 0, canvas_height-height), mode="replicate")
    return value.mul_(2).sub_(1).permute(1, 0, 2, 3).unsqueeze(0).contiguous()


@contextlib.contextmanager
def preserve_rng(device):
    py, np_state = random.getstate(), np.random.get_state()
    index = torch.device(device).index
    devices = list(range(torch.cuda.device_count()))
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(py)
        np.random.set_state(np_state)


class ModelHandle:
    def __init__(self, *, mode="turbo", precision="bf16", attention="auto", decoder="tiny",
                 directory=None, auto_download=True, keep_loaded=False, device="cuda:0"):
        if mode not in ("turbo", "base", "v1") or decoder not in ("tiny", "vae"):
            raise ValueError("Invalid official model preset or decoder")
        if precision not in ("bf16", "fp16", "int8_convrot"):
            raise ValueError("Precision must be bf16, fp16 or int8_convrot")
        self.mode, self.precision, self.attention, self.decoder = mode, precision, attention, decoder
        self.directory = Path(directory or ROOT/"ckpts").expanduser().resolve()
        self.auto_download, self.keep_loaded, self.device = auto_download, keep_loaded, device
        self._lock, self._pipeline, self._key = threading.RLock(), None, None

    def description(self):
        return {"mode": self.mode, "precision": self.precision, "decoder": self.decoder,
                "attention": self.attention, "model_directory": str(self.directory),
                "device": self.device, "backends": diagnostics(self.device)}

    def release(self):
        if self._pipeline is not None:
            bundle = self._pipeline.bundle
            bundle.denoiser.interrupt_callback = None
            bundle.decoder.interrupt_callback = None
            bundle.denoiser.to("cpu")
            bundle.decoder.module.to("cpu")
            self._pipeline = None
            self._key = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _build(self, config, progress):
        target = self.directory/self.mode/"dit_int8_convrot.safetensors"
        paths = model_paths(self.mode, self.directory, self.decoder,
                            auto_download=self.auto_download, progress=progress,
                            include_dit=(self.precision != "int8_convrot" or (not target.is_file() and not self.auto_download)),
                            include_int8=(self.precision == "int8_convrot" and not target.is_file() and self.auto_download))
        if self.precision == "int8_convrot":
            if not target.exists():
                convert_checkpoint(paths.dit, target, mode=self.mode, device=self.device, progress=progress)
            spec = read_quantization(target)
            if spec is None or spec["mode"] != self.mode:
                raise ValueError("INT8 checkpoint mode/format does not match the requested model")
            expected_source = next(x[-1] for x in FILES if x[0] == self.mode + "_dit")
            if spec["source_sha256"] != expected_source:
                raise ValueError("INT8 checkpoint was not converted from the expected official preset")
            paths.dit = str(target)
        progress("Loading FlashVSR+ weights")
        return build_inference_pipeline(config, paths)

    def infer(self, request: InferenceRequest, *, seed=1, options: Options | None = None,
              progress: Callable[[str], None] | None = None, interrupt: Callable[[], None] | None = None):
        options = (options or Options(decode_every=(0 if self.decoder == "vae" else 28))).validate()
        progress = progress or (lambda message: log.info("FlashVSR+: %s", message))
        interrupt = interrupt or (lambda: None)
        with self._lock:
            interrupt()
            request.validate()
            backend = resolve_backend(self.attention, self.device)
            h, w = ((request.height, request.width) if request.height is not None else
                    request.lq_video.shape[-2:] if request.lq_video is not None else (request.height, request.width))
            validate_attention_size(h, w, backend)
            if self.decoder == "vae" and request.input_path is not None and options.decode_every:
                raise ValueError("Official VAE decoding requires decode_every=0 (auto); use Tiny for bounded long-video decoding")
            decode_every = options.decode_every if self.decoder == "tiny" and options.decode_every else "auto"
            config = stream_config_for_mode(self.mode, device=self.device, seed=int(seed),
                decode_every=decode_every, tiny_decoder_mode=options.tiny_decoder_mode,
                tiny_decoder_channels_last=options.tiny_decoder_channels_last,
                offload_original_cache=options.offload_original_cache, cache_offload_mode=options.cache_offload_mode,
                temporal_compression=options.temporal_compression, temporal_merge=options.temporal_merge,
                temporal_gaussian_sigma=options.temporal_gaussian_sigma, attn_kernel=backend,
                color_correction=options.color_correction, color_correction_levels=options.color_correction_levels,
                color_correction_chunk_size=options.color_correction_chunk_size)
            config.dtype = "fp16" if self.precision == "fp16" else "bf16"
            config.stream_ffn_chunk_tokens = options.ffn_chunk_tokens or None
            config.decoder.kind, config.decoder.tiled = self.decoder, options.vae_tiled
            config.decoder.tile_size = (options.vae_tile_height, options.vae_tile_width)
            config.decoder.tile_stride = (options.vae_tile_height-options.vae_overlap, options.vae_tile_width-options.vae_overlap)
            config.validate()
            key = dataclasses.asdict(config)
            key.pop("seed")
            try:
                # Cooperate with the host before loading our private pipeline.
                try:
                    from comfy import model_management
                except ImportError:
                    pass
                else:
                    needed = (3.8 if self.precision != "int8_convrot" else 2.5)*1024**3 + h*w*800
                    model_management.free_memory(needed, torch.device(self.device))
                with preserve_rng(self.device), torch.inference_mode():
                    if self._pipeline is None or self._key != key:
                        self.release()
                        self._pipeline = self._build(config, progress)
                        self._key = key
                    else:
                        self._pipeline.config.seed = int(seed)
                        self._pipeline.bundle.denoiser.to(self.device)
                        self._pipeline.bundle.decoder.module.to(self.device)
                    self._pipeline.bundle.denoiser.interrupt_callback = interrupt
                    self._pipeline.bundle.decoder.interrupt_callback = interrupt
                    def callback(message):
                        interrupt()
                        progress(message)
                    result = self._pipeline.infer(request, progress=callback, video_preset=options.video_preset)
                    result.metadata.update({"mode": self.mode, "precision": self.precision,
                                            "actual_backend": backend, "seed": int(seed)})
                    return result
            finally:
                if self._pipeline is not None:
                    self._pipeline.bundle.denoiser.interrupt_callback = None
                    self._pipeline.bundle.decoder.interrupt_callback = None
                    self._pipeline.bundle.denoiser.to("cpu")
                    self._pipeline.bundle.decoder.module.to("cpu")
                if not self.keep_loaded:
                    self.release()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


def upscale_images(model: ModelHandle, images, *, scale=4.0, width=0, height=0,
                   seed=1, fps=24.0, options=None, progress=None, interrupt=None):
    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise ValueError("Expected a ComfyUI IMAGE batch")
    width, height, cw, ch = geometry(images.shape[2], images.shape[1], scale, width, height)
    validate_attention_size(ch, cw, resolve_backend(model.attention, model.device))
    video = image_to_model(images, width, height, cw, ch)
    request = InferenceRequest(lq_video=video, fps=float(fps))
    result = model.infer(request, seed=seed, options=options, progress=progress, interrupt=interrupt)
    output = result.video[0, :, :, :height, :width].permute(1, 2, 3, 0).to(
        dtype=torch.float32, copy=True, memory_format=torch.contiguous_format).add_(1).div_(2).clamp_(0, 1)
    result.metadata.update({"width": width, "height": height, "canvas_width": cw, "canvas_height": ch})
    return output, result
