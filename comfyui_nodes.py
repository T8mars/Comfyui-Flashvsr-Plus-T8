from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
import tempfile
import threading
from pathlib import Path

import folder_paths
import comfy.model_management as mm
import comfy.utils

from .flashvsr_plus.comfy_runtime import ModelHandle, Options, geometry, upscale_images
from .flashvsr_plus.inference import InferenceRequest
from .flashvsr_plus.weights import ROOT, FILES

CATEGORY = "FlashVSR+"
MODEL_TYPE = "FLASHVSR_PLUS_MODEL"
_OUTPUT_COMMIT_LOCK = threading.Lock()
OPTIONS_TYPE = "FLASHVSR_PLUS_OPTIONS"
MODEL_DIR = Path(folder_paths.models_dir)/"flashvsr_plus"
folder_paths.add_model_folder_path("flashvsr_plus", str(MODEL_DIR))
if (ROOT/"ckpts").is_dir():
    folder_paths.add_model_folder_path("flashvsr_plus", str(ROOT/"ckpts"))


def _progress():
    bar = comfy.utils.ProgressBar(100)
    chunks = 0
    def update(message):
        nonlocal chunks
        mm.throw_exception_if_processing_interrupted()
        logging.info("[FlashVSR+] %s", message)
        match = re.match(r"DiT chunk (\d+)/(\d+)", message)
        if match:
            bar.update_absolute(min(95, round(90*int(match[1])/int(match[2]))), 100)
        elif message.startswith("DiT chunk"):
            chunks += 1
            bar.update_absolute(min(90, chunks), 100)
        elif "complete" in message.lower() or "saved" in message.lower():
            bar.update_absolute(98, 100)
    return bar, update


class FlashVSRPlusModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mode": (["turbo", "base", "v1"],),
            "precision": (["bf16", "int8_convrot", "fp16"],),
            "attention": (["auto", "triton_block_sparse", "block_sparse", "sdpa"],),
            "decoder": (["tiny", "vae"],),
            "auto_download": ("BOOLEAN", {"default": True}),
            "keep_loaded": ("BOOLEAN", {"default": False, "tooltip": "Keep weights in CPU RAM between requests. GPU memory is released after each request."}),
            "model_directory": ("STRING", {"default": "", "tooltip": "Optional model root; blank discovers models/flashvsr_plus and the bundled ckpts folder."}),
        }}
    RETURN_TYPES = (MODEL_TYPE, "STRING")
    RETURN_NAMES = ("model", "model_info")
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(self, mode="turbo", precision="bf16", attention="auto", decoder="tiny",
             auto_download=True, keep_loaded=False, model_directory=""):
        device = mm.get_torch_device()
        if device.type != "cuda":
            raise RuntimeError("Official FlashVSR+ requires an NVIDIA CUDA GPU")
        directories = [Path(x) for x in folder_paths.get_folder_paths("flashvsr_plus")]
        if model_directory.strip():
            directory = Path(model_directory).expanduser()
        else:
            weight_names = (("dit_int8_convrot.safetensors", "dit.safetensors")
                            if precision == "int8_convrot" else ("dit.safetensors",))
            candidates = [d for name in weight_names for d in directories if (d/mode/name).is_file()]
            components = [entry[1] for entry in FILES if entry[0] in {mode+"_lq", decoder}]
            directory = next((d for d in candidates if all((d/name).is_file() for name in components)),
                             candidates[0] if candidates else MODEL_DIR)
        handle = ModelHandle(mode=mode, precision=precision, attention=attention, decoder=decoder,
            directory=directory, auto_download=auto_download, keep_loaded=keep_loaded, device=str(device))
        return handle, json.dumps(handle.description(), ensure_ascii=False, indent=2)


class FlashVSRPlusOptions:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "temporal_compression": ([4, 2, 1],),
            "temporal_merge": (["gaussian", "first"],),
            "temporal_gaussian_sigma": ("FLOAT", {"default": 1., "min": .01, "max": 10., "step": .05}),
            "decode_every": ("INT", {"default": 28, "min": 0, "max": 1024, "step": 4, "tooltip": "Latent-frame interval: 28 is about 112 pixel frames at compression 4. 0=auto/full clip. Use positive multiples of 4 for Tiny."}),
            "cache_offload_mode": (["auto", "never", "always"],),
            "offload_original_cache": ("BOOLEAN", {"default": False}),
            "tiny_decoder_mode": (["auto", "serial", "parallel"],),
            "tiny_decoder_channels_last": ("BOOLEAN", {"default": True}),
            "color_correction": ("BOOLEAN", {"default": False}),
            "color_correction_levels": ("INT", {"default": 5, "min": 1, "max": 8}),
            "color_correction_chunk_size": ("INT", {"default": 16, "min": 1, "max": 128}),
            "ffn_chunk_tokens": ("INT", {"default": 8192, "min": 0, "max": 262144, "step": 1024}),
            "vae_tiled": ("BOOLEAN", {"default": True}),
            "vae_tile_height": ("INT", {"default": 30, "min": 4, "max": 256}),
            "vae_tile_width": ("INT", {"default": 52, "min": 4, "max": 256}),
            "vae_overlap": ("INT", {"default": 15, "min": 0, "max": 128}),
            "video_preset": (["veryfast", "ultrafast", "superfast", "faster", "fast", "medium", "slow"],),
        }}
    RETURN_TYPES = (OPTIONS_TYPE,)
    RETURN_NAMES = ("options",)
    FUNCTION = "configure"
    CATEGORY = CATEGORY

    def configure(self, **kwargs):
        return (Options(**kwargs).validate(),)


class FlashVSRPlusUpscaleFrames:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": (MODEL_TYPE,), "images": ("IMAGE",),
            "scale": ("FLOAT", {"default": 4., "min": .25, "max": 16., "step": .25}),
            "width": ("INT", {"default": 0, "min": 0, "max": 16384, "tooltip": "0 uses scale or preserves aspect ratio from height."}),
            "height": ("INT", {"default": 0, "min": 0, "max": 16384}),
            "seed": ("INT", {"default": 1, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
        }, "optional": {"options": (OPTIONS_TYPE,)}}
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "info")
    FUNCTION = "upscale"
    CATEGORY = CATEGORY

    def upscale(self, model, images, scale=4., width=0, height=0, seed=1, options=None):
        bar, progress = _progress()
        output, result = upscale_images(model, images, scale=scale, width=width, height=height,
            seed=seed, options=options, progress=progress,
            interrupt=mm.throw_exception_if_processing_interrupted)
        bar.update_absolute(100, 100)
        return output, json.dumps(result.metadata, ensure_ascii=False, indent=2)


class FlashVSRPlusUpscaleVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": (MODEL_TYPE,), "video": ("VIDEO",),
            "scale": ("FLOAT", {"default": 4., "min": .25, "max": 16., "step": .25}),
            "width": ("INT", {"default": 0, "min": 0, "max": 16384}),
            "height": ("INT", {"default": 0, "min": 0, "max": 16384}),
            "seed": ("INT", {"default": 1, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            "max_frames": ("INT", {"default": 0, "min": 0, "max": 10000000}),
            "filename_prefix": ("STRING", {"default": "FlashVSRPlus/FlashVSRPlus"}),
        }, "optional": {"options": (OPTIONS_TYPE,)}}
    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "info")
    FUNCTION = "upscale"
    CATEGORY = CATEGORY

    def upscale(self, model, video, scale=4., width=0, height=0, seed=1,
                max_frames=0, filename_prefix="FlashVSRPlus/FlashVSRPlus", options=None):
        from comfy_api.input_impl import VideoFromFile
        from comfy_api.latest import Types
        options = options or Options(decode_every=(0 if model.decoder == "vae" else 28))
        if model.decoder == "vae" and options.decode_every:
            raise ValueError("VAE uses full-clip decoding: set decode_every=0 in FlashVSR+ Options. Tiny supports incremental long videos.")
        sw, sh = video.get_dimensions()
        width, height, cw, ch = geometry(sw, sh, scale, width, height)
        folder, name, counter, _, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), width, height)
        destination = Path(folder)/f"{name}_{counter:05}_.mp4"
        Path(folder).mkdir(parents=True, exist_ok=True)
        bar, progress = _progress()
        with tempfile.TemporaryDirectory(prefix="flashvsr_plus_", dir=folder_paths.get_temp_directory()) as temporary, \
                tempfile.TemporaryDirectory(prefix=".flashvsr_plus_", dir=folder) as output_temporary:
            source, restored = Path(temporary)/"source.mp4", Path(output_temporary)/"restored.mp4"
            progress("Preparing video with its active trim, crop and audio")
            video.save_to(str(source), format=Types.VideoContainer.MP4,
                          codec=Types.VideoCodec.H264, crf=0, preset="ultrafast")
            mm.throw_exception_if_processing_interrupted()
            result = model.infer(InferenceRequest(input_path=source, output_path=restored,
                height=ch, width=cw, content_height=height, content_width=width,
                max_frames=max_frames or None, fps=float(video.get_frame_rate())), seed=seed, options=options, progress=progress,
                interrupt=mm.throw_exception_if_processing_interrupted)
            # Committing only a complete file prevents cancelled jobs leaving published partial output.
            with _OUTPUT_COMMIT_LOCK:
                mm.throw_exception_if_processing_interrupted()
                while destination.exists():
                    counter += 1
                    destination = Path(folder)/f"{name}_{counter:05}_.mp4"
                os.replace(restored, destination)
        result.output_path = str(destination)
        if result.metadata.get("artifact"):
            result.metadata["artifact"]["path"] = str(destination)
        result.metadata.update({"path": str(destination), "width": width, "height": height,
                                "canvas_width": cw, "canvas_height": ch})
        bar.update_absolute(100, 100)
        return VideoFromFile(str(destination)), json.dumps(result.metadata, ensure_ascii=False, indent=2)


NODE_CLASS_MAPPINGS = {
    "FlashVSRPlusModelLoader": FlashVSRPlusModelLoader,
    "FlashVSRPlusOptions": FlashVSRPlusOptions,
    "FlashVSRPlusUpscaleFrames": FlashVSRPlusUpscaleFrames,
    "FlashVSRPlusUpscaleVideo": FlashVSRPlusUpscaleVideo,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FlashVSRPlusModelLoader": "FlashVSR+ Model Loader",
    "FlashVSRPlusOptions": "FlashVSR+ Options",
    "FlashVSRPlusUpscaleFrames": "FlashVSR+ Upscale Frames",
    "FlashVSRPlusUpscaleVideo": "FlashVSR+ Upscale Video",
}
