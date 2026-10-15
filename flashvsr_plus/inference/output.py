"""Inference output serialization helpers."""

from __future__ import annotations

from pathlib import Path
from fractions import Fraction
import subprocess
from typing import Any

import cv2
import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
import torch


def tensor_to_uint8_frame(
    frame: torch.Tensor, *, value_range: str = "neg_one_to_one"
) -> np.ndarray:
    if frame.ndim != 3:
        raise ValueError(f"Expected [C,H,W] or [H,W,C], got {frame.shape}")
    if frame.shape[0] in (1, 3) and frame.shape[-1] not in (1, 3):
        frame = frame.permute(1, 2, 0)
    if frame.shape[-1] == 1:
        frame = frame.repeat(1, 1, 3)
    value = frame.detach().float().cpu()
    if value_range == "neg_one_to_one":
        value = (value + 1.0) / 2.0
    elif value_range != "zero_to_one":
        raise ValueError(
            f"value_range must be 'neg_one_to_one' or 'zero_to_one', got {value_range!r}"
        )
    value = value.clamp(0, 1).numpy()
    return np.rint(value * 255.0).astype(np.uint8)


def tensor_to_uint8_frames(
    video: torch.Tensor, *, value_range: str = "neg_one_to_one"
) -> np.ndarray:
    if video.ndim == 5:
        if video.shape[0] != 1:
            raise ValueError("Pass one video at a time when saving")
        video = video[0].permute(1, 0, 2, 3)
    elif video.ndim == 4:
        if video.shape[0] in (1, 3) and video.shape[1] not in (1, 3):
            video = video.permute(1, 0, 2, 3)
    else:
        raise ValueError(f"Expected [F,C,H,W] or [B,C,F,H,W], got {video.shape}")
    if video.shape[1] == 1:
        video = video.repeat(1, 3, 1, 1)
    return np.stack(
        [tensor_to_uint8_frame(frame.permute(1, 2, 0), value_range=value_range) for frame in video]
    )


class IncrementalVideoWriter:
    """Append bounded CPU tensors to a single encoder session."""

    def __init__(self, path, fps, *, preset="veryfast", crop_hw=None):
        self.destination = Path(path)
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps)
        self.crop_hw = crop_hw
        self.pixel_format = "yuv444p" if crop_hw and any(x % 2 for x in crop_hw) else "yuv420p"
        self.writer = imageio.get_writer(
            str(self.destination), format="FFMPEG", mode="I", fps=self.fps,
            codec="libx264", pixelformat=self.pixel_format, macro_block_size=None,
            input_params=["-r", str(Fraction(self.fps).limit_denominator(1_000_000))],
            output_params=["-preset", preset],
        )
        self.frame_count = 0
        self.first_frame = None

    def append(self, video, *, value_range="neg_one_to_one"):
        # Explicitly accept [B,C,F,H,W], avoiding ambiguous short F dimensions.
        if video.ndim != 5 or video.shape[0] != 1:
            raise ValueError("Expected one video [1,C,F,H,W]")
        if self.crop_hw is not None:
            video = video[..., :self.crop_hw[0], :self.crop_hw[1]]
        for tensor_frame in video[0].permute(1, 0, 2, 3):
            frame = tensor_to_uint8_frame(tensor_frame.permute(1, 2, 0), value_range=value_range)
            if self.first_frame is None:
                self.first_frame = frame
            self.writer.append_data(frame)
            self.frame_count += 1

    def close(self):
        self.writer.close()

    def verify(self):
        destination, fps = self.destination, self.fps
        frame_count, first_frame = self.frame_count, self.first_frame
        if first_frame is None:
            raise RuntimeError("No frames were written")
        height, width = first_frame.shape[:2]
        capture = cv2.VideoCapture(str(destination))
        ok, decoded_bgr = capture.read()
        saved_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        saved_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        saved_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        saved_fps = float(capture.get(cv2.CAP_PROP_FPS))
        capture.release()
        expected = (frame_count, width, height)
        actual = (saved_frames, saved_width, saved_height)
        if actual != expected:
            raise RuntimeError(
                f"Saved video round-trip mismatch for {destination}: {actual} != {expected}"
            )
        if not ok or decoded_bgr is None:
            raise RuntimeError(f"Could not decode the saved video: {destination}")
        if not np.isclose(saved_fps, fps, rtol=1e-9, atol=1e-6):
            raise RuntimeError(f"Saved video FPS mismatch for {destination}: {saved_fps} != {fps}")
        if first_frame is None:
            raise RuntimeError("No frames were written")
        decoded_rgb = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)
        color_mae = float(
            np.mean(np.abs(decoded_rgb.astype(np.float32) - first_frame.astype(np.float32)))
        )
        return {
            "path": str(destination),
            "frames": saved_frames,
            "width": width,
            "height": height,
            "fps": saved_fps,
            "codec": "libx264",
            "pixel_format": self.pixel_format,
            "first_frame_color_mae": color_mae,
        }


def save_tensor_video(
    video: torch.Tensor,
    path: str | Path,
    fps: float = 16,
    *,
    value_range: str = "neg_one_to_one",
    preset: str = "veryfast",
) -> dict[str, Any]:
    """Save an RGB tensor as H.264 and verify the first decoded frame."""
    if video.ndim == 4:
        if video.shape[0] in (1, 3) and video.shape[1] not in (1, 3):
            video = video.permute(1, 0, 2, 3)
        video = video.permute(1, 0, 2, 3).unsqueeze(0)
    writer = IncrementalVideoWriter(path, fps, preset=preset, crop_hw=video.shape[-2:])
    try:
        writer.append(video, value_range=value_range)
    finally:
        writer.close()
    return writer.verify()


def mux_source_audio(
    video_path: str | Path,
    source_path: str | Path,
    destination: str | Path,
) -> None:
    """Copy the source audio stream when one exists, otherwise keep video-only."""
    video_path = Path(video_path)
    source_path = Path(source_path)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    try:
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if frames <= 0 or not np.isfinite(fps) or fps <= 0:
        raise RuntimeError(f"Could not determine generated video duration: {video_path}")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg,
        "-y",
        "-i", str(video_path),
        "-i", str(source_path),
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "copy",
        "-c:a", "copy",
        "-t", f"{frames / fps:.9f}",
        "-movflags", "+faststart",
        str(destination),
    ]
    for audio_codec in ("copy", "aac"):
        command[command.index("-c:a") + 1] = audio_codec
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, encoding="utf-8", errors="replace")
        if completed.returncode == 0:
            return
        if "codec not currently supported in container" not in completed.stderr:
            break
    raise RuntimeError("Could not preserve source audio with ffmpeg: " + completed.stderr.strip())
