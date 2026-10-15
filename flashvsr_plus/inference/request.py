from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math
from numbers import Integral
from typing import Any

import torch


@dataclass
class InferenceRequest:
    input_path: str | Path | None = None
    lq_video: torch.Tensor | None = None
    output_path: str | Path | None = None
    context: torch.Tensor | None = None
    height: int | None = None
    width: int | None = None
    content_height: int | None = None
    content_width: int | None = None
    max_frames: int | None = None
    fps: float | None = None

    def validate(self) -> None:
        if (self.input_path is None) == (self.lq_video is None):
            raise ValueError("Provide exactly one of input_path or lq_video")
        if self.lq_video is not None and self.lq_video.ndim != 5:
            raise ValueError("lq_video must be [B,C,F,H,W] or [B,F,C,H,W]")
        if self.lq_video is not None and self.lq_video.shape[0] != 1:
            raise ValueError("inference currently supports exactly one video (batch size 1)")
        if self.lq_video is not None:
            if not self.lq_video.numel() or (self.lq_video.shape[1] != 3 and self.lq_video.shape[2] != 3):
                raise ValueError("lq_video must contain nonempty RGB frames")
        for name in ("height", "width", "content_height", "content_width", "max_frames"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, Integral) or isinstance(value, bool) or value <= 0):
                raise ValueError(f"{name} must be a positive integer")
        if self.fps is not None and (not math.isfinite(self.fps) or self.fps <= 0):
            raise ValueError("fps must be positive and finite")
        if (self.content_height is None) != (self.content_width is None):
            raise ValueError("content height/width must be provided together")
        if self.content_height is not None:
            if self.height is None or self.width is None or not (0 < self.content_height <= self.height and 0 < self.content_width <= self.width):
                raise ValueError("Content dimensions must fit the model canvas")
        if (self.height is None) != (self.width is None):
            raise ValueError("height and width must be provided together")


@dataclass
class InferenceResult:
    video: torch.Tensor | None
    output_path: str | None
    frames: int
    fps: float
    metadata: dict[str, Any] = field(default_factory=dict)
