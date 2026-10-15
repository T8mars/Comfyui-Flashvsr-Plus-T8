from __future__ import annotations

from typing import Any

import torch

from .base import DecodeResult


class VAEDecoder:
    """Adapter for a Wan VAE-like module exposing ``decode``."""

    def __init__(self, module, *, tiled: bool = True, tile_size=(30, 52), tile_stride=(15, 26)):
        if not hasattr(module, "decode"):
            raise TypeError("VAE module must expose decode(latents, ...)")
        self.module = module
        self.tiled = tiled
        self.tile_size = tuple(tile_size)
        self.tile_stride = tuple(tile_stride)
        self.interrupt_callback = None

    def reset(self) -> None:
        if hasattr(self.module, "clear_cache"):
            self.module.clear_cache()

    def decode(
        self,
        latents: torch.Tensor,
        *,
        lq_video: torch.Tensor | None = None,
        state: Any = None,
        stream: bool = False,
    ) -> DecodeResult:
        setter = getattr(self.module, "set_interrupt_callback", None)
        try:
            if self.interrupt_callback is not None:
                self.interrupt_callback()
            if setter is not None:
                setter(self.interrupt_callback)
            return self._decode(latents, lq_video=lq_video, state=state, stream=stream)
        except BaseException:
            self.reset()
            raise
        finally:
            if setter is not None:
                setter(None)

    def _decode(
        self,
        latents: torch.Tensor,
        *,
        lq_video: torch.Tensor | None = None,
        state: Any = None,
        stream: bool = False,
    ) -> DecodeResult:
        del lq_video
        if not self.tiled and isinstance(self.module, torch.nn.Module):
            parameter = next(self.module.parameters(), None)
            if parameter is not None:
                latents = latents.to(device=parameter.device, dtype=parameter.dtype)
        tiled_kwargs = {}
        if self.tiled:
            tiled_kwargs.update(
                tiled=True, tile_size=self.tile_size, tile_stride=self.tile_stride
            )
        if stream and hasattr(self.module, "stream_decode"):
            value = self.module.stream_decode(latents, **tiled_kwargs)
        else:
            attempts = (
                {"device": latents.device, **tiled_kwargs},
                tiled_kwargs,
                {"device": latents.device},
                {},
            )
            last_error = None
            for kwargs in attempts:
                try:
                    value = self.module.decode(latents, **kwargs)
                    break
                except TypeError as error:
                    last_error = error
            else:
                raise TypeError("VAE decode signature is not supported") from last_error
        if self.interrupt_callback is not None:
            self.interrupt_callback()
        if isinstance(value, (tuple, list)):
            value = value[0]
        elif hasattr(value, "sample"):
            value = value.sample
        if isinstance(value, torch.Tensor) and value.ndim == 4:
            value = value.unsqueeze(0)
        if not isinstance(value, torch.Tensor) or value.ndim != 5:
            raise TypeError("VAE decode must return [B,C,F,H,W] Tensor")
        next_state = state
        if stream and hasattr(self.module, "stream_state"):
            next_state = self.module.stream_state
        return DecodeResult(value.to("cpu").clamp(-1, 1), next_state)
