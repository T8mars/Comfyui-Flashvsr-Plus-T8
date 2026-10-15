# Validation / 验证说明

## Tested configuration

- NVIDIA RTX 5090 Laptop, 24 GB VRAM; Windows.
- PyTorch `2.7.0+cu128`, `triton-windows==3.3.1.post21`.
- Turbo, Base, and V1.1; BF16 and INT8 ConvRot. Turbo FP16 also has a short-clip check.
- Tiny and Wan VAE; temporal compression 4 / 2 / 1; IMAGE and native ComfyUI VIDEO.

## Results

| Check | Result |
| --- | --- |
| Regression suite | 45 tests passed |
| Real 1080p video | BF16 and INT8: 1920×1080, 5 seconds, 24 FPS, 120 frames |
| Output validation | Every frame decoded, continuous timestamps, audio preserved |
| Streaming | 65-frame multi-segment check; 1080p workflow uses 8 DiT chunks and a peak 20-frame input buffer |
| Cancellation | Actual Tiny / Wan GPU decode interruption, cache cleanup, same-handle recovery |
| VIDEO integration | Active trims/crops, fractional FPS, short and long audio, cross-volume temporary/output directories |
| File handling | Home-relative paths, atomic output publication, independent concurrent VIDEO output files |
| INT8 integrity | Original checkpoint SHA-256 provenance, strict metadata, independent INT8 directory without the floating DiT |
| transformers 4.57.1 | Four nodes registered, all 45 tests passed, full 1080p five-second GPU workflow passed in isolation |

The 1080p configuration is **Turbo / Tiny / Triton / decode_every=4**. This is a correctness check; timings include model loading and validation and are not a general speed benchmark. [View the output clip](../assets/validation_1080p_5s.mp4).

## Limits / 范围

- Native BSA is optional and was not available for Windows kernel testing; Triton and SDPA were exercised.
- Other presets retain short-clip coverage; the five-second 1080p cases do not cover every preset/decoder combination.
- No new 4K clip testing was performed. Historical 4K evidence is single-frame only.
- Wan VAE uses full-clip decoding; Tiny is required for bounded incremental decoding.
- VFR input follows the ComfyUI core's CFR conversion; original per-frame PTS are not preserved.
- INT8 may affect fine detail. The included quality comparison describes one sample rather than a general quality guarantee.

实际验证基于上述配置；不同设备、驱动、ComfyUI 版本和视频内容可能产生不同表现。
