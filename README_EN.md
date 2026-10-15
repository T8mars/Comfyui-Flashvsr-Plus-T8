# FlashVSR+ T8 · Video Super-Resolution for ComfyUI

[简体中文](README.md) | **English**

A ComfyUI integration of [FlashVSR+](https://github.com/OpenImagingLab/FlashVSR_Plus), with native VIDEO / IMAGE nodes, audio preservation, streaming video processing, and INT8 ConvRot. Requires an NVIDIA CUDA GPU.

[Models](https://huggingface.co/t8star/Flashvsr-Plus-Comfy) · [Example workflows](example_workflows) · [Validation](docs/VALIDATION.md)

## Installation

Search for **FlashVSR+ T8** in **ComfyUI Manager**, install, and restart. Registry ID: `flashvsr-plus-t8`; publisher: `t8star`.

For manual installation, run from the ComfyUI directory and install dependencies with **ComfyUI's Python**:

```sh
git clone https://github.com/T8mars/Comfyui-Flashvsr-Plus-T8.git custom_nodes/Comfyui-Flashvsr-Plus-T8
python -m pip install -r custom_nodes/Comfyui-Flashvsr-Plus-T8/requirements.txt
```

The dependency list contains neither PyTorch nor transformers and does not replace them. `transformers==4.57.1` was tested in an isolated environment. On Windows, match Triton to your installed PyTorch; the tested combination is PyTorch `2.7.0+cu128` with `triton-windows==3.3.1.post21`.

## Quick start

1. Import the [VIDEO workflow](example_workflows/FlashVSR_Plus_Video.json) or [IMAGE workflow](example_workflows/FlashVSR_Plus_Frames.json). Copy the [sample video](example_workflows/flashvsr_plus_demo.mp4) into `ComfyUI/input`.
2. Choose the preset, precision, and decoder in **Model Loader**, leaving `auto_download=true`. Start with `turbo` + `int8_convrot` + `tiny`; INT8 requires Comfy Kitchen `>=0.2.35`.
3. Set the scale or target dimensions and run. The VIDEO path preserves audio, FPS, and upstream trims/crops; use VIDEO for longer clips.

| Node | Purpose |
| --- | --- |
| Model Loader | Turbo / Base / V1.1; BF16 / FP16 / INT8 ConvRot; Tiny / Wan VAE |
| Upscale Video | Native VIDEO input/output with incremental reading and encoding |
| Upscale Frames | IMAGE batch restoration with runtime information |
| Options | Temporal compression, cache offloading, decoding, and color correction |

## Models and acceleration

Models are discovered in `ComfyUI/models/flashvsr_plus` and this node's `ckpts` directory. A custom model root can be set in the loader. See the [model repository](https://huggingface.co/t8star/Flashvsr-Plus-Comfy) for the file layout. Downloads use pinned revisions and SHA-256 verification.

- **INT8 ConvRot:** each preconverted DiT is approximately **1.45 GB**. Use it with the corresponding LQ projection and decoder; the original DiT is optional. Transformer-block linear layers use Comfy Kitchen W8A8; other components stay floating point.
- **Attention:** official BSA, Triton, and SDPA are supported. `auto` selects an available backend; use Triton for high-resolution output. SDPA rejects oversized masks before allocation. Native BSA was not tested on this Windows host.
- **Segmented decoding:** Tiny accepts positive `decode_every` multiples of four, such as `4`. Wan VAE requires `0` for full-clip decoding.

INT8 primarily reduces weight memory; speed and quality vary by content and hardware. IMAGE output needs RAM for the full batch. VIDEO uses a constant-FPS timeline; normalize VFR sources to CFR first. Audio that cannot be copied into MP4, such as PCM, is encoded as AAC. Weights return to CPU after each request; `keep_loaded` retains the CPU copy.

## Validation

**45 regression tests passed.** Turbo / Tiny / Triton completed real **1920×1080, five-second, 24 FPS, 120-frame** ComfyUI workflows with BF16 and INT8. Every frame, timestamps, and audio were checked. An additional full GPU workflow passed with `transformers==4.57.1`. [Validation scope](docs/VALIDATION.md) · [1080p output clip](assets/validation_1080p_5s.mp4)

## Community and resources

[Bilibili](https://space.bilibili.com/385085361) · [YouTube](https://www.youtube.com/@T8star-Aix/) · [API](https://api.seedance.nz/sign-up?aff=5f4w) · [Free gallery](https://www.openzhenzhen.com) · [Online AI apps](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121) · [ComfyUI bundle](https://pan.quark.cn/s/264edb7e36bd) · [Hugging Face](https://huggingface.co/t8star)

## Credits and license

Based on [OpenImagingLab/FlashVSR_Plus](https://github.com/OpenImagingLab/FlashVSR_Plus) and [FlashVSR V1.1](https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1). INT8 linear operations use [Comfy Kitchen](https://github.com/Comfy-Org/comfy-kitchen). ComfyUI integration and derived checkpoints are maintained by T8star. Code and model files are Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
