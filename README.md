# FlashVSR+ T8 · ComfyUI 视频与单图超分辨率

**简体中文** | [English](README_EN.md)

将 [FlashVSR+](https://github.com/OpenImagingLab/FlashVSR_Plus) 接入 ComfyUI，支持原生 VIDEO / IMAGE、音频保留、分段处理与 INT8 ConvRot。适用于 NVIDIA CUDA GPU。

[模型下载](https://huggingface.co/t8star/Flashvsr-Plus-Comfy) · [示例工作流](example_workflows) · [验证记录](docs/VALIDATION.md)

## 安装

在 **ComfyUI Manager** 搜索 **FlashVSR+ T8**，安装并重启。Registry 节点 ID：`flashvsr-plus-t8`，Publisher：`t8star`。

手动安装时，在 ComfyUI 目录执行，并使用 **ComfyUI 自己的 Python** 安装依赖：

```sh
git clone https://github.com/T8mars/Comfyui-Flashvsr-Plus-T8.git custom_nodes/Comfyui-Flashvsr-Plus-T8
python -m pip install -r custom_nodes/Comfyui-Flashvsr-Plus-T8/requirements.txt
```

依赖不包含 PyTorch 或 transformers，不会主动替换它们。`transformers==4.57.1` 已完成隔离环境验证。Windows 的 Triton 需匹配现有 PyTorch；已验证组合为 PyTorch `2.7.0+cu128` 与 `triton-windows==3.3.1.post21`。

## 快速使用

1. 视频导入 [VIDEO 工作流](example_workflows/FlashVSR_Plus_Video.json) 或 [视频帧 IMAGE 工作流](example_workflows/FlashVSR_Plus_Frames.json)。将 [示例视频](example_workflows/flashvsr_plus_demo.mp4) 放入 `ComfyUI/input`。单张照片使用下方专用工作流。
2. 在 **Model Loader** 选择模式、精度和解码器，保持 `auto_download=true`。推荐先用 `turbo` + `int8_convrot` + `tiny`；INT8 需要 Comfy Kitchen `>=0.2.35`。
3. 设置倍率或目标宽高后运行。VIDEO 路径保留音频、帧率及上游剪辑/裁切；长视频优先使用 VIDEO。

| 节点 | 功能 |
| --- | --- |
| Model Loader | Turbo / Base / V1.1，BF16 / FP16 / INT8 ConvRot，Tiny / Wan VAE |
| Upscale Video | 原生 VIDEO 输入输出，增量读取和编码 |
| Upscale Frames | 单张 RGB 图片或连续视频帧超分辨率，输出图像与运行信息 |
| Options | 时间压缩、缓存卸载、解码与颜色修正 |

## 单图放大

导入 [单图工作流](example_workflows/FlashVSR_Plus_Single_Image.json)，在 **Load Image** 选择静态 PNG/JPEG，运行后由 **Save Image** 保存 PNG。可将 [示例图片](example_workflows/flashvsr_plus_sample.png) 放入 `ComfyUI/input` 直接复现；也可拖入 [带工作流的实测结果 PNG](assets/single_image/portrait.png)。[API 示例](example_workflows/FlashVSR_Plus_Single_Image_API.json)

默认 `turbo / int8_convrot / tiny`、4×、固定 seed 1，并开启颜色校正。已完成 BF16 / INT8 的人像、1080p 照片与文字图 GPU 实测；照片增强验收通过，小字与平色图表不保证保真。多张无关图片需逐张处理，透明通道需单独保留。[实测对比与验收范围](docs/SINGLE_IMAGE.md)

## 模型与加速

模型存放于 `ComfyUI/models/flashvsr_plus` 或本节点的 `ckpts`，也可在加载器指定根目录。完整文件结构见 [模型仓库](https://huggingface.co/t8star/Flashvsr-Plus-Comfy)。下载使用固定 revision 与 SHA-256 校验。

- **INT8 ConvRot**：预转换 DiT 每个约 **1.45 GB**，配合对应 LQ 与解码器即可运行；无需再下载原始 DiT。仅 transformer block 的 Linear 使用 Comfy Kitchen W8A8，其他组件保留浮点计算。
- **注意力**：支持官方 BSA、Triton 和 SDPA。`auto` 按可用性选择；高清建议使用 Triton。SDPA 会在分配过大的掩码之前报错。BSA 原生内核未在本机 Windows 验证。
- **分段解码**：Tiny 的 `decode_every` 使用 4 的正整数倍，例如 `4`；Wan VAE 设置为 `0`，使用全片解码。

INT8 主要减少权重显存，速度与画质取决于内容和硬件。IMAGE 输出需整批 RAM；VIDEO 按恒定帧率处理，VFR 素材请先规范为 CFR。PCM 等无法直接写入 MP4 的音频会转为 AAC。每次请求后权重回 CPU；`keep_loaded` 保留 CPU 副本。

## 验证

**45 项回归测试通过**。Turbo / Tiny / Triton 的 BF16 和 INT8 均完成 **1920×1080、5 秒、24 FPS、120 帧**真实 ComfyUI 验证，逐帧解码、时间戳及音频检查通过。`transformers==4.57.1` 另有完整 GPU 工作流验证。[验证范围](docs/VALIDATION.md) · [1080p 实测片段](assets/validation_1080p_5s.mp4)

## 社区与资源

[B站](https://space.bilibili.com/385085361) · [YouTube](https://www.youtube.com/@T8star-Aix/) · [API](https://api.seedance.nz/sign-up?aff=5f4w) · [免费画廊](https://www.openzhenzhen.com) · [在线 AI 应用](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121) · [ComfyUI 整合包](https://pan.quark.cn/s/264edb7e36bd) · [Hugging Face](https://huggingface.co/t8star)

## 致谢与许可

基于 [OpenImagingLab/FlashVSR_Plus](https://github.com/OpenImagingLab/FlashVSR_Plus) 与 [FlashVSR V1.1](https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1)。INT8 线性运算使用 [Comfy Kitchen](https://github.com/Comfy-Org/comfy-kitchen)。ComfyUI 集成与派生模型由 T8star 维护。代码与模型采用 Apache-2.0；详见 [LICENSE](LICENSE) 和 [NOTICE](NOTICE)。
