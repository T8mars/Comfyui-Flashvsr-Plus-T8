"""Stable public import surface for FlashVSR+ inference.

The implementation lives in this private namespace. The root ``inference``
package preserves the original CLI and public entry point.
"""

from .inference import (
    ColorCorrectionConfig,
    DecoderConfig,
    FlashVSRInferencePipeline,
    InferenceConfig,
    InferenceRequest,
    InferenceResult,
    ModelBundle,
    ModelPaths,
    StreamConfig,
    build_inference_pipeline,
    stream_config_for_mode,
)

__all__ = [
    "ColorCorrectionConfig",
    "DecoderConfig",
    "FlashVSRInferencePipeline",
    "InferenceConfig",
    "InferenceRequest",
    "InferenceResult",
    "ModelBundle",
    "ModelPaths",
    "StreamConfig",
    "build_inference_pipeline",
    "stream_config_for_mode",
]

__version__ = "0.2.0"
