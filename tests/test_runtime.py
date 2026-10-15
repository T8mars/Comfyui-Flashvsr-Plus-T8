import dataclasses
import json
import os
import tempfile
import unittest
import hashlib
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from flashvsr_plus.backends import validate_attention_size
from flashvsr_plus.comfy_runtime import Options, geometry, image_to_model
from flashvsr_plus.inference import InferenceRequest
from flashvsr_plus.inference.pipeline import FlashVSRInferencePipeline
from flashvsr_plus.inference.presets import stream_config_for_mode
from flashvsr_plus.inference.preprocessing.input import _frames_to_tensor
from flashvsr_plus.quantization import ConvRotLinear, convert_checkpoint, apply_quantized_checkpoint
from flashvsr_plus.weights import download_file, verified_model


class GeometryTests(unittest.TestCase):
    def test_handle_guards_the_resized_canvas_before_model_construction(self):
        from flashvsr_plus.comfy_runtime import ModelHandle
        handle = ModelHandle(device="cpu", attention="sdpa")
        class Built(Exception):
            pass
        with patch("flashvsr_plus.comfy_runtime.resolve_backend", return_value="sdpa"), \
                patch.object(handle, "_build", side_effect=Built()) as build, \
                patch("torch.cuda.is_available", return_value=False), patch("torch.cuda.device_count", return_value=0):
            with self.assertRaisesRegex(RuntimeError, "SDPA cannot process"):
                handle.infer(InferenceRequest(lq_video=torch.zeros(1, 3, 1, 128, 128), height=1152, width=1920))
            build.assert_not_called()
            with self.assertRaises(Built):
                handle.infer(InferenceRequest(lq_video=torch.zeros(1, 3, 1, 16, 24), height=128, width=128))
            build.assert_called_once()

    def test_invalid_requests_are_rejected_before_backend_and_loading(self):
        from flashvsr_plus.comfy_runtime import ModelHandle
        source = torch.zeros(1, 3, 1, 128, 128)
        requests = [InferenceRequest(lq_video=source, **values) for values in (
            {"fps": float("nan")}, {"fps": float("inf")}, {"max_frames": 1.5},
            {"height": -1, "width": 128}, {"height": 128.5, "width": 128},
            {"height": True, "width": 128}, {"input_path": "duplicate.mp4"})]
        requests += [InferenceRequest(lq_video=source[:, :, :0]),
                     InferenceRequest(lq_video=torch.zeros(1, 2, 5, 128, 128))]
        handle = ModelHandle(device="cpu")
        with patch("flashvsr_plus.comfy_runtime.resolve_backend") as backend, patch.object(handle, "_build") as build:
            for request in requests:
                with self.subTest(request=request):
                    with self.assertRaises(ValueError):
                        handle.infer(request)
            backend.assert_not_called()
            build.assert_not_called()
        import numpy as np
        InferenceRequest(lq_video=source, height=np.int64(19), width=27, fps=7.1234567).validate()

    def test_both_clis_reject_invalid_requests_before_loading_or_cuda(self):
        from flashvsr_plus.inference import cli, run_stream
        for module in (cli, run_stream):
            with self.subTest(module=module.__name__), \
                    patch.object(sys, "argv", ["flashvsr", "--input", "missing.mp4", "--output", "unused.mp4", "--height", "-1", "--width", "128"]), \
                    patch.object(module, "build_inference_pipeline") as build, \
                    patch("torch.cuda.is_available") as cuda:
                with self.assertRaises(ValueError):
                    module.main()
                build.assert_not_called()
                cuda.assert_not_called()

    def test_tensor_content_canvas_resizes_then_pads_in_both_input_modes(self):
        from flashvsr_plus.inference.preprocessing.input import load_lq_video, iter_lq_video, _resize_tensor_video
        source = torch.linspace(-1, 1, 5*31*47).reshape(1, 1, 5, 31, 47).expand(1, 3, 5, 31, 47).clone()
        request = InferenceRequest(lq_video=source, height=128, width=128, content_height=19, content_width=27)
        expected = _resize_tensor_video(source, 19, 27)
        eager, _ = load_lq_video(request)
        streamed = torch.cat(list(iter_lq_video(request, chunk_frames=2)), dim=2)
        self.assertTrue(torch.equal(eager, streamed))
        self.assertTrue(torch.equal(eager[..., :19, :27], expected))
        self.assertTrue(torch.equal(eager[..., -1:, -1:], expected[..., -1:, -1:]))

    def test_image_output_does_not_mutate_model_result(self):
        from flashvsr_plus.comfy_runtime import upscale_images
        for dtype in (torch.float32, torch.bfloat16):
            video = -torch.ones(1, 3, 3, 128, 128, dtype=dtype)
            result = SimpleNamespace(video=video, metadata={})
            model = SimpleNamespace(device="cpu", attention="sdpa", infer=lambda *a, **kw: result)
            with patch("flashvsr_plus.comfy_runtime.resolve_backend", return_value="sdpa"):
                images, returned = upscale_images(model, torch.zeros(3, 16, 16, 3), scale=1)
            self.assertTrue(torch.equal(video, -torch.ones_like(video)))
            self.assertEqual(images.shape, (3, 16, 16, 3))
            self.assertTrue(images.is_contiguous())
            self.assertTrue(torch.equal(images, torch.zeros_like(images)))
            self.assertIs(returned, result)

    def test_extreme_aspect_crop_keeps_real_pixel_content(self):
        import numpy as np
        from flashvsr_plus.inference.preprocessing.input import pil_center_crop_resize
        for shape, height, width in (((1, 1, 512, 3), 128, 1), ((1, 512, 1, 3), 1, 128)):
            original = np.full(shape, 255, dtype=np.uint8)
            result = pil_center_crop_resize(original, height, width)
            self.assertEqual(result.shape, (1, height, width, 3))
            self.assertTrue(np.all(result == 255))

    def test_gaussian_temporal_merge_is_finite_at_small_sigma(self):
        from flashvsr_plus.inference.preprocessing.temporal import _gaussian_weights
        from flashvsr_plus.inference.runners.stream import _merge_segment_groups
        for factor in (2, 4):
            for sigma in (.01, .1, 1., 10., 1e-300, 1e300):
                weights = _gaussian_weights(factor, sigma, device=torch.device("cpu"))
                self.assertTrue(torch.isfinite(weights).all())
                self.assertAlmostEqual(float(weights.sum()), 1., places=6)
                result = _merge_segment_groups(torch.ones(1, 3, factor+1, 2, 2),
                    factor, first=True, mode="gaussian", sigma=sigma)
                self.assertTrue(torch.isfinite(result).all())
                self.assertTrue(torch.equal(result, torch.ones_like(result)))
        for sigma in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                Options(temporal_gaussian_sigma=sigma).validate()

    def test_batch_video_conversion_preserves_three_frame_channel_order(self):
        import numpy as np
        from flashvsr_plus.inference.output import tensor_to_uint8_frames
        for width in (3, 16):
            video = -torch.ones(1, 3, 3, 8, width)
            for i in range(3):
                video[0, i, i] = 1
            frames = tensor_to_uint8_frames(video)
            self.assertEqual(frames.shape, (3, 8, width, 3))
            np.testing.assert_array_equal(frames[:, 0, 0], np.eye(3, dtype=np.uint8)*255)

    def test_fractional_fps_survives_video_encoding(self):
        import av
        from fractions import Fraction
        from flashvsr_plus.inference.output import save_tensor_video
        for rate in (Fraction(24000, 1001), Fraction(30000, 1001), Fraction(60000, 1001)):
            with self.subTest(fps=rate), tempfile.TemporaryDirectory() as d:
                path = Path(d)/"fractional.mp4"
                artifact = save_tensor_video(torch.zeros(1, 3, 7, 16, 24), path, float(rate))
                with av.open(str(path)) as container:
                    self.assertEqual(container.streams.video[0].average_rate, rate)
                self.assertAlmostEqual(artifact["fps"], float(rate), places=7)

    def test_high_frequency_color_is_not_rejected_by_lossy_encoder(self):
        from flashvsr_plus.inference.output import save_tensor_video
        video = torch.randint(0, 256, (1, 3, 1, 16, 24), generator=torch.Generator().manual_seed(42)).float()/127.5-1
        with tempfile.TemporaryDirectory() as d:
            artifact = save_tensor_video(video, Path(d)/"texture.mp4", fps=24)
        self.assertEqual((artifact["frames"], artifact["width"], artifact["height"]), (1, 24, 16))
        self.assertTrue(artifact["first_frame_color_mae"] >= 0)

    def test_odd_tiny_video_dimensions_and_rgb_roundtrip(self):
        import cv2
        from flashvsr_plus.inference.output import save_tensor_video
        video = -torch.ones(1, 3, 1, 5, 3)
        video[:, 0] = 1
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"odd.mp4"
            artifact = save_tensor_video(video, path, fps=8)
            self.assertEqual((artifact["width"], artifact["height"], artifact["frames"]), (3, 5, 1))
            self.assertEqual(artifact["pixel_format"], "yuv444p")
            capture = cv2.VideoCapture(str(path))
            ok, bgr = capture.read()
            capture.release()
            self.assertTrue(ok)
            self.assertGreater(bgr[..., 2].mean(), 240)
            self.assertLess(bgr[..., :2].mean(), 10)

    def test_exact_target_and_grid(self):
        self.assertEqual(geometry(480, 270, 4), (1920, 1080, 1920, 1152))
        self.assertEqual(geometry(480, 270, 4, width=960), (960, 540, 1024, 640))
        self.assertEqual(geometry(480, 270, 4, height=2160), (3840, 2160, 3840, 2176))

    def test_three_frames_not_confused_with_channels(self):
        x = torch.stack([torch.full((12, 19, 3), v) for v in (0., .5, 1.)])
        y = image_to_model(x, 19, 12, 128, 128)
        self.assertEqual(tuple(y.shape), (1, 3, 3, 128, 128))
        self.assertEqual(y[0, :, 0].unique().item(), -1)
        self.assertEqual(y[0, :, 1].unique().item(), 0)
        self.assertEqual(y[0, :, 2].unique().item(), 1)

    def test_bad_images_rejected(self):
        for x in (torch.zeros(0, 12, 19, 3), torch.zeros(1, 12, 19, 4), torch.full((1, 4, 4, 3), float("nan"))):
            with self.assertRaises(ValueError):
                image_to_model(x, 19, 12, 128, 128)

    def test_aspect_change_uses_official_center_crop(self):
        x = torch.zeros(1, 4, 8, 3)
        x[:, :, 2:6] = 1
        y = image_to_model(x, 4, 4, 128, 128)
        self.assertTrue(torch.equal(y, torch.ones_like(y)))

    def test_frame_file_padding_preserves_content(self):
        import numpy as np
        a = np.full((3, 12, 19, 3), 128, dtype=np.uint8)
        req = InferenceRequest(input_path="not-read", height=128, width=128, content_height=12, content_width=19)
        y = _frames_to_tensor(a, req, stream=True)
        self.assertEqual(tuple(y.shape), (1, 3, 3, 128, 128))
        self.assertTrue(torch.allclose(y, torch.full_like(y, 128/127.5-1), atol=1e-7))

    def test_full_cache_limit_preflight(self):
        validate_attention_size(384, 768, "sdpa")
        for h, w in ((512, 1024), (1024, 1920), (2176, 3840)):
            with self.assertRaisesRegex(RuntimeError, "full cache"):
                validate_attention_size(h, w, "sdpa")
            validate_attention_size(h, w, "triton_block_sparse")

    def test_options_and_vae_validation(self):
        Options().validate()
        for option in (Options(decode_every=3), Options(temporal_compression=3), Options(vae_overlap=30)):
            with self.assertRaises(ValueError):
                option.validate()


class DecoderTests(unittest.TestCase):
    def test_untiled_vae_uses_model_dtype_and_returns_cpu_output(self):
        from flashvsr_plus.inference.decoders.vae import VAEDecoder
        class VAE(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv3d(16, 3, 1).bfloat16()
            def stream_decode(self, value):
                self.last_dtype, self.last_device = value.dtype, value.device
                return self.conv(value)
            def decode(self, value, **kwargs):
                return self.stream_decode(value)
        module = VAE()
        with torch.inference_mode():
            result = VAEDecoder(module, tiled=False).decode(torch.zeros(1, 16, 2, 2, 2), stream=True)
        self.assertEqual(module.last_dtype, torch.bfloat16)
        self.assertEqual(result.video.device.type, "cpu")
        self.assertEqual(result.video.shape, (1, 3, 2, 2, 2))


class AudioTests(unittest.TestCase):
    def test_pcm_source_audio_is_encoded_when_mp4_cannot_copy_it(self):
        import av
        import imageio_ffmpeg
        from flashvsr_plus.inference.output import mux_source_audio
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source, target = root/"source.mov", root/"result.mp4"
            subprocess.run([ffmpeg, "-y", "-f", "lavfi", "-i", "color=size=24x16:rate=8:duration=1.125",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=1.125", "-c:v", "libx264",
                "-c:a", "pcm_s16le", str(source)], check=True, capture_output=True)
            mux_source_audio(source, source, target)
            with av.open(str(target)) as container:
                self.assertEqual(container.streams.audio[0].codec_context.name, "aac")
                self.assertEqual(sum(1 for _ in container.decode(video=0)), 9)
            with av.open(str(target)) as container:
                stream = container.streams.audio[0]
                duration = sum(frame.samples for frame in container.decode(audio=0))/stream.rate
                self.assertAlmostEqual(duration, 1.125, delta=.1)

    def test_short_audio_keeps_all_frames_and_long_audio_is_trimmed(self):
        import av
        import cv2
        import imageio_ffmpeg
        from flashvsr_plus.inference.output import mux_source_audio
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            video = directory/"video.mp4"
            subprocess.run([ffmpeg, "-y", "-f", "lavfi", "-i", "color=c=red:s=24x16:r=24:d=5",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)], check=True, capture_output=True)
            for seconds in (.3, 8):
                with self.subTest(audio_duration=seconds):
                    audio, output = directory/f"source_{seconds}.m4a", directory/f"output_{seconds}.mp4"
                    subprocess.run([ffmpeg, "-y", "-f", "lavfi", "-i",
                        f"sine=frequency=440:sample_rate=48000:duration={seconds}", "-c:a", "aac", str(audio)],
                        check=True, capture_output=True)
                    mux_source_audio(video, audio, output)
                    cap = cv2.VideoCapture(str(output))
                    self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 120)
                    self.assertEqual(cap.get(cv2.CAP_PROP_FPS), 24)
                    cap.release()
                    with av.open(str(output)) as container:
                        stream = container.streams.audio[0]
                        duration = sum(frame.samples for frame in container.decode(audio=0))/stream.rate
                    self.assertAlmostEqual(duration, min(seconds, 5), delta=.1)


class CancellationTests(unittest.TestCase):
    def test_home_paths_are_expanded_for_both_encoding_and_audio_mux(self):
        from flashvsr_plus.inference.output import save_tensor_video
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            video = torch.zeros(1, 3, 5, 128, 128)
            save_tensor_video(video, root/"source.mp4", 8.)
            config = stream_config_for_mode("turbo", device="cpu", hardware_profile="generic")
            pipe = FlashVSRInferencePipeline(config, SimpleNamespace())
            pipe.runner = SimpleNamespace(run=lambda *a, **kw: SimpleNamespace(video=video, metadata={}))
            with patch.dict(os.environ, {"USERPROFILE": str(root), "HOME": str(root)}):
                result = pipe.infer(InferenceRequest(input_path="~/source.mp4", output_path="~/nested/restored.mp4"))
            self.assertEqual(Path(result.output_path), root/"nested/restored.mp4")
            self.assertEqual(result.frames, 5)
            self.assertTrue((root/"nested/restored.mp4").is_file())
            self.assertEqual(list((root/"nested").iterdir()), [root/"nested/restored.mp4"])

    def test_handle_interrupt_stops_tiny_full_clip_before_next_microstep(self):
        from flashvsr_plus.comfy_runtime import ModelHandle
        from flashvsr_plus.inference.decoders.tiny import TinyDecoder
        class Interrupt(BaseException):
            pass
        cancelled = [False]
        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))
                self.mem, self.calls = [None], 0
            def clean_mem(self):
                self.mem = [None]
            def decode_video(self, value, *, cond, **kwargs):
                self.calls += 1
                cancelled[0] = True
                return torch.full_like(cond.transpose(1, 2), .5)
        module = Module()
        decoder = TinyDecoder(module, parallel_mode="serial", channels_last=False)
        bundle = SimpleNamespace(denoiser=torch.nn.Linear(1, 1), decoder=decoder)
        def infer(request, **kwargs):
            value = decoder.decode(torch.zeros(1, 16, 8, 2, 2), lq_video=torch.zeros(1, 3, 29, 16, 16)).video
            kwargs["progress"]("decoder complete")
            return SimpleNamespace(video=value, metadata={})
        pipeline = SimpleNamespace(bundle=bundle, infer=infer)
        def interrupt():
            if cancelled[0]:
                raise Interrupt()
        handle = ModelHandle(device="cpu", attention="sdpa")
        with patch("flashvsr_plus.comfy_runtime.resolve_backend", return_value="sdpa"), \
                patch.object(handle, "_build", return_value=pipeline), \
                patch("torch.cuda.is_available", return_value=False), patch("torch.cuda.device_count", return_value=0):
            with self.assertRaises(Interrupt):
                handle.infer(InferenceRequest(lq_video=torch.zeros(1, 3, 29, 128, 128)),
                    options=Options(decode_every=0), interrupt=interrupt)
        self.assertEqual(module.calls, 1)
        self.assertEqual(module.mem, [None])
        self.assertIsNone(getattr(decoder, "interrupt_callback", None))

    def test_mux_failure_cleans_encoded_intermediate(self):
        config = stream_config_for_mode("turbo", device="cpu", hardware_profile="generic")
        pipe = FlashVSRInferencePipeline(config, SimpleNamespace())
        pipe.runner = SimpleNamespace(run=lambda *a, **kw: SimpleNamespace(video=torch.zeros(1, 3, 1, 128, 128), metadata={}))
        def save(video, path, *a, **kw):
            Path(path).write_bytes(b"encoded intermediate")
            return {"frames": 1, "path": str(path)}
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            source = directory/"input.mp4"
            source.write_bytes(b"source fixture")
            with patch("flashvsr_plus.inference.pipeline.load_lq_video", return_value=(torch.zeros(1, 3, 1, 128, 128), 8.)), \
                    patch("flashvsr_plus.inference.pipeline.save_tensor_video", side_effect=save), \
                    patch("flashvsr_plus.inference.pipeline.mux_source_audio", side_effect=RuntimeError("mux failed")):
                with self.assertRaisesRegex(RuntimeError, "mux failed"):
                    pipe.infer(InferenceRequest(input_path=source, output_path=directory/"result.mp4"))
            self.assertEqual(list(directory.iterdir()), [source])

    def test_cache_policy_reason_matches_selected_residency(self):
        from flashvsr_plus.inference.streaming.memory_policy import choose_stream_cache_offload
        config = SimpleNamespace(is_hybrid_sliding=True, stream_cache_offload_mode="auto")
        model = SimpleNamespace(blocks=[SimpleNamespace(self_attn=object())])
        video = torch.empty(1, 3, 1, 2176, 3840, device="meta")
        # At this size the reference extrapolation can predict a larger offload
        # plan; the selected GPU-resident plan must not report offloading.
        with patch("flashvsr_plus.inference.streaming.memory_policy._memory_info", return_value=(24000., 24000.)):
            result = choose_stream_cache_offload(config, model, video, torch.device("cuda:0"))
        self.assertFalse(result.enabled)
        self.assertEqual(result.reason, "estimated peak fits available VRAM")

    def test_baseexception_closes_incremental_writer(self):
        class Interrupt(BaseException):
            pass
        writers = []
        class Writer:
            def __init__(self, *a, **kw):
                self.closed = False
                writers.append(self)
            def close(self):
                self.closed = True
        cfg = stream_config_for_mode("turbo", device="cpu", hardware_profile="generic", decode_every=4)
        pipe = FlashVSRInferencePipeline(cfg, SimpleNamespace())
        pipe.runner = SimpleNamespace(run=lambda *a, **kw: (_ for _ in ()).throw(Interrupt()))
        with tempfile.TemporaryDirectory() as d, patch("flashvsr_plus.inference.pipeline.IncrementalVideoWriter", Writer):
            with self.assertRaises(Interrupt):
                pipe.infer(InferenceRequest(lq_video=torch.zeros(1, 3, 5, 128, 128), output_path=Path(d)/"out.mp4"))
        self.assertEqual(len(writers), 1)
        self.assertTrue(writers[0].closed)

    def test_bsa_adapter_preserves_compact_mask_contract(self):
        from flashvsr_plus.models.attention_backends.kernels import _run_block_sparse
        q = torch.randn(1, 2, 128, 128)
        k, v = torch.randn(1, 2, 256, 128), torch.randn(1, 2, 256, 128)
        mask = torch.tensor([[[[True, False]], [[False, True]]]])
        calls = []
        def fake_bsa(qf, kf, vf, cu_q, cu_k, head_types, streaming, block_mask,
                     max_q, max_k, dropout, **kwargs):
            calls.append((block_mask, kwargs))
            self.assertEqual((max_q, max_k), (128, 256))
            self.assertEqual(cu_q.tolist(), [0, 128])
            self.assertEqual(cu_k.tolist(), [0, 256])
            self.assertEqual(head_types.tolist(), [1, 1])
            dense_mask = block_mask.repeat_interleave(128, -2).repeat_interleave(128, -1)
            value = torch.nn.functional.scaled_dot_product_attention(
                qf.transpose(0, 1).unsqueeze(0), kf.transpose(0, 1).unsqueeze(0),
                vf.transpose(0, 1).unsqueeze(0), attn_mask=dense_mask)
            return value[0].transpose(0, 1)
        with patch("flashvsr_plus.models.attention_backends.kernels._require_kernel",
                   return_value=SimpleNamespace(block_sparse_attn_func=fake_bsa)):
            out = _run_block_sparse(q, k, v, mask)
        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v,
            attn_mask=mask.repeat_interleave(128, -2).repeat_interleave(128, -1))
        self.assertTrue(torch.equal(out, ref))
        self.assertTrue(torch.equal(calls[0][0], mask))
        self.assertFalse(calls[0][1]["is_causal"])


class QuantizationTests(unittest.TestCase):
    def test_conversion_expands_home_in_function_cli_and_receipt(self):
        from safetensors.torch import save_file
        from flashvsr_plus.quantization import kitchen, read_quantization
        kitchen()
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList([torch.nn.Linear(16, 8)])
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root/"models/turbo/dit.safetensors"
            source.parent.mkdir(parents=True)
            save_file(Toy().state_dict(), str(source))
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            with patch.dict(os.environ, {"USERPROFILE": str(root), "HOME": str(root)}):
                report = convert_checkpoint("~/models/turbo/dit.safetensors", "~/direct.safetensors", mode="turbo")
                self.assertEqual(report["source_sha256"], digest)
                with self.assertRaisesRegex(ValueError, "overwrite"):
                    convert_checkpoint("~/models/turbo/dit.safetensors", source, mode="turbo")
                subprocess.run([sys.executable, "-B", "-m", "flashvsr_plus.convert", "--modes", "turbo",
                    "--model-dir", "~/models", "--device", "cpu"], check=True, capture_output=True)
            target = source.with_name("dit_int8_convrot.safetensors")
            self.assertEqual(read_quantization(target)["source_sha256"], digest)
            self.assertEqual(json.loads(target.with_suffix(".conversion.json").read_text())["source_sha256"], digest)
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), digest)

    def test_meta_loader_expands_home_before_checking_checkpoint_format(self):
        from safetensors.torch import save_file
        from flashvsr_plus.inference.loading import ComponentFactories, ModelPaths, loader
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList([torch.nn.Linear(16, 8)])
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source, quantized = root/"source.safetensors", root/"quantized.safetensors"
            original = Toy()
            save_file(original.state_dict(), str(source))
            convert_checkpoint(source, quantized, mode="turbo")
            config = stream_config_for_mode("turbo", device="cpu", hardware_profile="generic")
            with patch.dict(os.environ, {"USERPROFILE": str(root), "HOME": str(root)}), \
                    patch.object(loader, "build_model", side_effect=lambda config: Toy()):
                for path in (source, quantized):
                    model = loader._build_denoiser(config, ModelPaths(dit="~/"+path.name), ComponentFactories())
                    self.assertFalse(any(value.is_meta for value in model.state_dict().values()))
                    self.assertEqual(isinstance(model.blocks[0], ConvRotLinear), path == quantized)
                    if path == source:
                        self.assertTrue(torch.equal(model.blocks[0].weight, original.blocks[0].weight))

    def test_convrot_preserves_empty_batch_shape(self):
        layer = ConvRotLinear(torch.zeros(32, 64, dtype=torch.int8), torch.ones(32), None, 64)
        for shape in ((0, 64), (2, 0, 64), (0, 3, 64)):
            result = layer(torch.empty(shape, dtype=torch.bfloat16))
            self.assertEqual(result.shape, (*shape[:-1], 32))
            self.assertEqual(result.dtype, torch.bfloat16)

    def test_concurrent_conversion_uses_separate_temporary_files(self):
        from concurrent.futures import ThreadPoolExecutor
        from safetensors import safe_open
        from safetensors.torch import save_file
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList([torch.nn.Linear(64, 32)])
        with tempfile.TemporaryDirectory() as d:
            source, target = Path(d)/"source.safetensors", Path(d)/"int8.safetensors"
            save_file(Toy().state_dict(), str(source))
            barrier = threading.Barrier(2)
            temporary_paths = []
            def synchronized_save(tensors, filename, **kwargs):
                save_file(tensors, filename, **kwargs)
                temporary_paths.append(filename)
                barrier.wait(timeout=20)
            with patch("safetensors.torch.save_file", side_effect=synchronized_save), ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(convert_checkpoint, source, target, mode="turbo") for _ in range(2)]
                for future in futures:
                    self.assertGreater(future.result(timeout=30)["output_bytes"], 0)
            self.assertEqual(len(set(temporary_paths)), 2)
            self.assertFalse(list(Path(d).glob("*.partial")))
            with safe_open(str(target), framework="pt", device="cpu") as saved:
                self.assertEqual(saved.get_tensor("blocks.0.weight").dtype, torch.int8)

    def test_convrot_rejects_nonfloating_or_nonfinite_bias_and_scale(self):
        weight = torch.zeros(32, 64, dtype=torch.int8)
        for bias in (torch.zeros(32, dtype=torch.int8), torch.zeros(32, dtype=torch.float64), torch.full((32,), float("nan")), torch.zeros(17)):
            with self.assertRaisesRegex(ValueError, "bias"):
                ConvRotLinear(weight, torch.ones(32), bias, 64)
        with self.assertRaisesRegex(ValueError, "scales"):
            ConvRotLinear(weight, torch.ones(32, dtype=torch.int8), None, 64)
        for magnitude in (1e300, 1e-300):
            with self.assertRaisesRegex(ValueError, "scales"):
                ConvRotLinear(weight, torch.full((32,), magnitude, dtype=torch.float64), None, 64)

    def test_quantization_metadata_cannot_change_weight_interpretation(self):
        import copy
        from safetensors import safe_open
        from safetensors.torch import load_file, save_file
        from flashvsr_plus.quantization import read_quantization
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(64, 32))])
        with tempfile.TemporaryDirectory() as d:
            source, target, corrupt = [Path(d)/f"{name}.safetensors" for name in ("source", "int8", "bad")]
            save_file(Toy().state_dict(), str(source))
            convert_checkpoint(source, target, mode="turbo")
            with safe_open(str(target), framework="pt", device="cpu") as f:
                metadata = f.metadata()
            original = json.loads(metadata["flashvsr_plus_quantization"])
            state = load_file(str(target))
            variants = []
            for key, value in (("format", "other"), ("mode", "unknown"), ("source_sha256", "wrong")):
                spec = copy.deepcopy(original)
                spec[key] = value
                variants.append(spec)
            for key, value in (("format", "other"), ("convrot", False), ("convrot_groupsize", 0), ("convrot_groupsize", 4.0)):
                spec = copy.deepcopy(original)
                spec["layers"]["blocks.0.0"][key] = value
                variants.append(spec)
            for spec in variants:
                modified = dict(metadata, flashvsr_plus_quantization=json.dumps(spec))
                save_file(state, str(corrupt), metadata=modified)
                with self.assertRaisesRegex(ValueError, "quantization"):
                    read_quantization(corrupt)

    def test_quantized_checkpoint_rejects_missing_or_wrong_bias(self):
        from safetensors import safe_open
        from safetensors.torch import load_file, save_file
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(64, 32))])
        with tempfile.TemporaryDirectory() as d:
            source, target, corrupt = [Path(d)/f"{name}.safetensors" for name in ("source", "int8", "bad")]
            save_file(Toy().state_dict(), str(source))
            convert_checkpoint(source, target, mode="turbo")
            with safe_open(str(target), framework="pt", device="cpu") as f:
                metadata = f.metadata()
            for bias in (None, torch.zeros(17)):
                state = load_file(str(target))
                state.pop("blocks.0.0.bias")
                if bias is not None:
                    state["blocks.0.0.bias"] = bias
                save_file(state, str(corrupt), metadata=metadata)
                with self.assertRaisesRegex(ValueError, "bias mismatch"):
                    apply_quantized_checkpoint(Toy(), corrupt)

    def test_meta_checkpoint_loading_materializes_exact_parameters(self):
        from flashvsr_plus.model_io import load_model_checkpoint
        original = torch.nn.Linear(64, 32)
        with torch.device("meta"):
            target = torch.nn.Linear(64, 32)
        self.assertTrue(target.weight.is_meta)
        load_model_checkpoint(target, original.state_dict(), min_coverage=1.0, assign=True)
        self.assertFalse(target.weight.is_meta)
        self.assertTrue(torch.equal(target.weight, original.weight))
        self.assertTrue(torch.equal(target.bias, original.bias))

    def test_row_chunking_matches_native_convrot_execution(self):
        from comfy_kitchen.backends.eager.quantization import quantize_int8_convrot_weight
        from flashvsr_plus.quantization import kitchen
        torch.manual_seed(18)
        weight = torch.randn(32, 64)*.1
        q, scale = quantize_int8_convrot_weight(weight, 64)
        layer = ConvRotLinear(q, scale, None, 64)
        x = torch.randn(2, 57, 64).bfloat16()
        ref = kitchen().int8_linear(x, q, scale, None, x.dtype, convrot=True, convrot_groupsize=64)
        with patch("flashvsr_plus.quantization.LINEAR_ROW_CHUNK", 16):
            result = layer(x)
        self.assertTrue(torch.equal(result, ref))

    def test_roundtrip_layer_scales_and_execution(self):
        from safetensors.torch import save_file
        torch.manual_seed(123)
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(64, 32))])
                self.head = torch.nn.Linear(32, 16)
            def forward(self, x):
                return self.head(self.blocks[0](x))
        original, target = Toy(), Toy()
        with tempfile.TemporaryDirectory() as d:
            src, dst = Path(d)/"original.safetensors", Path(d)/"quant.safetensors"
            save_file(original.state_dict(), str(src))
            report = convert_checkpoint(src, dst, mode="turbo")
            self.assertEqual(len(report["layers"]), 1)
            apply_quantized_checkpoint(target, dst)
            layer = target.blocks[0][0]
            self.assertIsInstance(layer, ConvRotLinear)
            scales = layer.weight_scale.clone()
            target.to(dtype=torch.bfloat16)
            self.assertEqual(layer.weight.dtype, torch.int8)
            self.assertEqual(layer.weight_scale.dtype, torch.float32)
            self.assertTrue(torch.equal(scales, layer.weight_scale))
            x = torch.randn(80, 64).bfloat16()
            ref, got = original.bfloat16()(x).float(), target(x).float()
            error = (ref-got).norm()/ref.norm()
            self.assertLess(float(error), .04)


class ModelDownloadTests(unittest.TestCase):
    def test_int8_loader_downloads_pinned_derived_file_without_original_dit(self):
        from safetensors.torch import save_file
        from flashvsr_plus import weights, comfy_runtime as runtime
        from flashvsr_plus.comfy_runtime import ModelHandle
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList([torch.nn.Linear(16, 8)])
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source, converted = root/"source.safetensors", root/"converted.safetensors"
            save_file(Toy().state_dict(), str(source))
            convert_checkpoint(source, converted, mode="turbo")
            bodies = {"turbo/dit.safetensors": source.read_bytes(),
                      "turbo/dit_int8_convrot.safetensors": converted.read_bytes(),
                      "turbo/lq_proj_in.safetensors": b"projection", "TCDecoder.ckpt": b"decoder"}
            def entry(role, name):
                body = bodies[name]
                return (role, name, "t8star/Flashvsr-Plus-Comfy", "pinned-revision", name,
                        len(body), hashlib.sha256(body).hexdigest())
            official = [entry("turbo_dit", "turbo/dit.safetensors"),
                        entry("turbo_lq", "turbo/lq_proj_in.safetensors"), entry("tiny", "TCDecoder.ckpt")]
            derived = [entry("turbo_int8_convrot", "turbo/dit_int8_convrot.safetensors")]
            requested = []
            def get(url, **kwargs):
                prefix = "https://huggingface.co/t8star/Flashvsr-Plus-Comfy/resolve/pinned-revision/"
                self.assertTrue(url.startswith(prefix))
                name = url.removeprefix(prefix)
                self.assertNotEqual(name, "turbo/dit.safetensors")
                requested.append(name)
                return self.response(bodies[name])
            target = root/"models"
            handle = ModelHandle(precision="int8_convrot", device="cpu", directory=target, auto_download=True)
            sentinel = object()
            with patch.object(weights, "FILES", official), patch.object(weights, "INT8_FILES", derived), \
                    patch.object(runtime, "FILES", official), patch("requests.get", side_effect=get), \
                    patch.object(runtime, "build_inference_pipeline", return_value=sentinel) as build:
                self.assertIs(handle._build(stream_config_for_mode("turbo", device="cpu"), lambda message: None), sentinel)
            self.assertEqual(set(requested), {"turbo/dit_int8_convrot.safetensors", "turbo/lq_proj_in.safetensors", "TCDecoder.ckpt"})
            self.assertFalse((target/"turbo/dit.safetensors").exists())
            self.assertEqual(Path(build.call_args.args[1].dit), target/"turbo/dit_int8_convrot.safetensors")

    def test_existing_int8_model_does_not_require_original_dit(self):
        from safetensors.torch import save_file
        from flashvsr_plus.comfy_runtime import ModelHandle
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList([torch.nn.Linear(64, 32)])
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/"turbo").mkdir()
            source, target = root/"turbo/dit.safetensors", root/"turbo/dit_int8_convrot.safetensors"
            save_file(Toy().state_dict(), str(source))
            spec = convert_checkpoint(source, target, mode="turbo")
            entries = [("turbo_dit", "turbo/dit.safetensors", "repo", "rev", "dit", source.stat().st_size, spec["source_sha256"])]
            for role, relative in (("turbo_lq", "turbo/lq_proj_in.safetensors"), ("tiny", "TCDecoder.ckpt")):
                body = role.encode()
                (root/relative).write_bytes(body)
                entries.append((role, relative, "repo", "rev", relative, len(body), hashlib.sha256(body).hexdigest()))
            source.unlink()
            config = stream_config_for_mode("turbo", device="cpu", hardware_profile="generic")
            with patch("flashvsr_plus.weights.FILES", entries), patch("flashvsr_plus.comfy_runtime.FILES", entries), \
                    patch("flashvsr_plus.comfy_runtime.build_inference_pipeline", return_value=object()) as build:
                ModelHandle(precision="int8_convrot", directory=root, auto_download=False, device="cpu")._build(config, lambda _: None)
                self.assertEqual(build.call_args.args[1].dit, str(target))
                with self.assertRaises(FileNotFoundError):
                    ModelHandle(precision="bf16", directory=root, auto_download=False, device="cpu")._build(config, lambda _: None)

    def test_verified_weights_load_when_optional_receipt_cannot_be_written(self):
        body = b"valid shared read-only model"
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"model.safetensors"
            path.write_bytes(body)
            with patch.object(Path, "write_text", side_effect=PermissionError("read-only model directory")):
                verified_model(path, len(body), hashlib.sha256(body).hexdigest())
                with self.assertRaisesRegex(ValueError, "checksum"):
                    verified_model(path, len(body), "0"*64)
            self.assertEqual(path.read_bytes(), body)

    def test_checksum_failure_can_redownload_and_invalid_ledger_recovers(self):
        good, bad = b"good data", b"bad bytes"
        entry = ("test", "model.safetensors", "owner/repo", "revision", "file", len(good), hashlib.sha256(good).hexdigest())
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with patch("requests.get", return_value=self.response(bad)):
                with self.assertRaisesRegex(ValueError, "checksum"):
                    download_file(entry, root)
            self.assertFalse((root/"model.safetensors.download.json").exists())
            (root/"model.safetensors.download.json").write_text("incomplete JSON {")
            with patch("requests.get", return_value=self.response(good)) as get:
                target = download_file(entry, root)
                self.assertEqual(get.call_count, 1)
            self.assertEqual(target.read_bytes(), good)

    def test_existing_wrong_preset_same_size_is_rejected(self):
        body = b"official preset"
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"dit.safetensors"
            p.write_bytes(body)
            verified_model(p, len(body), hashlib.sha256(body).hexdigest())
            p.write_bytes(b"wrong!!! preset")
            with self.assertRaisesRegex(ValueError, "checksum"):
                verified_model(p, len(body), hashlib.sha256(body).hexdigest())

    def response(self, body):
        class Response:
            status_code = 206
            headers = {"Content-Range": f"bytes 0-{len(body)-1}/{len(body)}"}
            def __enter__(self):
                return self
            def __exit__(self, *a):
                pass
            def raise_for_status(self):
                pass
            def iter_content(self, n):
                yield body
        return Response()

    def test_cancel_resume_and_checksum_commit(self):
        class Interrupt(BaseException):
            pass
        body = b"verified official test bytes"
        entry = ("test", "mode/model.safetensors", "owner/repo", "pinned-revision", "file", len(body), hashlib.sha256(body).hexdigest())
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with patch("requests.get", return_value=self.response(body)) as get:
                with self.assertRaises(Interrupt):
                    download_file(entry, root, progress=lambda m: (_ for _ in ()).throw(Interrupt()))
                self.assertIn("/resolve/pinned-revision/", get.call_args.args[0])
            self.assertFalse((root/entry[1]).exists())
            with patch("requests.get", side_effect=AssertionError("Completed ranges should be resumed without downloading again")):
                result = download_file(entry, root)
            self.assertEqual(result.read_bytes(), body)

    def test_corrupt_download_not_published(self):
        body = b"bad bytes"
        entry = ("test", "model.safetensors", "owner/repo", "revision", "file", len(body), "0"*64)
        with tempfile.TemporaryDirectory() as d, patch("requests.get", return_value=self.response(body)):
            with self.assertRaisesRegex(ValueError, "checksum"):
                download_file(entry, Path(d))
            self.assertFalse((Path(d)/entry[1]).exists())

    def test_no_generic_module_namespace_pollution(self):
        code = """import sys,types
original={name:types.ModuleType(name) for name in ('inference','models','model_io')}
sys.modules.update(original)
import flashvsr_plus
from flashvsr_plus.inference import InferenceRequest
assert all(sys.modules[k] is v for k,v in original.items())
assert InferenceRequest.__module__.startswith('flashvsr_plus.')
"""
        subprocess.run([sys.executable, "-B", "-c", code], check=True, capture_output=True)


if __name__ == "__main__":
    unittest.main()
