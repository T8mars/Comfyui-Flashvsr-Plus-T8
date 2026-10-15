"""Create a derived INT8 ConvRot DiT without changing the official files."""
import argparse
import json
from pathlib import Path

from .quantization import convert_checkpoint
from .weights import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="+", choices=["turbo", "base", "v1"], default=["turbo", "base"])
    parser.add_argument("--model-dir", type=Path, default=ROOT/"ckpts")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    directory = args.model_dir.expanduser()
    for mode in args.modes:
        source = directory/mode/"dit.safetensors"
        target = directory/mode/"dit_int8_convrot.safetensors"
        report = convert_checkpoint(source, target, mode=mode, device=args.device, progress=print)
        target.with_suffix(".conversion.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Created {target}: {report['output_bytes']:,} bytes, {len(report['layers'])} quantized Linear layers")


if __name__ == "__main__":
    main()
