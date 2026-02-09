# !/usr/bin/env python3
# python ERP/convert_outputs_to_cubemap.py --input_dir="ERP/outputs" --output_dir="ERP/outputs_cubemap" --size 512
import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert ERP JPGs to cubemap images using convert360.")
    parser.add_argument(
        "--input_dir",
        default="ERP/outputs",
        help="Directory with ERP JPG files (default: ERP/outputs)",
    )
    parser.add_argument(
        "--output_dir",
        default="ERP/outputs_cubemap",
        help="Directory to write cubemap PNGs (default: ERP/outputs_cubemap)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Cubemap face size (default: 512)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    inputs = sorted(input_dir.glob("*.jpg"))
    if not inputs:
        print(f"No .jpg files found in {input_dir}")
        return

    for src_path in inputs:
        out_name = f"{src_path.stem}_cubemap.png"
        out_path = output_dir / out_name
        cmd = [
            "convert360",
            "e2c",
            str(src_path),
            str(out_path),
            "--size",
            str(args.size),
        ]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
