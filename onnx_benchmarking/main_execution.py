"""
ONNX benchmark: FPS, file size, person mAP@0.5 (data-root needs images/ and labels/).

Run:

  python onnx_benchmarking/main_execution.py --model models/onnx_quantized_models/FP32_onnx_models/640/yolov8s_640.onnx --device gpu --input-size 640 --data-root imagedata

# default output folder -> outputfolder/
# Optional: --output-dir path/to/outputfolder
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from onnx_inference import initialize_model, run_onnx_benchmark

OUTPUT_DIR = "outputfolder"
IOU = 0.7
CONF = 0.3


def _parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description="ONNX Runtime benchmark (YOLO-format data).")
    p.add_argument("--model", type=Path, required=True, help="Path to .onnx file.")
    p.add_argument("--device", type=str, choices=("cpu", "gpu"), required=True)
    p.add_argument("--input-size", type=int, choices=(320, 640, 1280), required=True)
    p.add_argument("--data-root", type=str, required=True, help="Folder with images/ and labels/")
    p.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    args = p.parse_args()

    m = Path(args.model)
    if m.is_file():
        args.model = str(m.resolve())
    else:
        alt = repo / m
        if alt.is_file():
            args.model = str(alt.resolve())
        else:
            p.error(f"Model not found: {m}")

    dr = Path(args.data_root)
    if dr.is_dir():
        args.data_root = str(dr.resolve())
    else:
        alt = repo / args.data_root
        if alt.is_dir():
            args.data_root = str(alt.resolve())
        else:
            p.error(f"Data root not found: {dr}")

    for sub in ("images", "labels"):
        if not (Path(args.data_root) / sub).is_dir():
            p.error(f"Missing '{sub}' under data root")

    return args


def main() -> None:
    args = _parse_args()
    input_size = (args.input_size, args.input_size)
    images_dir = str(Path(args.data_root) / "images")
    labels_dir = str(Path(args.data_root) / "labels")

    # Default outputfolder is treated as a scratch directory and is recreated per run.
    # If a custom --output-dir is provided, do not delete it automatically.
    if args.output_dir == OUTPUT_DIR and os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    ort_model = initialize_model(args.model, args.device)
    run_onnx_benchmark(
        ort_model,
        images_dir,
        labels_dir,
        args.output_dir,
        input_size,
        CONF,
        IOU,
        model_path=args.model,
    )


if __name__ == "__main__":
    main()

