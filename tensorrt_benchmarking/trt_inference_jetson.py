"""
TensorRT benchmark on NVIDIA Jetson Orin Nano (8 GB).

Same loop and printed metrics as main_execution.py: FPS, device line, model size on disk,
person mAP@0.5 over images/ and labels/, annotated JPGs to disk.

Important on Jetson
-------------------
- Build the .engine on this Jetson for its exact GPU. Engines built on an x86 desktop GPU
  are not valid on Orin and will fail to load or are unsupported.
- Use TensorRT and PyCUDA from JetPack (L4T). Follow NVIDIA Jetson setup docs instead of
  the desktop conda TensorRT steps in the main README.

From project root on the Jetson (Linux):

  python3 tensorrt_benchmarking/trt_inference_jetson.py \\
    --engine models/tensorrt_quantized_models/FP16_trt_models/640/yolov8n_640_trt_fp16.engine \\
    --input-size 640 \\
    --data-root imagedata

# default output folder -> outputfolder_trt_jetson/
# Optional: --output-dir path/to/custom_output
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from trt_inference import run_trt_benchmark
from trt_runtime import TrtSession

OUTPUT_DIR = "outputfolder_trt_jetson"
IOU = 0.7
CONF = 0.3
JETSON_DEVICE_LINE = "device=gpu (TensorRT, Jetson Orin Nano 8GB)"


def _parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="TensorRT .engine benchmark on Jetson (YOLO-format images/ + labels/).",
    )
    parser.add_argument(
        "--engine",
        type=Path,
        required=True,
        help="Path to .engine built on this Jetson from the matching YOLOv8 ONNX export.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        choices=(320, 640, 1280),
        required=True,
        help="Square letterbox imgsz matching ONNX and engine build.",
    )
    parser.add_argument("--data-root", type=str, required=True, help="Folder with images/ and labels/")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"Where to save annotated JPGs (default {OUTPUT_DIR}).",
    )

    args = parser.parse_args()

    eng = Path(args.engine)
    if eng.is_file():
        args.engine = str(eng.resolve())
    else:
        cand = repo / eng
        if cand.is_file():
            args.engine = str(cand.resolve())
        else:
            parser.error(f"Engine not found: {eng}")

    dr = Path(args.data_root)
    if dr.is_dir():
        args.data_root = str(dr.resolve())
    else:
        alt = repo / args.data_root
        if alt.is_dir():
            args.data_root = str(alt.resolve())
        else:
            parser.error(f"Data root not found: {dr}")

    for sub in ("images", "labels"):
        pth = Path(args.data_root) / sub
        if not pth.is_dir():
            parser.error(f"Missing '{sub}' under data root ({pth})")

    return args


def main() -> None:
    args = _parse_args()
    input_size = (args.input_size, args.input_size)
    images_dir = str(Path(args.data_root) / "images")
    labels_dir = str(Path(args.data_root) / "labels")
    out_dir = args.output_dir

    if out_dir == OUTPUT_DIR and os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    session = TrtSession(args.engine)
    run_trt_benchmark(
        session,
        images_dir,
        labels_dir,
        out_dir,
        input_size,
        CONF,
        IOU,
        engine_path=args.engine,
        device_line=JETSON_DEVICE_LINE,
    )


if __name__ == "__main__":
    main()
