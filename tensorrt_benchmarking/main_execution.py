"""
TensorRT GPU benchmark: FPS, file size, person mAP@0.5 (same prints as code_files/inference workflow).

Needs: data-root with images/ and labels/. NVIDIA GPUs only.

Run:

  python tensorrt_benchmarking/main_execution.py --engine models/tensorrt_quantized_models/FP16_trt_models/640/yolov8n_640_trt_fp16.engine --input-size 640 --data-root imagedata

# default output folder -> outputfolder/
# Optional: --output-dir path/to/outputfolder

GPU match : The .engine was built for one GPU type and software stack. Running it on a
different NVIDIA model (for example engine built on RTX 4050 but loaded on RTX 4000 Ada, or the other
way around) is not supported. TensorRT ties the serialized plan to that device context; you may get
wrong results, crashes, deadlocks, or only slower runs. For GPU1 versus GPU2 in this repo, use an engine
built on that same machine. Rebuild on each target GPU instead of copying one .engine between them.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from trt_inference import run_trt_benchmark
from trt_runtime import TrtSession

REPO = Path(__file__).resolve().parent.parent
OUTPUT_DIR = "outputfolder"
IOU = 0.7
CONF = 0.3


def _parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="TensorRT .engine inference benchmark (YOLO-format data).")
    parser.add_argument(
        "--engine",
        type=Path,
        required=True,
        help="TensorRT .engine built on this same GPU from YOLOv8 ONNX (do not reuse across different GPU models).",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        choices=(320, 640, 1280),
        required=True,
        help="Square letterbox imgsz matching ONNX/export.",
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

    # Default outputfolder is treated as a scratch directory and is recreated per run.
    # If a custom --output-dir is provided, do not delete it automatically.
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
    )


if __name__ == "__main__":
    main()
