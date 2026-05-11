"""
TFLite benchmark: FPS, file size, person mAP@0.5 (data-root needs images/ and labels/).

Run (FP32):

  python tflite_benchmarking/main_execution.py --model models/tflite_quantized_models/FP32_tflite_models/640/yolov8n_640_fp32.tflite --device cpu --input-size 640 --data-root imagedata

Run (FP16):

  python tflite_benchmarking/main_execution.py --model models/tflite_quantized_models/FP16_tflite_models/640/yolov8n_640_fp16.tflite --device cpu --input-size 640 --data-root imagedata

On Windows, `pip install tensorflow` usually has no TFLite GPU delegate; use `--device cpu` or
`--device gpu-fallback`. For NVIDIA GPU inference, use `tensorrt_benchmarking/`.

# default output folder -> outputfolder/
# Optional: --output-dir path/to/outputfolder
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from tflite_inference import run_tflite_benchmark
from tflite_runtime import TfliteSession

OUTPUT_DIR = "outputfolder"
IOU = 0.7
CONF = 0.3


def _open_tflite_session(model_path: str, device: str) -> TfliteSession:
    if device == "gpu-fallback":
        try:
            return TfliteSession(model_path, device="gpu")
        except RuntimeError as e:
            msg = str(e)
            if "TFLite GPU delegate could not be loaded" not in msg and "load_delegate" not in msg:
                raise
            print("Warning: TFLite GPU delegate unavailable; using CPU instead.")
            return TfliteSession(model_path, device="cpu")
    resolved = "gpu" if device == "gpu" else "cpu"
    return TfliteSession(model_path, device=resolved)


def _parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description="TFLite .tflite inference benchmark (YOLO-format data).")
    p.add_argument("--model", type=Path, required=True, help="Path to .tflite model file.")
    p.add_argument(
        "--device",
        type=str,
        choices=("cpu", "gpu", "gpu-fallback"),
        default="cpu",
        help="cpu (default), gpu (TFLite GPU delegate — often missing on Windows pip TF), "
        "or gpu-fallback (try gpu, then cpu).",
    )
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

    session = _open_tflite_session(args.model, str(args.device))
    run_tflite_benchmark(
        session,
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

