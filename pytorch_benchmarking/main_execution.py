"""
PyTorch Ultralytics .pt benchmark: FPS, file size, person mAP@0.5. data-root needs images/ and labels/.

Standard Ultralytics checkpoint:

  python pytorch_benchmarking/main_execution.py --model models/yolo_models/yolov8n.pt --device cpu --input-size 640 --data-root imagedata

nn.Module-only export (--base-weights must match variant, e.g. yolov8n):

  python pytorch_benchmarking/main_execution.py --model models/pytorch_quantized_models/FP16_pytorch_models/640/yolov8n_640_fp16.pt --base-weights models/yolo_models/yolov8n.pt --device gpu --input-size 640 --data-root imagedata

# default output folder -> outputfolder/
# Optional: --output-dir path/to/outputfolder
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from pytorch_inference import (
    _repo_root,
    load_yolo_model,
    reported_device_pytorch,
    run_pytorch_benchmark,
)

OUTPUT_DIR = "outputfolder"
IOU = 0.7
CONF = 0.3


def _parse_args() -> argparse.Namespace:
    repo = _repo_root()
    parser = argparse.ArgumentParser(
        description="YOLOv8 PyTorch (.pt) inference benchmark (FPS, size, mAP@0.5).",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to .pt (Ultralytics checkpoint or saved nn.Module with --base-weights).",
    )
    parser.add_argument(
        "--base-weights",
        type=str,
        default=None,
        help=(
            "Matching Ultralytics .pt when --model is a raw Module export "
            "(FP16 / dynamic INT8 under pytorch_quantized_models/). "
            "Example: models/yolo_models/yolov8n.pt"
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=("cpu", "gpu"),
        required=True,
        help="cpu or gpu (CUDA).",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        choices=(320, 640, 1280),
        required=True,
        help="Square imgsz; must match how the checkpoint was exported/trained.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Dataset root with 'images' and 'labels' subfolders.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"Annotated images output folder (default: {OUTPUT_DIR}).",
    )

    args = parser.parse_args()

    model_file = Path(args.model)
    if model_file.is_file():
        args.model = str(model_file.resolve())
    else:
        alt = repo / model_file
        if alt.is_file():
            args.model = str(alt.resolve())
        else:
            parser.error(f"Model file not found: {model_file}")

    root = Path(args.data_root)
    if not root.is_dir():
        alt = repo / args.data_root
        if not alt.is_dir():
            parser.error(f"Data root not found: {root.resolve()}")
        args.data_root = str(alt)
    else:
        args.data_root = str(root.resolve())

    images_dir = Path(args.data_root) / "images"
    labels_dir = Path(args.data_root) / "labels"
    if not images_dir.is_dir():
        parser.error(f"Missing images/: {images_dir}")
    if not labels_dir.is_dir():
        parser.error(f"Missing labels/: {labels_dir}")

    if args.base_weights:
        bw = Path(args.base_weights)
        if bw.is_file():
            args.base_weights = str(bw.resolve())
        elif (repo / bw).is_file():
            args.base_weights = str((repo / bw).resolve())
        else:
            parser.error(f"--base-weights not found: {args.base_weights}")

    return args


def main() -> None:
    args = _parse_args()
    repo = _repo_root()
    model_path = args.model
    base = Path(args.base_weights) if args.base_weights else None

    input_size = (args.input_size, args.input_size)
    images_dir = str(Path(args.data_root) / "images")
    labels_dir = str(Path(args.data_root) / "labels")
    out_dir = args.output_dir

    # Default outputfolder is treated as a scratch directory and is recreated per run.
    # If a custom --output-dir is provided, do not delete it automatically.
    if out_dir == OUTPUT_DIR and os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    yolo = load_yolo_model(
        Path(model_path),
        device=args.device.lower(),
        base_weights=base,
        repo=repo,
    )
    tag = reported_device_pytorch(args.device.lower())

    run_pytorch_benchmark(
        yolo,
        images_dir,
        labels_dir,
        out_dir,
        input_size,
        CONF,
        IOU,
        model_path=model_path,
        device_tag=tag,
    )


if __name__ == "__main__":
    main()
