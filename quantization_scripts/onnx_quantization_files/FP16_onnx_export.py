"""
Export FP16 ONNX from YOLOv8 weights (.pt) using Ultralytics (fixed square size).

Default output (project root):

    models/onnx_quantized_models/FP16_onnx_models/<img-size>/<stem>_<img-size>_fp16.onnx

Run from project root:

    python quantization_scripts/onnx_quantization_files/FP16_onnx_export.py --weights models/yolo_models/yolov8n.pt --img-size 320

Optional explicit output:

    python quantization_scripts/onnx_quantization_files/FP16_onnx_export.py --weights models/yolo_models/yolov8n.pt --img-size 320 --output-onnx models/onnx_quantized_models/FP16_onnx_models/320/yolov8n_320_fp16.onnx

Default ``simplify=False``; pass ``--simplify`` for onnxslim. When CUDA is available, export uses GPU index ``0`` unless ``--device`` is set.

Runs ONNX checker unless ``--skip-onnx-check`` (use ``--strict-onnx-check`` to fail immediately on checker errors). Runs ONNX Runtime CPU session creation unless ``--skip-ort-check``.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import onnx
import onnxruntime as ort
import torch
from ultralytics import YOLO

# Project root (parent of quantization_scripts/); script lives in onnx_quantization_files/
_REPO = Path(__file__).resolve().parent.parent.parent


def _default_output(weights: Path, img_size: int) -> Path:
    stem = weights.stem
    return (
        _REPO
        / "models"
        / "onnx_quantized_models"
        / "FP16_onnx_models"
        / str(img_size)
        / f"{stem}_{img_size}_fp16.onnx"
    )


def export_fp16_ultralytics_onnx(
    *,
    weights_pt: Path,
    img_size: int,
    output_onnx: Path | None,
    opset: int = 17,
    simplify: bool = False,
    device: str | int | None = None,
    skip_onnx_check: bool = False,
    strict_onnx_check: bool = False,
    skip_ort_check: bool = False,
) -> Path:
    weights_pt = Path(weights_pt)
    if not weights_pt.is_file():
        raise FileNotFoundError(f"Weights not found: {weights_pt}")

    dest = Path(output_onnx) if output_onnx else _default_output(weights_pt, img_size)
    dest.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights_pt))
    export_device = device
    if export_device is None and torch.cuda.is_available():
        export_device = 0

    export_kw: dict = dict(
        format="onnx",
        imgsz=img_size,
        opset=opset,
        simplify=simplify,
        dynamic=False,
        half=True,
    )
    if export_device is not None:
        export_kw["device"] = export_device

    exported = Path(model.export(**export_kw))

    if exported.resolve() != dest.resolve():
        shutil.move(str(exported), str(dest))

    if not skip_onnx_check:
        try:
            onnx.checker.check_model(str(dest))
            print(f"onnx.checker passed: {dest}")
        except onnx.checker.ValidationError:
            if strict_onnx_check:
                raise
            print(f"Warning: onnx.checker rejected {dest}; continuing with ONNX Runtime smoke test.")

    if not skip_ort_check:
        ort.InferenceSession(str(dest), providers=["CPUExecutionProvider"])
        print("ONNX Runtime CPU InferenceSession(create) passed.")

    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export FP16 ONNX from YOLOv8 .pt (Ultralytics, fixed shape).",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="YOLOv8 .pt checkpoint; repo-relative examples: models/yolo_models/yolov8n.pt "
        "(resolved vs CWD, then vs project root if missing).",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        choices=(320, 640, 1280),
        required=True,
        help="Fixed square export size.",
    )
    parser.add_argument(
        "--output-onnx",
        type=Path,
        default=None,
        help="Destination .onnx file. Default: models/onnx_quantized_models/FP16_onnx_models/<img-size>/<stem>_<img-size>_fp16.onnx",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset for export (>=13 required for valid INT8 QDQ with per-channel Dq axis).",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Run onnxslim after export (default: off).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Export device, e.g. 0 or cuda:0. Default: GPU 0 if CUDA available, else Ultralytics default (CPU).",
    )
    parser.add_argument(
        "--skip-onnx-check",
        action="store_true",
        help="Skip onnx.checker validation.",
    )
    parser.add_argument(
        "--strict-onnx-check",
        action="store_true",
        help="Treat onnx.checker ValidationError as a hard failure.",
    )
    parser.add_argument(
        "--skip-ort-check",
        action="store_true",
        help="Skip ONNX Runtime session creation smoke test.",
    )

    args = parser.parse_args()
    if args.skip_onnx_check and args.strict_onnx_check:
        parser.error("Use only one of --skip-onnx-check and --strict-onnx-check.")

    weights_arg = Path(args.weights)
    path = weights_arg if weights_arg.is_file() else _REPO / weights_arg

    out = export_fp16_ultralytics_onnx(
        weights_pt=path,
        img_size=args.img_size,
        output_onnx=args.output_onnx,
        opset=args.opset,
        simplify=args.simplify,
        device=args.device,
        skip_onnx_check=args.skip_onnx_check,
        strict_onnx_check=args.strict_onnx_check,
        skip_ort_check=args.skip_ort_check,
    )
    print(f"Done: {out}")
