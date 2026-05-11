r"""
Half-precision (FP16) PyTorch checkpoint from YOLOv8 ``.pt`` weights (experimental).

Loads Ultralytics ``YOLO`` weights, converts the underlying ``torch.nn.Module`` to ``float16``,
runs a small CPU forward sanity check, and saves ``torch.save(model, path)``.

Use **ONNX** FP16 for deployment with ONNX Runtime: ``onnx_quantization_files/FP16_onnx_export.py``.
This script is for **keeping a native PyTorch** FP16 Module on disk alongside ONNX experiments.

Default output layout (native PyTorch FP16 checkpoints):

    models/pytorch_quantized_models/FP16_pytorch_models/<img-size>/<stem>_<img-size>_fp16.pt

Run from project root:

    python quantization_scripts/pytorch_quantization_files/FP16_pytorch.py \
      --weights models/yolo_models/yolov8n.pt \
      --img-size 640
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parent.parent.parent


def _resolve_weights(path_arg: Path, repo: Path) -> Path:
    p = Path(path_arg)
    return p if p.is_file() else (repo / p).resolve()


def _default_output_pt(weights: Path, img_size: int) -> Path:
    stem = weights.stem
    return (
        _REPO
        / "models"
        / "pytorch_quantized_models"
        / "FP16_pytorch_models"
        / str(img_size)
        / f"{stem}_{img_size}_fp16.pt"
    )


def _load_model(weights: Path) -> nn.Module:
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Ultralytics is required (see requirements.txt / pip install ultralytics)."
        ) from e

    y = YOLO(str(weights))
    m = y.model
    if not isinstance(m, nn.Module):
        raise TypeError(f"Unexpected Ultralytics model type: {type(m)}")
    return m


@torch.no_grad()
def _smoke_forward_fp16(model: nn.Module, img_size: int) -> None:
    model.eval()
    x = torch.zeros(1, 3, int(img_size), int(img_size), dtype=torch.float16)
    _ = model(x)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Save YOLOv8 weights as an FP16 torch.nn.Module checkpoint (experimental).",
    )
    p.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Source .pt (e.g. models/yolo_models/yolov8n.pt), relative to CWD or repo root.",
    )
    p.add_argument(
        "--img-size",
        type=int,
        choices=(320, 640, 1280),
        required=True,
        help="Square input size for smoke forward; used in default output path.",
    )
    p.add_argument(
        "--output-pt",
        type=Path,
        default=None,
        help="Output .pt path. Default: models/pytorch_quantized_models/FP16_pytorch_models/"
        "<img-size>/<stem>_<img-size>_fp16.pt",
    )
    args = p.parse_args(argv)

    weights = _resolve_weights(args.weights, _REPO)
    if not weights.is_file():
        raise FileNotFoundError(f"Weights not found: {weights}")

    model = _load_model(weights)
    model = model.cpu().half()

    _smoke_forward_fp16(model, args.img_size)
    print("FP16 smoke forward pass OK.")

    out = Path(args.output_pt) if args.output_pt else _default_output_pt(weights, args.img_size)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, str(out))
    print(f"Saved FP16 module: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
