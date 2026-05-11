r"""
Dynamic quantization (PyTorch) for YOLOv8 (experimental).

What it does
------------
- Applies post-training *dynamic quantization* (weights-only) to supported layers, mainly nn.Linear.
- Usually limited benefit for YOLOv8 because the backbone/head are largely convolution-based.

How to run (from project root)
------------------------------
UNIX-style:

    python quantization_scripts/pytorch_quantization_files/INT8_dynamic_quantization_pytorch.py \
      --weights models/yolo_models/yolov8n.pt \
      --img-size 640

Optional explicit output (.pt):

    python quantization_scripts/pytorch_quantization_files/INT8_dynamic_quantization_pytorch.py \
      --weights models/yolo_models/yolov8n.pt \
      --img-size 640 \
      --output-pt models/pytorch_quantized_models/INT8_pytorch_dynamic_quantized_models/640/yolov8n_640_pytorch_dynamic_quantized.pt

Default output:

    models/pytorch_quantized_models/INT8_pytorch_dynamic_quantized_models/<img-size>/<stem>_<img-size>_pytorch_dynamic_quantized.pt

Notes
-----
- Targets ``nn.Linear`` only (typical / safe for ``quantize_dynamic``).
- Saved object is a pickled ``torch.nn.Module``; keep PyTorch versions consistent across machines.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Workaround for common Windows OpenMP duplicate runtime issue.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn

# Project root (parent of quantization_scripts/)
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
        / "INT8_pytorch_dynamic_quantized_models"
        / str(img_size)
        / f"{stem}_{img_size}_pytorch_dynamic_quantized.pt"
    )


def _load_ultralytics_module(weights: Path) -> nn.Module:
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


def dynamic_quantize_linear_only(model: nn.Module, *, dtype: torch.dtype) -> nn.Module:
    model = model.eval()
    return torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=dtype)


@torch.no_grad()
def _smoke_forward(model: nn.Module, img_size: int) -> None:
    model.eval()
    x = torch.zeros(1, 3, int(img_size), int(img_size), dtype=torch.float32)
    _ = model(x)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="PyTorch dynamic quantization (nn.Linear) for YOLOv8 weights (experimental)."
    )
    p.add_argument(
        "--weights",
        type=Path,
        required=True,
        help=".pt weights path (e.g. models/yolo_models/yolov8n.pt), relative to CWD or repo root.",
    )
    p.add_argument(
        "--img-size",
        type=int,
        choices=(320, 640, 1280),
        required=True,
        help="Square input size used for smoke forward; also used in default output path (like ONNX scripts).",
    )
    p.add_argument(
        "--dtype",
        type=str,
        default="qint8",
        choices=("qint8", "float16"),
        help="Dynamic quant dtype for supported layers (default: qint8).",
    )
    p.add_argument(
        "--output-pt",
        type=Path,
        default=None,
        help="Output .pt path. Default: models/pytorch_quantized_models/INT8_pytorch_dynamic_quantized_models/"
        "<img-size>/<stem>_<img-size>_pytorch_dynamic_quantized.pt",
    )
    args = p.parse_args(argv)

    weights = _resolve_weights(args.weights, _REPO)
    if not weights.is_file():
        raise FileNotFoundError(f"Weights not found: {weights}")

    dtype = torch.qint8 if args.dtype == "qint8" else torch.float16

    model = _load_ultralytics_module(weights)
    qmodel = dynamic_quantize_linear_only(model, dtype=dtype)

    _smoke_forward(qmodel, args.img_size)
    print("Smoke forward pass OK.")

    out = Path(args.output_pt) if args.output_pt else _default_output_pt(weights, args.img_size)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(qmodel, str(out))
    print(f"Saved quantized module: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
