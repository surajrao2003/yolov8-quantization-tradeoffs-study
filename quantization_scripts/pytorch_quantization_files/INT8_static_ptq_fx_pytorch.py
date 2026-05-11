r"""
Static Post-Training Quantization (PTQ) for YOLOv8 using PyTorch FX graph mode (experimental).

What it does
------------
- Static PTQ (weights + activations) on CPU: ``prepare_fx`` → calibration forwards → ``convert_fx``.
- More relevant than dynamic quant for CNNs, but YOLOv8 may need graph tweaks for FX to succeed.

How to run (from project root)
------------------------------
Use the repo **relative** calibration folder (same as ONNX static INT8):

    python quantization_scripts/pytorch_quantization_files/INT8_static_ptq_fx_pytorch.py \
      --weights models/yolo_models/yolov8n.pt \
      --calib-dir quantization_scripts/calibration_dataset \
      --img-size 640 \
      --num-calib 50

Alternative calibration folder (e.g. ``imagedata/images``), optional ``--output-pt``:

    python quantization_scripts/pytorch_quantization_files/INT8_static_ptq_fx_pytorch.py \
      --weights models/yolo_models/yolov8n.pt \
      --calib-dir imagedata/images \
      --img-size 640 \
      --output-pt models/pytorch_quantized_models/INT8_pytorch_static_quantized_models/640/yolov8n_640_pytorch_static_quantized.pt

Default output (PyTorch-side static INT8; ONNX static INT8 is under ``models/onnx_quantized_models/INT8_onnx_static_quantized_models/``):

    models/pytorch_quantized_models/INT8_pytorch_static_quantized_models/<img-size>/<stem>_<img-size>_pytorch_static_quantized.pt

Calibration images: ``*.jpg``, ``*.jpeg``, ``*.png``, ``*.bmp`` under ``--calib-dir``.
Preprocessing here is resize-to-square (float 0–1); not identical to letterbox infer.

Notes
-----
- Default ``--backend fbgemm`` (Linux/server). **Many Windows CPU wheels omit FBGEMM and QNNPACK** but include
  **OneDNN** (``onednn``) — this script tries **fbgemm → onednn → qnnpack → fbgemm** (deduped) until one works.
  Pin with ``--backend onednn`` if you want to skip warnings.
- If no backend loads, use ONNX static INT8 instead: ``quantization_scripts/onnx_quantization_files/INT8_static_quantization_onnx.py``.
- If ``prepare_fx`` fails on YOLOv8, try eager PTQ/QAT or a traceable subgraph; FX often struggles on full detectors.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parent.parent.parent


def _resolve_weights(path_arg: Path, repo: Path) -> Path:
    p = Path(path_arg)
    return p if p.is_file() else (repo / p).resolve()


def _resolve_calib_dir(path_arg: Path, repo: Path) -> Path:
    p = Path(path_arg)
    return p if p.is_dir() else (repo / p).resolve()


def _default_output_pt(weights: Path, img_size: int) -> Path:
    stem = weights.stem
    return (
        _REPO
        / "models"
        / "pytorch_quantized_models"
        / "INT8_pytorch_static_quantized_models"
        / str(img_size)
        / f"{stem}_{img_size}_pytorch_static_quantized.pt"
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


def _iter_images(calib_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        paths.extend(calib_dir.glob(ext))
    return sorted(paths)


def _preprocess_image_to_tensor(image_path: Path, img_size: int) -> torch.Tensor:
    try:
        from PIL import Image
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Pillow required for calibration (pip install pillow).") from e

    img = Image.open(image_path).convert("RGB")
    img = img.resize((int(img_size), int(img_size)))
    x = torch.from_numpy(np.array(img)).permute(2, 0, 1).contiguous().float().div_(255.0).unsqueeze(0)
    return x


def _get_fx_quant_fns():
    try:
        from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("This PyTorch build lacks FX quantization APIs.") from e
    return prepare_fx, convert_fx


def _ensure_quant_backend(requested: str) -> str:
    """
    Pick a working ``torch.backends.quantized.engine``.

    Typical PyTorch CPU builds expose **onednn** (especially on Windows) even when **fbgemm** / **qnnpack**
    are disabled. Order: user choice first, then onednn, qnnpack, fbgemm, x86.
    """
    ordered: list[str] = []
    for b in (requested, "onednn", "qnnpack", "fbgemm", "x86"):
        if b in ordered:
            continue
        ordered.append(b)

    failures: list[str] = []
    for b in ordered:
        try:
            torch.backends.quantized.engine = b
            if b != requested:
                print(
                    f"Note: quantized backend {requested!r} is not supported on this PyTorch build; "
                    f"using {b!r} instead."
                )
            return b
        except RuntimeError as e:
            failures.append(f"{b}: {e}")

    onnx_hint = "quantization_scripts/onnx_quantization_files/INT8_static_quantization_onnx.py"
    raise RuntimeError(
        "No usable CPU quantized backend on this PyTorch install (tried: "
        + ", ".join(ordered)
        + "). Try another PyTorch build (e.g. official CPU wheel from pytorch.org), "
        "or run static INT8 via ONNX Runtime instead:\n"
        f"  python {onnx_hint} --model-fp32 <FP32.onnx> --img-size ... --calib-dir ...\n"
        "Underlying errors:\n  "
        + "\n  ".join(failures)
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Static PTQ for YOLOv8 via torch.ao.quantization FX (experimental)."
    )
    p.add_argument(
        "--weights",
        type=Path,
        required=True,
        help=".pt weights (e.g. models/yolo_models/yolov8n.pt), relative to CWD or repo root.",
    )
    p.add_argument(
        "--calib-dir",
        type=Path,
        required=True,
        help=(
            "Folder of calibration images (relative to CWD or repo root). "
            "Default bundle for this repo: quantization_scripts/calibration_dataset"
        ),
    )
    p.add_argument(
        "--img-size",
        type=int,
        choices=(320, 640, 1280),
        required=True,
        help="Square input size for calibration tensors and smoke forward.",
    )
    p.add_argument(
        "--num-calib",
        type=int,
        default=50,
        help="Max number of calibration images to run (default: 50).",
    )
    p.add_argument(
        "--backend",
        type=str,
        default="fbgemm",
        choices=("fbgemm", "onednn", "qnnpack"),
        help=(
            "CPU quantization backend for FX PTQ (default: fbgemm). "
            "Auto-falls back through onednn / qnnpack / x86 internally if unsupported (common on Windows)."
        ),
    )
    p.add_argument(
        "--output-pt",
        type=Path,
        default=None,
        help="Output .pt path. Default: models/pytorch_quantized_models/INT8_pytorch_static_quantized_models/"
        "<img-size>/<stem>_<img-size>_pytorch_static_quantized.pt",
    )
    args = p.parse_args(argv)

    weights = _resolve_weights(args.weights, _REPO)
    calib_dir = _resolve_calib_dir(args.calib_dir, _REPO)
    if not weights.is_file():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not calib_dir.is_dir():
        raise FileNotFoundError(f"Calibration dir not found: {calib_dir}")

    img_paths = _iter_images(calib_dir)
    if not img_paths:
        raise FileNotFoundError(f"No calibration images found under {calib_dir}")

    backend = _ensure_quant_backend(args.backend)

    model = _load_ultralytics_module(weights).eval().cpu()

    prepare_fx, convert_fx = _get_fx_quant_fns()

    try:
        qconfig_mapping = torch.ao.quantization.get_default_qconfig_mapping(backend)
        example_inputs = (torch.zeros(1, 3, int(args.img_size), int(args.img_size), dtype=torch.float32),)
        prepared = prepare_fx(model, qconfig_mapping, example_inputs)
    except Exception as e:
        print("FX prepare_fx failed (common on full YOLOv8 graphs).")
        print("Error:", repr(e))
        print("Tip: eager-mode PTQ, QAT, or a smaller traceable submodule may be needed.")
        return 2

    n = min(int(args.num_calib), len(img_paths))
    with torch.no_grad():
        for i, ip in enumerate(img_paths[:n], start=1):
            x = _preprocess_image_to_tensor(ip, args.img_size)
            _ = prepared(x)
            if i % 10 == 0 or i == n:
                print(f"Calibrated {i}/{n}")

    try:
        quantized = convert_fx(prepared)
    except Exception as e:
        print("FX convert_fx failed after calibration.")
        print("Error:", repr(e))
        return 3

    with torch.no_grad():
        _ = quantized(torch.zeros(1, 3, int(args.img_size), int(args.img_size), dtype=torch.float32))
    print("Quantized model smoke forward pass OK.")

    out = Path(args.output_pt) if args.output_pt else _default_output_pt(weights, args.img_size)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(quantized, str(out))
    print(f"Saved quantized module: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
