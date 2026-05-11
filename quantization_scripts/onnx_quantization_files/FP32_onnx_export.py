"""
Export a YOLOv8 PyTorch model (.pt) to FP32 ONNX with a fixed input size.

How to run
----------
1. Open your terminal in the project root (the folder that contains "code_files" and "models").
2. Either pass CLI arguments (recommended), or edit IMGSZ / WEIGHTS_PT / OUT_NAME below and run with no args.

CLI example:

       python quantization_scripts/onnx_quantization_files/FP32_onnx_export.py --weights models/yolo_models/yolov8n.pt --imgsz 320 --output-name yolov8n_320.onnx

No-args legacy mode uses constants at the bottom of this file.

Output:

       models/onnx_quantized_models/FP32_onnx_models/<IMGSZ>/

   For example IMGSZ=640 uses folder .../FP32_onnx_models/640/

ai.onnx opset defaults to 17
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

# Project root (parent of quantization_scripts/)
_REPO = Path(__file__).resolve().parent.parent.parent

IMGSZ = 320  # 320 | 640 | 1280
WEIGHTS_PT = _REPO / "models" / "yolo_models" / "yolov8s.pt"
# If None, ONNX is moved next to Ultralytics export; if set, file is renamed (e.g. yolov8l_640.onnx)
OUT_NAME: str | None = None

# ai.onnx >= 13: QDQ static quantization emits DequantizeLinear with `axis`; opset 12 graphs are invalid.
ONNX_OPSET = 17


def export_fp32_ultralytics_onnx(
    *,
    weights: Path,
    imgsz: int,
    out_name: str | None = None,
    opset: int = ONNX_OPSET,
    repo: Path = _REPO,
) -> Path:
    weights = Path(weights)
    if not weights.is_file():
        raise FileNotFoundError(f"Weights not found: {weights}")

    out_dir = repo / "models" / "onnx_quantized_models" / "FP32_onnx_models" / str(imgsz)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    exported = Path(
        model.export(
            format="onnx",
            imgsz=imgsz,
            opset=opset,
            simplify=True,
            dynamic=False,
        )
    )

    target = out_dir / (out_name if out_name else exported.name)
    if exported.resolve() != target.resolve():
        shutil.move(str(exported), str(target))

    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export YOLOv8 .pt to FP32 ONNX (Ultralytics).",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help=(
            "YOLOv8 .pt under models/yolo_models/ … (relative to CWD or project root when not found locally). "
            f"Default (no CLI args): {WEIGHTS_PT}"
        ),
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        choices=(320, 640, 1280),
        default=None,
        help=f"Square export size. Default (no CLI): {IMGSZ}",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Filename under FP32_onnx_models/<imgsz>/ (e.g. yolov8n_320.onnx). Default: Ultralytics export name.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=None,
        help=f"ONNX opset (>=13 for static QDQ). Default: {ONNX_OPSET}",
    )
    args = parser.parse_args(argv)

    weights = (args.weights or WEIGHTS_PT).resolve()
    imgsz = args.imgsz if args.imgsz is not None else IMGSZ
    out_name = args.output_name if args.output_name is not None else OUT_NAME
    opset = args.opset if args.opset is not None else ONNX_OPSET

    target = export_fp32_ultralytics_onnx(
        weights=weights,
        imgsz=imgsz,
        out_name=out_name,
        opset=opset,
        repo=_REPO,
    )
    print(f"Export complete: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
