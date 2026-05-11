"""
YOLOv8 .pt → FP16 TFLite.

Run:

  python quantization_scripts/tflite_quantization_files/tflite_fp16_quantization.py --weights models/yolo_models/yolov8n.pt --imgsz 640

Writes models/tflite_quantized_models/FP16_tflite_models/<imgsz>/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from ultralytics import YOLO

from _tflite_common import REPO, move_to_dest, resolve_weights, sanitize_export_return, tflite_out_dir

IMGSZ = 640
WEIGHTS_PT = REPO / "models" / "yolo_models" / "yolov8n.pt"


def export_fp16_tflite(*, weights: Path, imgsz: int, repo: Path = REPO) -> Path:
    weights = resolve_weights(weights)
    out_dir = tflite_out_dir(repo, "FP16_tflite_models", imgsz)
    dest = out_dir / f"{weights.stem}_{imgsz}_fp16.tflite"

    model = YOLO(str(weights))
    exported = sanitize_export_return(
        model.export(
            format="tflite",
            imgsz=imgsz,
            half=True,
            int8=False,
            simplify=True,
            nms=True,
        )
    )
    return move_to_dest(exported, dest)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Export YOLOv8 .pt to FP16 TFLite (Ultralytics).")
    p.add_argument("--weights", type=Path, default=None, help="Path to .pt (default: yolov8n under models/yolo_models/).")
    p.add_argument("--imgsz", type=int, choices=(320, 640, 1280), default=None, help=f"Square export size. Default: {IMGSZ}")
    args = p.parse_args(argv)

    weights = args.weights if args.weights is not None else WEIGHTS_PT
    imgsz = args.imgsz if args.imgsz is not None else IMGSZ

    path = export_fp16_tflite(weights=weights, imgsz=imgsz, repo=REPO)
    print(f"Export complete: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
