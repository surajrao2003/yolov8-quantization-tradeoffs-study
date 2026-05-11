"""
YOLOv8 .pt → SavedModel → TFLite dynamic-range (TensorFlow Optimize.DEFAULT).

Run:

  python quantization_scripts/tflite_quantization_files/tflite_dynamic_range_quantization.py --weights models/yolo_models/yolov8n.pt --imgsz 640

Needs tensorflow installed.

Writes models/tflite_quantized_models/dynamic_range_tflite_models/<imgsz>/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import tensorflow as tf
from ultralytics import YOLO

from _tflite_common import REPO, resolve_weights, sanitize_export_return, tflite_out_dir

IMGSZ = 640
WEIGHTS_PT = REPO / "models" / "yolo_models" / "yolov8n.pt"


def saved_model_path_for_weights(weights: Path) -> Path:
    return weights.parent / f"{weights.stem}_saved_model"


def export_dynamic_range_tflite(*, weights: Path, imgsz: int, repo: Path = REPO) -> Path:
    weights = resolve_weights(weights)
    sm_dir = saved_model_path_for_weights(weights)

    model = YOLO(str(weights))
    out = sanitize_export_return(model.export(format="saved_model", imgsz=imgsz, simplify=True, nms=True))
    saved = Path(out)
    if not saved.is_dir():
        saved = sm_dir
    if not saved.is_dir():
        raise FileNotFoundError(f"SavedModel folder not found (expected Ultralytics export at {saved})")

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_blob = converter.convert()

    dest_dir = tflite_out_dir(repo, "dynamic_range_tflite_models", imgsz)
    dest = dest_dir / f"{weights.stem}_{imgsz}_dynamic_range.tflite"
    dest.write_bytes(tflite_blob)
    return dest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="YOLOv8 .pt → TFLite with TensorFlow dynamic-range quantization.")
    p.add_argument("--weights", type=Path, default=None, help="Path to .pt (default: yolov8n under models/yolo_models/).")
    p.add_argument("--imgsz", type=int, choices=(320, 640, 1280), default=None, help=f"Square export size. Default: {IMGSZ}")
    args = p.parse_args(argv)

    weights = args.weights if args.weights is not None else WEIGHTS_PT
    imgsz = args.imgsz if args.imgsz is not None else IMGSZ

    path = export_dynamic_range_tflite(weights=weights, imgsz=imgsz, repo=REPO)
    print(f"Export complete: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
