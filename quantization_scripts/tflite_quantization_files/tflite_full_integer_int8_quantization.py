"""
YOLOv8 .pt → full INT8 TFLite (--data YAML points at calibration images).

Run:

  python quantization_scripts/tflite_quantization_files/tflite_full_integer_int8_quantization.py --weights models/yolo_models/yolov8n.pt --imgsz 640 --data models/config_custom_data.yaml

Quick calib set from Ultralytics (may download):

  python quantization_scripts/tflite_quantization_files/tflite_full_integer_int8_quantization.py --weights models/yolo_models/yolov8n.pt --imgsz 640 --data coco128.yaml

Writes models/tflite_quantized_models/full_integer_INT8_tflite_models/<imgsz>/
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
DEFAULT_DATA_YAML = REPO / "models" / "config_custom_data.yaml"


def export_full_int8_tflite(*, weights: Path, imgsz: int, data_yaml: Path, repo: Path = REPO) -> Path:
    weights = resolve_weights(weights)

    cand = Path(data_yaml).expanduser()
    if cand.is_file():
        data_spec = str(cand.resolve())
    elif (repo / cand).is_file():
        data_spec = str((repo / cand).resolve())
    elif cand.is_absolute():
        raise FileNotFoundError(f"Dataset YAML not found: {cand}")
    else:
        # e.g. ``coco128.yaml`` resolved by Ultralytics under ``cfg/datasets/``
        data_spec = cand.as_posix()

    out_dir = tflite_out_dir(repo, "full_integer_INT8_tflite_models", imgsz)
    dest = out_dir / f"{weights.stem}_{imgsz}_int8_full_integer.tflite"

    model = YOLO(str(weights))
    exported = sanitize_export_return(
        model.export(
            format="tflite",
            imgsz=imgsz,
            half=False,
            int8=True,
            data=data_spec,
            simplify=True,
            nms=True,
        )
    )
    return move_to_dest(exported, dest)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Export YOLOv8 .pt to full INT8 TFLite (Ultralytics, calibrated).")
    p.add_argument("--weights", type=Path, default=None, help="Path to .pt (default: yolov8n under models/yolo_models/).")
    p.add_argument("--imgsz", type=int, choices=(320, 640, 1280), default=None, help=f"Square export size. Default: {IMGSZ}")
    p.add_argument(
        "--data",
        type=Path,
        default=None,
        help=f"Ultralytics data YAML for INT8 calibration. Default: {DEFAULT_DATA_YAML.name}",
    )
    args = p.parse_args(argv)

    weights = args.weights if args.weights is not None else WEIGHTS_PT
    imgsz = args.imgsz if args.imgsz is not None else IMGSZ
    data_yaml = args.data if args.data is not None else DEFAULT_DATA_YAML

    path = export_full_int8_tflite(weights=weights, imgsz=imgsz, data_yaml=data_yaml, repo=REPO)
    print(f"Export complete: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
