"""
TensorRT INT8 engine from ONNX + image calibration.

Run (defaults: calib quantization_scripts/calibration_dataset, 50 images):

  python quantization_scripts/tensorrt_quantization_files/trt_int8_engine_build.py --onnx models/onnx_quantized_models/FP32_onnx_models/640/yolov8n_640.onnx --imgsz 640

Override calibration:

  python quantization_scripts/tensorrt_quantization_files/trt_int8_engine_build.py --onnx ... --imgsz 640 --calib-dir imagedata/images --num-calib 100

Writes: models/tensorrt_quantized_models/INT8_trt_models/<imgsz>/<onnx_stem>_trt_int8.engine

GPU match : A .engine file is a TensorRT plan tied to the GPU (and driver or arch settings)
it was built on. Building on one NVIDIA model (for example RTX 4050) and loading on another (for example
RTX 4000 Ada), or swapping laptop versus desktop chips, is not supported. TensorRT may warn that you can
get wrong results, crashes, deadlocks, or only slower runs. For this project, treat GPU1 and GPU2 as
needing their own builds: build the engine on the same machine where you benchmark it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_TD = Path(__file__).resolve().parent
if str(_TD) not in sys.path:
    sys.path.insert(0, str(_TD))

from _tensorrt_common import (  # noqa: E402
    REPO,
    calibration_inputs_list,
    iter_calibration_images,
    out_engine_dir,
    resolve_onnx,
)
from _trt_builder import make_entropy_calibrator, serialize_engine_from_onnx  # noqa: E402

_DEFAULT_CALIB_DIR = REPO / "quantization_scripts" / "calibration_dataset"
_DEFAULT_NUM_CALIB = 50


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="ONNX → TensorRT INT8 engine with calibration.")
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--imgsz", type=int, choices=(320, 640, 1280), required=True)
    p.add_argument(
        "--calib-dir",
        type=Path,
        default=None,
        help="Calibration images (.jpg/.png…). Default: quantization_scripts/calibration_dataset",
    )
    p.add_argument(
        "--num-calib",
        type=int,
        default=_DEFAULT_NUM_CALIB,
        help=f"Max images to use for calibration tensors (batch 1 each). Default: {_DEFAULT_NUM_CALIB}",
    )
    p.add_argument("--workspace-gb", type=float, default=4.0)
    p.add_argument(
        "--cache-file",
        type=Path,
        default=None,
        help="Entropy calibration cache path (default beside output engine).",
    )
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args(argv)

    onnx = resolve_onnx(args.onnx)

    imgsz = int(args.imgsz)
    calib_raw = Path(args.calib_dir) if args.calib_dir is not None else _DEFAULT_CALIB_DIR
    calib = calib_raw if calib_raw.is_dir() else (REPO / calib_raw).resolve()
    if not calib.is_dir():
        raise FileNotFoundError(f"Calibration folder not found: {args.calib_dir}")

    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    paths = iter_calibration_images(calib, exts)
    if not paths:
        raise FileNotFoundError(f"No calibration images matched {exts} under {calib}")

    tensors = calibration_inputs_list(paths, (imgsz, imgsz), int(args.num_calib))

    out_eng = Path(args.output) if args.output else out_engine_dir("INT8_trt_models", imgsz, REPO) / f"{onnx.stem}_trt_int8.engine"
    out_eng.parent.mkdir(parents=True, exist_ok=True)

    cache = Path(args.cache_file) if args.cache_file else out_eng.with_suffix("").parent / (out_eng.stem + "_calibration.cache")

    calibrator = make_entropy_calibrator(onnx, batch_arrays_host=tensors, cache_file=cache)

    blob = serialize_engine_from_onnx(
        onnx,
        fp16=False,
        use_int8=True,
        int8_calibrator=calibrator,
        workspace_gb=float(args.workspace_gb),
    )
    out_eng.write_bytes(blob)
    print(f"Serialized engine saved: {out_eng}")
    print(f"Calibration cache (if written): {cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
