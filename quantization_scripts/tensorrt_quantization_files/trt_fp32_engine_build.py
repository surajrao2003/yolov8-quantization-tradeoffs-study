"""
TensorRT FP32 engine from ONNX (needs CUDA GPU + TensorRT + pycuda).

Run from repo root (`yolov8-quantization-tradeoffs-study/`):

  python quantization_scripts/tensorrt_quantization_files/trt_fp32_engine_build.py --onnx models/onnx_quantized_models/FP32_onnx_models/640/yolov8n_640.onnx --imgsz 640

PowerShell OK with backslashes. Optional: --workspace-gb 4 --output path/to/engine.engine

Writes: models/tensorrt_quantized_models/FP32_trt_models/<imgsz>/<onnx_stem>_trt_fp32.engine

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

from _tensorrt_common import REPO, out_engine_dir, resolve_onnx  # noqa: E402
from _trt_builder import serialize_engine_from_onnx  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="ONNX → TensorRT FP32 engine.")
    p.add_argument(
        "--onnx",
        type=Path,
        required=True,
        help="FP32 ONNX path (static batch recommended).",
    )
    p.add_argument("--imgsz", type=int, choices=(320, 640, 1280), required=True)
    p.add_argument(
        "--workspace-gb",
        type=float,
        default=4.0,
        help="GPU workspace ceiling for TensorRT builder (GiB equivalent). Default: 4",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Explicit .engine path (default models/tensorrt_quantized_models/FP32_trt_models/<imgsz>/).",
    )
    args = p.parse_args(argv)

    onnx = resolve_onnx(args.onnx)
    imgsz = int(args.imgsz)
    if imgsz <= 0:
        raise ValueError("imgsz must be positive")

    out_eng = Path(args.output) if args.output else out_engine_dir("FP32_trt_models", imgsz, REPO) / f"{onnx.stem}_trt_fp32.engine"
    out_eng.parent.mkdir(parents=True, exist_ok=True)

    blob = serialize_engine_from_onnx(
        onnx,
        fp16=False,
        use_int8=False,
        int8_calibrator=None,
        workspace_gb=float(args.workspace_gb),
    )
    out_eng.write_bytes(blob)
    print(f"Serialized engine saved: {out_eng}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
