"""
Dynamic INT8 (OnnxRuntime quantize_dynamic): mostly INT8 weights, no calibration dataset needed.

How to run
----------
1. Open your terminal in the project root.
2. Point --model-fp32 at FP32 ONNX, for example:

       models/onnx_quantized_models/FP32_onnx_models/640/yolov8m_640.onnx

3. Set --img-size to the same export size (320, 640 or 1280). It only sets the output folder.

Example (PowerShell, one line):

       python quantization_scripts/onnx_quantization_files/INT8_dynamic_quantization_onnx.py --model-fp32 "models/onnx_quantized_models/FP32_onnx_models/640/yolov8m_640.onnx" --img-size 640

Default output:

       models/onnx_quantized_models/INT8_onnx_dynamic_quantized_models/<img-size>/<base>_<img-size>_dynamic_quantized.onnx

``<base>`` is the FP32 stem without a trailing ``_<img-size>`` (e.g. ``yolov8n_640.onnx`` → ``yolov8n_640_dynamic_quantized.onnx``).

Weights use ``QuantType.QUInt8``. Only ``MatMul`` and ``Gemm`` are quantized — ``Conv`` is excluded so ORT does not insert
``ConvInteger`` nodes (which often lack usable CPU/GPU kernels). Convs stay FP32.

Use --output if you want a custom path.
"""

from __future__ import annotations


import argparse
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic


def _default_dynamic_quant_filename(model_fp32: Path, img_size: int) -> str:
    """
    ``{base}_{img_size}_dynamic_quantized.onnx`` so names match resolution (e.g. yolov8n_320_...).
    If the FP32 stem already ends with ``_<img_size>``, that suffix is stripped once before composing.
    """
    stem = model_fp32.stem
    suffix = f"_{img_size}"
    if stem.endswith(suffix):
        base = stem[: -len(suffix)]
    else:
        base = stem
    if not base:
        base = stem
    return f"{base}_{img_size}_dynamic_quantized.onnx"


def _default_output_path(model_fp32: Path, img_size: int) -> Path:
    return (
        Path("models")
        / "onnx_quantized_models"
        / "INT8_onnx_dynamic_quantized_models"
        / str(img_size)
        / _default_dynamic_quant_filename(model_fp32, img_size)
    )


def dynamic_quantize_yolov8_onnx(input_model_path: Path, output_model_path: Path) -> None:
    input_model_path = Path(input_model_path)
    output_model_path = Path(output_model_path)

    if not input_model_path.is_file():
        raise FileNotFoundError(f"Input ONNX not found: {input_model_path}")

    output_model_path.parent.mkdir(parents=True, exist_ok=True)

    quantize_dynamic(
        model_input=str(input_model_path),
        model_output=str(output_model_path),
        weight_type=QuantType.QUInt8,
        per_channel=False,
        reduce_range=False,
        op_types_to_quantize=["MatMul", "Gemm"],
    )

    print(f"Saved: {output_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dynamic INT8 weight quantization for YOLOv8 ONNX (ORT quantize_dynamic)."
    )
    parser.add_argument(
        "--model-fp32",
        type=Path,
        required=True,
        help="FP32 ONNX under models/onnx_quantized_models/FP32_onnx_models/<size>/....",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        required=True,
        help="Matches export size (320/640/1280); used for default output folder and filename.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path. Default: .../INT8_dynamic_quantized_models/<img-size>/<base>_<img-size>_dynamic_quantized.onnx",
    )

    args = parser.parse_args()
    out = args.output or _default_output_path(args.model_fp32, args.img_size)

    dynamic_quantize_yolov8_onnx(args.model_fp32, out)
