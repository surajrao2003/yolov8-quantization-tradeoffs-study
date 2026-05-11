"""
Static INT8 quantization (OnnxRuntime, QDQ style) using calibration images.

How to run
----------
1. Open your terminal in the project root.
2. Prepare a folder of sample images (.jpg/.png) for calibration data. They should look like your real data.
3. Point --model-fp32 at FP32 ONNX under:

       models/onnx_quantized_models/FP32_onnx_models/<size>/

   where <size> is 320, 640, or 1280 (folder name matches resolution).

4. Set --img-size to match that export (320, 640, or 1280).
5. Set --calib-dir to your calibration image folder.

Example (PowerShell, one line):

       python quantization_scripts/onnx_quantization_files/INT8_static_quantization_onnx.py --model-fp32 "models/onnx_quantized_models/FP32_onnx_models/640/yolov8n_640.onnx" --img-size 640 --calib-dir "quantization_scripts/calibration_dataset"

Default output:

       models/onnx_quantized_models/INT8_onnx_static_quantized_models/<img-size>/<name>_static_quantized.onnx

If your FP32 ONNX is ai.onnx opset < 13 (e.g. legacy opset 12), this script ONNX version-converts
it (default target opset 17) before quantization so QDQ DequantizeLinear nodes stay valid.

Note: Inference in this repo feeds input name "images" as float tensors. Change --input-name only if your ONNX uses a different name.
"""

from __future__ import annotations

import argparse

from pathlib import Path

import cv2
import numpy as np
import onnx
from onnx import version_converter
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quant_pre_process,
    quantize_static,
)

DEFAULT_MAX_CALIB_IMAGES = 300
# DequantizeLinear with per-channel `axis` is only valid from ai.onnx opset 13 onward.
_MIN_FP32_OPSET_FOR_STATIC_QDQ = 13
_DEFAULT_TARGET_OPSET = 17


def _ai_onnx_opset_version(model_path: Path) -> int | None:
    m = onnx.load(str(model_path))
    for oi in m.opset_import:
        if oi.domain in ("", "ai.onnx", None):
            return int(oi.version)
    return None


def prepare_fp32_onnx_for_static_qdq(
    model_fp32: Path,
    *,
    target_opset: int = _DEFAULT_TARGET_OPSET,
    converted_out: Path | None = None,
) -> Path:
    """
    Return a path to an FP32 ONNX suitable for static QDQ (ai.onnx opset >= 13).
    If ``model_fp32`` is already >= 13, returns it unchanged.
    Otherwise writes a version-converted copy and returns that path.
    """
    model_fp32 = Path(model_fp32).resolve()
    v = _ai_onnx_opset_version(model_fp32)
    if v is None or v >= _MIN_FP32_OPSET_FOR_STATIC_QDQ:
        return model_fp32

    if target_opset < _MIN_FP32_OPSET_FOR_STATIC_QDQ:
        raise ValueError(
            f"--target-opset must be >= {_MIN_FP32_OPSET_FOR_STATIC_QDQ} when converting "
            f"(got {target_opset}); otherwise QDQ DequantizeLinear remains invalid."
        )

    if converted_out is None:
        converted_out = model_fp32.with_name(
            f"{model_fp32.stem}_converted_opset{target_opset}{model_fp32.suffix}"
        )
    else:
        converted_out = Path(converted_out).resolve()

    converted_out.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"FP32 model ai.onnx opset is {v}; converting to {target_opset} -> {converted_out}"
    )
    model = onnx.load(str(model_fp32))
    converted = version_converter.convert_version(model, target_opset)
    onnx.checker.check_model(converted)
    onnx.save(converted, str(converted_out))
    return converted_out


def letterbox(image, new_shape=(640, 640), color=(114, 114, 114)):
    h, w = image.shape[:2]
    new_h, new_w = new_shape

    scale = min(new_w / w, new_h / h)
    resized_w, resized_h = int(round(w * scale)), int(round(h * scale))

    pad_w = new_w - resized_w
    pad_h = new_h - resized_h

    pad_left = int(round(pad_w / 2 - 0.1))
    pad_right = int(round(pad_w / 2 + 0.1))
    pad_top = int(round(pad_h / 2 - 0.1))
    pad_bottom = int(round(pad_h / 2 + 0.1))

    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=color,
    )

    return padded


def preprocess_image(image_path: Path, img_size: int = 640) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = letterbox(image, (img_size, img_size))

    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
    image = np.expand_dims(image, axis=0)   # CHW -> NCHW

    image = np.ascontiguousarray(image)
    return image.astype(np.float32)


class YOLOv8CalibrationDataReader(CalibrationDataReader):
    def __init__(
        self,
        image_dir: Path,
        model_path: Path,
        *,
        img_size: int = 640,
        max_images: int = DEFAULT_MAX_CALIB_IMAGES,
        input_name: str | None = None,
    ):
        self.image_dir = Path(image_dir)
        self.img_size = int(img_size)

        image_paths: list[Path] = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            image_paths.extend(self.image_dir.glob(ext))

        self.image_paths = sorted(image_paths)[: int(max_images)]
        if not self.image_paths:
            raise FileNotFoundError(f"No calibration images found in {self.image_dir}")

        if input_name is None:
            model = onnx.load(str(model_path))
            input_name = model.graph.input[0].name
        self.input_name = input_name
        self.data_iter = iter(self.image_paths)

    def get_next(self):
        try:
            image_path = next(self.data_iter)
        except StopIteration:
            return None

        input_tensor = preprocess_image(image_path, self.img_size)
        return {self.input_name: input_tensor}

    def rewind(self):
        self.data_iter = iter(self.image_paths)


def _default_preprocessed_path(model_fp32: Path) -> Path:
    return model_fp32.with_suffix("").with_name(f"{model_fp32.stem}_preprocessed.onnx")


def _default_output_path(model_fp32: Path, img_size: int) -> Path:
    # Keep outputs inside repo by default (relative to current working dir)
    # and group by input size for easy sweeps.
    return (
        Path("models")
        / "onnx_quantized_models"
        / "INT8_onnx_static_quantized_models"
        / str(img_size)
        / f"{model_fp32.stem}_static_quantized.onnx"
    )


def quantize_yolov8_static_int8(
    *,
    model_fp32: Path,
    calib_image_dir: Path,
    img_size: int,
    output_int8: Path,
    output_preprocessed: Path | None = None,
    max_calib_images: int = DEFAULT_MAX_CALIB_IMAGES,
    input_name: str | None = "images",
    target_opset: int = _DEFAULT_TARGET_OPSET,
    converted_fp32_out: Path | None = None,
):
    model_fp32 = Path(model_fp32)
    calib_image_dir = Path(calib_image_dir)
    output_int8 = Path(output_int8)

    effective_fp32 = prepare_fp32_onnx_for_static_qdq(
        model_fp32,
        target_opset=target_opset,
        converted_out=converted_fp32_out,
    )

    if output_preprocessed is None:
        output_preprocessed = _default_preprocessed_path(effective_fp32)
    else:
        output_preprocessed = Path(output_preprocessed)

    output_int8.parent.mkdir(parents=True, exist_ok=True)
    output_preprocessed.parent.mkdir(parents=True, exist_ok=True)

    quant_pre_process(
        input_model=str(effective_fp32),
        output_model_path=str(output_preprocessed),
        skip_optimization=False,
        skip_onnx_shape=False,
        skip_symbolic_shape=False,
    )

    calibration_reader = YOLOv8CalibrationDataReader(
        image_dir=calib_image_dir,
        model_path=output_preprocessed,
        img_size=img_size,
        max_images=max_calib_images,
        input_name=input_name,
    )

    quantize_static(
        model_input=str(output_preprocessed),
        model_output=str(output_int8),
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        per_channel=True,
        reduce_range=False,
        op_types_to_quantize=["Conv"],
        extra_options={
            "ActivationSymmetric": True,
            "WeightSymmetric": True,
        },
    )

    print(f"Saved INT8 model: {output_int8}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 ONNX Runtime INT8 static quantization (QDQ).")
    parser.add_argument(
        "--model-fp32",
        type=Path,
        required=True,
        help="FP32 ONNX (e.g. models/onnx_quantized_models/FP32_onnx_models/640/model.onnx).",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        required=True,
        help="Input image size used when exporting the ONNX (e.g. 320/640/1280).",
    )
    parser.add_argument(
        "--calib-dir",
        type=Path,
        required=True,
        help=(
            "Directory with calibration images (.jpg/.png/etc); relative to CWD or repo root. "
            "Example: quantization_scripts/calibration_dataset"
        ),
    )
    parser.add_argument(
        "--max-calib-images",
        type=int,
        default=DEFAULT_MAX_CALIB_IMAGES,
        help=f"Max calibration images to use (default: {DEFAULT_MAX_CALIB_IMAGES}).",
    )
    parser.add_argument(
        "--output-int8",
        type=Path,
        default=None,
        help="Where to write the quantized INT8 model. If omitted, a repo-local default path is used.",
    )
    parser.add_argument(
        "--output-preprocessed",
        type=Path,
        default=None,
        help="Preprocessed FP32 before quantize_static. If omitted, next to the effective FP32 input "
        "(if opset was auto-converted, that is the converted copy, not necessarily --model-fp32).",
    )
    parser.add_argument(
        "--input-name",
        type=str,
        default="images",
        help="Model input name. Default is 'images' to match your inference pipeline.",
    )
    parser.add_argument(
        "--target-opset",
        type=int,
        default=_DEFAULT_TARGET_OPSET,
        help=f"If FP32 ai.onnx opset < {_MIN_FP32_OPSET_FOR_STATIC_QDQ}, convert to this opset "
        f"(default: {_DEFAULT_TARGET_OPSET}).",
    )
    parser.add_argument(
        "--converted-fp32-out",
        type=Path,
        default=None,
        help="Where to save opset-converted FP32 when conversion runs. "
        "Default: next to --model-fp32 as <stem>_converted_opset<target>.onnx",
    )

    args = parser.parse_args()

    output_int8 = args.output_int8 or _default_output_path(args.model_fp32, args.img_size)

    quantize_yolov8_static_int8(
        model_fp32=args.model_fp32,
        calib_image_dir=args.calib_dir,
        img_size=args.img_size,
        output_int8=output_int8,
        output_preprocessed=args.output_preprocessed,
        max_calib_images=args.max_calib_images,
        input_name=args.input_name,
        target_opset=args.target_opset,
        converted_fp32_out=args.converted_fp32_out,
    )