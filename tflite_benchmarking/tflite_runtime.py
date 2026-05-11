"""
TFLite runtime for batch-1 inference (reuses tensors across calls).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _gpu_delegate_candidates() -> list[str | Path]:
    """Paths / short names to try with tf.lite.experimental.load_delegate."""
    env = os.environ.get("TFLITE_GPU_DELEGATE", "").strip()
    if env:
        return [Path(env).expanduser()]

    if sys.platform == "win32":
        # Pip TensorFlow wheels often ship no delegate; avoid .so probes that add noise on Windows.
        return ["tensorflowlite_gpu_delegate.dll"]

    return [
        "libtensorflowlite_gpu_delegate.so",
        "tensorflowlite_gpu_delegate.so",
    ]


@dataclass(frozen=True)
class TfliteTensorSpec:
    index: int
    name: str
    dtype: np.dtype
    shape: tuple[int, ...]
    scale: float | None
    zero_point: int | None


def _quant_params(details: dict) -> tuple[float | None, int | None]:
    q = details.get("quantization", None)
    if not q:
        return None, None
    scale, zp = q
    if scale in (None, 0.0):
        return None, None
    return float(scale), int(zp)


class TfliteSession:
    def __init__(self, model_path: str | Path, *, device: str = "cpu"):
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(model_path)

        import tensorflow as tf

        self._tf = tf
        dev = (device or "cpu").lower()
        if dev not in {"cpu", "gpu"}:
            raise ValueError("device must be 'cpu' or 'gpu'")
        self.device = dev

        delegates = None
        if dev == "gpu":
            load_delegate = getattr(tf.lite.experimental, "load_delegate", None)
            if load_delegate is None:
                raise RuntimeError("This TensorFlow build does not expose tf.lite.experimental.load_delegate().")

            tried: list[str] = []
            last_err: Exception | None = None
            for cand in _gpu_delegate_candidates():
                label = str(cand)
                tried.append(label)
                try:
                    delegates = [load_delegate(str(cand))]
                    break
                except Exception as e:  # noqa: BLE001
                    last_err = e
            if delegates is None:
                hint = (
                    "Standard `pip install tensorflow` on Windows does not include "
                    "tensorflowlite_gpu_delegate.dll. Use --device cpu for TFLite, or "
                    "NVIDIA GPU + tensorrt_benchmarking/ for GPU inference. "
                    "To try a manually installed delegate, set env TFLITE_GPU_DELEGATE "
                    "to the full path of the delegate DLL/SO. "
                )
                raise RuntimeError(
                    "TFLite GPU delegate could not be loaded. " + hint + f"Tried: {tried}. Last error: {last_err!r}"
                )

        self._interp = tf.lite.Interpreter(model_path=str(model_path), experimental_delegates=delegates)
        self._interp.allocate_tensors()

        in_det = self._interp.get_input_details()
        out_det = self._interp.get_output_details()
        if len(in_det) != 1:
            raise RuntimeError(f"Expected 1 input tensor, got {len(in_det)}")

        d0 = in_det[0]
        s, zp = _quant_params(d0)
        self.input = TfliteTensorSpec(
            index=int(d0["index"]),
            name=str(d0.get("name", "input0")),
            dtype=np.dtype(d0["dtype"]),
            shape=tuple(int(x) for x in d0["shape"]),
            scale=s,
            zero_point=zp,
        )

        outs: list[TfliteTensorSpec] = []
        for od in out_det:
            s1, zp1 = _quant_params(od)
            outs.append(
                TfliteTensorSpec(
                    index=int(od["index"]),
                    name=str(od.get("name", "output")),
                    dtype=np.dtype(od["dtype"]),
                    shape=tuple(int(x) for x in od["shape"]),
                    scale=s1,
                    zero_point=zp1,
                )
            )
        self.outputs = outs

    def infer(self, x_fp32_nchw: np.ndarray) -> list[np.ndarray]:
        x = np.ascontiguousarray(x_fp32_nchw)
        # Support both NCHW (repo preprocessing) and NHWC (common TFLite exports).
        if tuple(x.shape) != self.input.shape:
            if (
                x.ndim == 4
                and self.input.shape == (int(x.shape[0]), int(x.shape[2]), int(x.shape[3]), int(x.shape[1]))
            ):
                x = np.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
            else:
                raise ValueError(
                    f"Input shape mismatch: expected {self.input.shape}, got {tuple(x.shape)}"
                )

        # Input cast/quantize
        if np.issubdtype(self.input.dtype, np.floating):
            x_in = x.astype(self.input.dtype, copy=False)
        else:
            if self.input.scale is None or self.input.zero_point is None:
                raise RuntimeError("Integer input tensor without quantization params.")
            q = np.round(x / self.input.scale + self.input.zero_point)
            info = np.iinfo(self.input.dtype)
            x_in = np.clip(q, info.min, info.max).astype(self.input.dtype)

        self._interp.set_tensor(self.input.index, x_in)
        self._interp.invoke()

        outs: list[np.ndarray] = []
        for o in self.outputs:
            y = self._interp.get_tensor(o.index)
            if not np.issubdtype(o.dtype, np.floating):
                if o.scale is None or o.zero_point is None:
                    raise RuntimeError("Integer output tensor without quantization params.")
                y = (y.astype(np.float32) - float(o.zero_point)) * float(o.scale)
            outs.append(y)
        return outs

