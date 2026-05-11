"""ONNX → serialized TensorRT plan; INT8 uses entropy calibrator."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def serialize_engine_from_onnx(
    onnx_path: Path,
    *,
    fp16: bool,
    use_int8: bool,
    int8_calibrator: object | None,
    workspace_gb: float,
) -> bytes:
    import tensorrt as trt

    onnx_path = Path(onnx_path)
    logger = trt.Logger(trt.Logger.WARNING)

    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    with onnx_path.open("rb") as f:
        parsed = parser.parse(f.read())

    if not parsed:
        err_lines: list[str] = []
        try:
            for ei in range(int(parser.num_errors)):
                try:
                    err_lines.append(parser.get_error(ei).desc())
                except Exception:
                    pass
        except Exception:
            pass
        raise RuntimeError(
            "TensorRT could not parse ONNX:\n  " + "\n  ".join(err_lines[:20] or ["(no parser detail)"])
        )

    config = builder.create_builder_config()
    from _tensorrt_common import configure_workspace  # noqa: WPS433

    configure_workspace(builder, config, workspace_gb)

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if use_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)
        if int8_calibrator is None:
            raise ValueError("INT8 build requires an int8_calibrator.")
        config.int8_calibrator = int8_calibrator

    if hasattr(builder, "build_serialized_network"):
        ser = builder.build_serialized_network(network, config)
        if ser is None:
            raise RuntimeError(
                "TensorRT build_serialized_network returned None (unsupported graph, GPU/driver, or OOM)."
            )
        return bytes(ser)

    engine = builder.build_engine(network, config)
    if engine is None:
        raise RuntimeError("TensorRT build_engine returned None.")
    return bytes(engine.serialize())


def make_entropy_calibrator(
    onnx_path: Path,
    *,
    batch_arrays_host: Sequence[np.ndarray],
    cache_file: Path,
):
    import tensorrt as trt

    if not Path(onnx_path).is_file():
        raise FileNotFoundError(onnx_path)

    try:
        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as cuda
    except ImportError as e:
        raise RuntimeError("INT8 engine build requires pycuda and a working CUDA runtime.") from e

    blobs = [np.ascontiguousarray(x, dtype=np.float32) for x in batch_arrays_host]
    if not blobs:
        raise ValueError("batch_arrays_host is empty.")

    bytes_per = blobs[0].nbytes
    if any(b.shape != blobs[0].shape for b in blobs):
        nbytes = max(b.nbytes for b in blobs)
    else:
        nbytes = bytes_per

    class _Calib(trt.IInt8EntropyCalibrator2):  # type: ignore[misc, valid-type]
        def __init__(self_inner) -> None:  # noqa: N807
            trt.IInt8EntropyCalibrator2.__init__(self_inner)
            self_inner._blobs = blobs
            self_inner._idx = 0
            self_inner._cuda = cuda
            self_inner._dev = cuda.mem_alloc(nbytes)
            self_inner._stream = cuda.Stream()
            self_inner._cache_file = str(cache_file)

        def get_algorithm(self_inner):  # noqa: ANN001
            return trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2

        def get_batch_size(self_inner) -> int:
            return max(b.shape[0] for b in self_inner._blobs)

        def get_batch(self_inner, names):  # noqa: ANN001
            i = self_inner._idx
            if i >= len(self_inner._blobs):
                return None
            block = self_inner._blobs[i]
            self_inner._idx += 1
            self_inner._cuda.memcpy_htod_async(self_inner._dev, block, self_inner._stream)
            self_inner._stream.synchronize()
            ptr = int(self_inner._dev)
            return [ptr for _ in names]

        def read_calibration_cache(self_inner) -> bytes | None:
            p = Path(self_inner._cache_file)
            if not p.is_file():
                return None
            return p.read_bytes()

        def write_calibration_cache(self_inner, cache) -> None:  # noqa: ANN001
            Path(self_inner._cache_file).write_bytes(cache)

    return _Calib()
