"""
TensorRT runtime for batch-1 inference using the modern IO-tensor API.

Used by: tensorrt_benchmarking/trt_inference.py

GPU match: Deserialize only engines built on this same NVIDIA GPU model (and matching driver or arch).
A plan built on another chip is unsupported and may produce wrong results, crashes, deadlocks, or poor
performance. Build a separate .engine on each machine (for example GPU1 laptop and GPU2 workstation).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TrtTensorSpec:
    name: str
    dtype: np.dtype
    shape: tuple[int, ...]
    is_input: bool


class TrtSession:
    """Load a TensorRT engine built for this GPU and reuse buffers for batch-1 inference."""

    def __init__(self, engine_path: str | Path):
        engine_path = Path(engine_path)
        if not engine_path.is_file():
            raise FileNotFoundError(engine_path)

        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as cuda
        import tensorrt as trt

        self._cuda = cuda
        self._trt = trt

        logger = trt.Logger(trt.Logger.WARNING)
        try:
            trt.init_libnvinfer_plugins(logger, "")
        except Exception:
            pass

        runtime = trt.Runtime(logger)
        self._engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self._engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine.")

        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError("Failed to create TensorRT execution context.")

        # Discover IO tensors
        names: list[str] = []
        n_io = int(self._engine.num_io_tensors)
        for i in range(n_io):
            names.append(str(self._engine.get_tensor_name(i)))

        specs: list[TrtTensorSpec] = []
        input_specs: list[TrtTensorSpec] = []
        output_specs: list[TrtTensorSpec] = []
        for name in names:
            mode = self._engine.get_tensor_mode(name)
            is_input = mode == trt.TensorIOMode.INPUT
            dtype = np.dtype(trt.nptype(self._engine.get_tensor_dtype(name)))
            shape = tuple(int(x) for x in self._context.get_tensor_shape(name))
            if any(d <= 0 for d in shape):
                raise RuntimeError(f"Dynamic shape not supported in this harness. Tensor {name} shape={shape}")
            spec = TrtTensorSpec(name=name, dtype=dtype, shape=shape, is_input=is_input)
            specs.append(spec)
            (input_specs if is_input else output_specs).append(spec)

        if len(input_specs) != 1:
            raise RuntimeError(f"Expected exactly 1 input tensor, got {len(input_specs)}: {[s.name for s in input_specs]}")

        self.specs = specs
        self.input = input_specs[0]
        self.outputs = output_specs
        self.stream = cuda.Stream()

        # Allocate pagelocked host + device buffers once
        self._host: dict[str, np.ndarray] = {}
        self._device: dict[str, object] = {}
        for s in specs:
            vol = int(np.prod(s.shape))
            host = cuda.pagelocked_empty(vol, dtype=s.dtype)
            dev = cuda.mem_alloc(int(host.nbytes))
            self._host[s.name] = host
            self._device[s.name] = dev

        # Set tensor addresses once (static pointers)
        for s in specs:
            self._context.set_tensor_address(s.name, int(self._device[s.name]))

    def infer(self, x: np.ndarray) -> list[np.ndarray]:
        """Run one batch-1 inference. Returns outputs as a list of numpy arrays."""
        x = np.ascontiguousarray(x)
        if x.shape != self.input.shape:
            raise ValueError(f"Input shape mismatch: expected {self.input.shape}, got {x.shape}")
        if x.dtype != self.input.dtype:
            raise ValueError(f"Input dtype mismatch: expected {self.input.dtype}, got {x.dtype}")

        cuda = self._cuda

        inp_name = self.input.name
        host_in = self._host[inp_name]
        np.copyto(host_in, x.ravel(), casting="unsafe")
        cuda.memcpy_htod_async(self._device[inp_name], host_in, self.stream)

        ok = self._context.execute_async_v3(self.stream.handle)
        if ok is False:
            raise RuntimeError("TensorRT execute_async_v3 returned False.")

        for s in self.outputs:
            cuda.memcpy_dtoh_async(self._host[s.name], self._device[s.name], self.stream)
        self.stream.synchronize()

        out: list[np.ndarray] = []
        for s in self.outputs:
            out.append(self._host[s.name].reshape(s.shape).copy())
        return out

