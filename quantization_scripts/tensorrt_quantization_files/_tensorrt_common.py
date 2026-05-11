"""Helpers for TensorRT ONNX→engine scripts (paths, calibration tensors, workspace)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent


def prepend_code_files() -> Path:
    cf = REPO / "code_files"
    if str(cf) not in sys.path:
        sys.path.insert(0, str(cf))
    return cf


def resolve_onnx(path: Path | str, repo: Path = REPO) -> Path:
    p = Path(path).expanduser()
    if p.is_file():
        return p.resolve()
    cand = (repo / p).resolve()
    if cand.is_file():
        return cand
    raise FileNotFoundError(f"ONNX not found: {path}")


def default_onnx_fp32(stem: str, imgsz: int, repo: Path = REPO) -> Path:
    return repo / "models" / "onnx_quantized_models" / "FP32_onnx_models" / str(imgsz) / f"{stem}_{imgsz}.onnx"


def out_engine_dir(kind: str, imgsz: int, repo: Path = REPO) -> Path:
    root = repo / "models" / "tensorrt_quantized_models" / kind / str(imgsz)
    root.mkdir(parents=True, exist_ok=True)
    return root


def iter_calibration_images(calib_dir: Path, extensions: tuple[str, ...]) -> list[Path]:
    paths: list[Path] = []
    for pat in extensions:
        paths.extend(sorted(calib_dir.glob(pat)))
    return paths


def calibration_inputs_list(
    paths: list[Path],
    img_size: tuple[int, int],
    max_images: int,
) -> list[np.ndarray]:
    prepend_code_files()
    import cv2
    from preprocessing import preprocess_frame  # noqa: WPS433

    out: list[np.ndarray] = []
    for pth in paths:
        frame = cv2.imread(str(pth))
        if frame is None:
            continue
        inp, *_rest = preprocess_frame(frame, img_size, feed_dtype=np.float32)
        if inp.shape[0] != 1:
            raise ValueError(f"Expected batch dimension 1, got shape {inp.shape}")
        out.append(inp)
        if len(out) >= max_images:
            break
    if not out:
        raise FileNotFoundError("No usable calibration images (failed cv2.imread).")
    return out


def configure_workspace(_builder, config, workspace_gb: float) -> None:
    nbytes = max(int(workspace_gb * (1 << 30)), int(512 * (1 << 20)))
    try:
        import tensorrt as trt  # noqa: PLC0415

        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, nbytes)
    except (AttributeError, TypeError):
        if hasattr(config, "max_workspace_size"):
            setattr(config, "max_workspace_size", nbytes)
