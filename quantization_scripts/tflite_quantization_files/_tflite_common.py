"""Paths and file moves shared by quantization_scripts/tflite_quantization_files/*.py."""
from __future__ import annotations

import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent


def resolve_weights(path: Path) -> Path:
    p = Path(path).expanduser()
    if p.is_file():
        return p.resolve()
    cand = (REPO / p).resolve()
    if cand.is_file():
        return cand
    raise FileNotFoundError(f"Weights not found: {path}")


def tflite_out_dir(repo: Path, category: str, imgsz: int) -> Path:
    d = repo / "models" / "tflite_quantized_models" / category / str(imgsz)
    d.mkdir(parents=True, exist_ok=True)
    return d


def sanitize_export_return(result: str | Path | tuple | list) -> str | Path:
    if isinstance(result, (list, tuple)):
        return result[0]
    return result


def move_to_dest(exported: str | Path, dest: Path) -> Path:
    exported = Path(exported)
    if not exported.is_file():
        raise FileNotFoundError(f"Expected a .tflite file at {exported}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.resolve() == exported.resolve():
        return dest
    if dest.exists():
        dest.unlink()
    shutil.move(str(exported), str(dest))
    return dest
