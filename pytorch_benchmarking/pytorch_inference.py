"""Backend for pytorch_benchmarking/main_execution.py — YOLO.predict + same metrics code as code_files."""
from __future__ import annotations

import os
import sys
import time
import inspect
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

_REPO_DIR = Path(__file__).resolve().parent.parent
_CODE_FILES = _REPO_DIR / "code_files"
if str(_CODE_FILES) not in sys.path:
    sys.path.insert(0, str(_CODE_FILES))

from inference import draw_predictions, format_model_disk_size  # noqa: E402
from metrics_map import load_yolo_ground_truth, mean_average_precision_person  # noqa: E402

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
PERSON_CLASS_ID = 0
MAP_IOU = 0.5


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _torch_load_module(path: Path, map_location):
    """Load pickled nn.Module; compatible with PyTorch versions before ``weights_only``."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def reported_device_pytorch(device_requested: str) -> str:
    if device_requested == "gpu" and torch.cuda.is_available():
        return "gpu"
    return "cpu"


def load_yolo_model(
    model_path: Path,
    *,
    device: str,
    base_weights: Path | None,
    repo: Path | None = None,
) -> object:
    """
    Load an Ultralytics YOLO model.

    - Standard Ultralytics ``.pt`` checkpoints: ``YOLO(model_path)``.
    - Raw ``torch.nn.Module`` checkpoints (e.g. FP16 / dynamic INT8 exports): pass ``base_weights``
      to the matching official ``.pt`` (same variant n/s/m/l), then swap ``yolo.model``.
    """
    repo = repo or _repo_root()
    mp = Path(model_path)
    if not mp.is_file():
        alt = repo / mp
        if alt.is_file():
            mp = alt
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError("ultralytics is required (pip install ultralytics).") from e

    dev = reported_device_pytorch(device)
    torch_device = torch.device("cuda:0" if dev == "gpu" else "cpu")

    try:
        yolo = YOLO(str(mp))
        yolo.to(torch_device)
        # Some module-only checkpoints may "load" but produce a bare Sequential that can't handle
        # Ultralytics' inference kwargs (augment/visualize/embed). Detect and fall back.
        try:
            sig0 = inspect.signature(yolo.model.forward)  # type: ignore[arg-type]
            p0 = sig0.parameters
            accepts_kwargs0 = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in p0.values()
            )
            accepts_augment0 = "augment" in p0 or accepts_kwargs0
        except Exception:
            accepts_augment0 = False
        if accepts_augment0:
            return yolo
    except Exception:
        pass

    if base_weights is None:
        raise RuntimeError(
            f"Could not load {mp} with ultralytics.YOLO(). "
            "If this file is a saved nn.Module (e.g. FP16 / dynamic INT8 export), pass "
            "--base-weights with the matching Ultralytics checkpoint "
            "(e.g. models/yolo_models/yolov8n.pt)."
        )

    bw = Path(base_weights)
    if not bw.is_file():
        bw = (repo / bw).resolve()
    if not bw.is_file():
        raise FileNotFoundError(f"Base weights not found: {base_weights}")

    yolo = YOLO(str(bw))
    inner = _torch_load_module(mp, torch_device)
    if hasattr(inner, "eval"):
        inner = inner.eval()
    inner = inner.to(torch_device)

    # FP16 on CPU is often unsupported/slow and can raise dtype mismatch errors.
    # For CPU benchmarks we always run FP32.
    requested = reported_device_pytorch(device)
    if requested == "cpu":
        try:
            inner = inner.float()
        except Exception:
            pass
        bench_half = False
    else:
        bench_half = any(
            getattr(p, "dtype", None) == torch.float16 for p in getattr(inner, "parameters", lambda: [])()
        )

    # Ultralytics calls model(..., augment=..., visualize=..., embed=...).
    # If the saved object is a plain Module (often nn.Sequential), it won't accept these kwargs.
    # In that case, keep the Ultralytics wrapper and replace only the underlying .model graph.
    try:
        sig = inspect.signature(inner.forward)  # type: ignore[arg-type]
        params = sig.parameters
        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        accepts_augment = "augment" in params or accepts_kwargs
    except Exception:
        accepts_augment = False

    if isinstance(inner, nn.Sequential) or not accepts_augment:
        if not hasattr(yolo.model, "model"):
            raise RuntimeError(
                "Loaded a module-only checkpoint, but Ultralytics wrapper has no `.model` attribute to swap."
            )
        yolo.model.model = inner  # type: ignore[attr-defined]
    else:
        yolo.model = inner

    # Tell predict() to cast inputs to FP16 when benchmarking FP16 on GPU.
    yolo._bench_half = bool(bench_half)  # type: ignore[attr-defined]

    # Important: Ultralytics predictor.setup_model() fuses Conv+BN with float math by default.
    # If we pre-cast parts of the graph to FP16, the fuse step can hit Half vs Float matmul errors.
    # Keep the wrapper graph FP32 here; we still request FP16 inference via `half=True` in predict().
    if requested == "gpu" and bench_half:
        try:
            yolo.model.float()  # type: ignore[call-arg]
        except Exception:
            pass

    yolo.to(torch_device)
    return yolo


def predict_person_boxes(
    yolo: object,
    frame_bgr: np.ndarray,
    imgsz: int,
    conf: float,
    iou_nms: float,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Run YOLO predict; return person-class XYXY boxes, scores, (H, W)."""
    # Ultralytics expects BGR path or ndarray; single image list returns Results list.
    results = yolo.predict(
        source=frame_bgr,
        imgsz=imgsz,
        conf=conf,
        iou=iou_nms,
        classes=[PERSON_CLASS_ID],
        half=bool(getattr(yolo, "_bench_half", False)),
        verbose=False,
        stream=False,
    )
    r = results[0]
    h, w = frame_bgr.shape[:2]
    if r.boxes is None or len(r.boxes) == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            (h, w),
        )
    boxes = r.boxes.xyxy.cpu().numpy().astype(np.float32)
    scores = r.boxes.conf.cpu().numpy().astype(np.float32)
    return boxes, scores, (h, w)


def run_pytorch_benchmark(
    yolo: object,
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    input_size: tuple[int, int],
    conf: float,
    iou_nms: float,
    *,
    model_path: str,
    device_tag: str,
) -> None:
    labels_path = Path(labels_dir)
    total_time = 0.0
    total_frames = 0
    imgsz = int(input_size[0])

    files = sorted(
        f
        for f in os.listdir(images_dir)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    )

    list_gt: list[np.ndarray] = []
    list_preds: list[tuple[np.ndarray, np.ndarray]] = []

    for image_file in files:
        image_path = os.path.join(images_dir, image_file)
        filename_no_ext = Path(image_file).stem
        t0 = time.time()
        input_frame = cv2.imread(image_path)
        if input_frame is None:
            continue

        boxes, scores, (h, w) = predict_person_boxes(
            yolo, input_frame, imgsz, conf, iou_nms
        )
        result_image = draw_predictions(input_frame, boxes, scores, np.zeros(len(boxes), dtype=np.int64))

        inference_time = time.time() - t0
        total_time += inference_time
        total_frames += 1

        list_gt.append(
            load_yolo_ground_truth(labels_path, filename_no_ext, h, w, PERSON_CLASS_ID)
        )
        list_preds.append((boxes, scores))

        out_path = os.path.join(output_dir, f"{filename_no_ext}.jpg")
        cv2.imwrite(out_path, result_image)

    size_str = format_model_disk_size(model_path)

    if total_frames == 0:
        print("FPS: N/A (no images processed)")
        print(f"device={device_tag}")
        print(f"Model size (on disk): {size_str}")
        print(f"mAP@IoU={MAP_IOU:.2f} : N/A (no images processed)")
        return

    fps = total_frames / total_time
    ap, n_gt, _ = mean_average_precision_person(list_gt, list_preds, iou_match=MAP_IOU)

    print(f"FPS: {fps:.4f}")
    print(f"device={device_tag}")
    print(f"Model size (on disk): {size_str}")
    if n_gt == 0:
        print(f"mAP@IoU={MAP_IOU:.2f} : undefined (zero person boxes in GT labels)")
    elif np.isnan(ap):
        print(f"mAP@IoU={MAP_IOU:.2f} : nan")
    else:
        print(f"mAP@IoU={MAP_IOU:.2f} : {ap:.6f}")
