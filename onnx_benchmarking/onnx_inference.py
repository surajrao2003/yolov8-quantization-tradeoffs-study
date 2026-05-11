"""
ONNX Runtime inference backend used by onnx_benchmarking/main_execution.py.
Imports shared preprocessing/postprocessing/metrics code from code_files/.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

_REPO = Path(__file__).resolve().parent.parent
_CODE_FILES = _REPO / "code_files"
if str(_CODE_FILES) not in sys.path:
    sys.path.insert(0, str(_CODE_FILES))

from metrics_map import load_yolo_ground_truth, mean_average_precision_person  # noqa: E402
from postprocessing import display_people_count_patch, postprocess_and_log_outputs  # noqa: E402
from preprocessing import preprocess_frame  # noqa: E402

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
PERSON_CLASS_ID = 0
MAP_IOU = 0.5


def _ort_input_feed_dtype(inp: ort.NodeArg) -> np.dtype:
    t = inp.type.lower()
    if "float16" in t:
        return np.dtype(np.float16)
    return np.dtype(np.float32)


_GPU_OR_CUDA_SETUP = (
    "GPU requested but CUDA is not active. Install a CUDA-enabled ORT build and compatible CUDA libraries. "
    "For pip use onnxruntime-gpu; on some installs you must `module load cuda`/`module load cudnn` first."
)


def initialize_model(model_path: str, device: str) -> ort.InferenceSession:
    dev = (device or "").lower()
    if dev == "gpu":
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
        sess = ort.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        if "CUDAExecutionProvider" not in sess.get_providers():
            raise RuntimeError(_GPU_OR_CUDA_SETUP)
        return sess

    if dev == "cpu":
        return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    raise ValueError("device must be 'cpu' or 'gpu'")


def infer_boxes(
    model: ort.InferenceSession,
    frame_bgr: np.ndarray,
    input_size: tuple[int, int],
    conf: float,
    iou_nms: float,
    *,
    feed_dtype: np.dtype | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    if feed_dtype is None:
        feed_dtype = _ort_input_feed_dtype(model.get_inputs()[0])

    pre, orig, orig_shape, scale, pad = preprocess_frame(frame_bgr, input_size, feed_dtype=feed_dtype)
    outputs = model.run(None, {"images": pre})
    _img, filtered = postprocess_and_log_outputs(
        orig,
        outputs,
        orig_shape,
        scale,
        pad,
        conf_threshold=conf,
        iou_threshold=iou_nms,
        person_class_id=PERSON_CLASS_ID,
    )

    if not filtered:
        return np.zeros((0, 4)), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64), orig_shape

    boxes = np.asarray([d[:4] for d in filtered], dtype=np.float32)
    scores = np.asarray([d[4] for d in filtered], dtype=np.float32)
    class_ids = np.asarray([d[5] for d in filtered], dtype=np.int64)
    return boxes, scores, class_ids, orig_shape


def draw_predictions(frame_bgr: np.ndarray, boxes, scores, class_ids) -> np.ndarray:
    result = frame_bgr.copy()
    if boxes is None or len(boxes) == 0:
        display_people_count_patch(result, 0)
        return result

    count = 0
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        score = float(scores[i])
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        _draw_text(result, f"P: {score:.2f}", (x1, y1 - 30))
        count += 1
    display_people_count_patch(result, count)
    return result


def _draw_text(frame, text, position):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, thickness=2)
    x, y = position
    cv2.rectangle(frame, (x, y - th - 10), (x + tw, y + 10), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


def format_model_disk_size(model_path: str) -> str:
    p = Path(model_path)
    n = p.stat().st_size
    mib = n / (1024**2)
    return f"{mib:.2f} MiB ({n:,} bytes)"


def reported_device_from_session(model: ort.InferenceSession) -> str:
    primary = model.get_providers()[0]
    return "cpu" if primary == "CPUExecutionProvider" else "gpu"


def run_onnx_benchmark(
    model: ort.InferenceSession,
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    input_size: tuple[int, int],
    conf: float,
    iou_nms: float,
    *,
    model_path: str,
) -> None:
    labels_path = Path(labels_dir)
    total_time = 0.0
    total_frames = 0

    files = sorted(f for f in os.listdir(images_dir) if f.lower().endswith(IMAGE_EXTENSIONS))
    feed_dtype = _ort_input_feed_dtype(model.get_inputs()[0])

    gts: list[np.ndarray] = []
    preds: list[tuple[np.ndarray, np.ndarray]] = []

    for image_file in files:
        image_path = os.path.join(images_dir, image_file)
        stem = Path(image_file).stem
        t0 = time.time()
        frame = cv2.imread(image_path)
        if frame is None:
            continue

        boxes, scores, class_ids, (h, w) = infer_boxes(
            model,
            frame,
            input_size,
            conf,
            iou_nms,
            feed_dtype=feed_dtype,
        )
        out_img = draw_predictions(frame, boxes, scores, class_ids)
        cv2.imwrite(os.path.join(output_dir, f"{stem}.jpg"), out_img)

        total_time += time.time() - t0
        total_frames += 1
        gts.append(load_yolo_ground_truth(labels_path, stem, h, w, PERSON_CLASS_ID))
        preds.append((boxes, scores))

    size_str = format_model_disk_size(model_path)
    device_tag = reported_device_from_session(model)

    if total_frames == 0:
        print("FPS: N/A (no images processed)")
        print(f"device={device_tag}")
        print(f"Model size (on disk): {size_str}")
        print(f"mAP@IoU={MAP_IOU:.2f} : N/A (no images processed)")
        return

    fps = total_frames / total_time
    ap, n_gt, _ = mean_average_precision_person(gts, preds, iou_match=MAP_IOU)
    print(f"FPS: {fps:.4f}")
    print(f"device={device_tag}")
    print(f"Model size (on disk): {size_str}")
    if n_gt == 0:
        print(f"mAP@IoU={MAP_IOU:.2f} : undefined (zero person boxes in GT labels)")
    elif np.isnan(ap):
        print(f"mAP@IoU={MAP_IOU:.2f} : nan")
    else:
        print(f"mAP@IoU={MAP_IOU:.2f} : {ap:.6f}")

