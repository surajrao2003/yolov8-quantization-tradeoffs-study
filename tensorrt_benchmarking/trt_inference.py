"""Benchmark loop for TensorRT engines (uses TrtSession in trt_runtime.py)."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_REPO = Path(__file__).resolve().parent.parent
_CODE_FILES = _REPO / "code_files"


def _ensure_code_files_path() -> None:
    if str(_CODE_FILES) not in sys.path:
        sys.path.insert(0, str(_CODE_FILES))


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
PERSON_CLASS_ID = 0
MAP_IOU = 0.5


def infer_boxes_trt(
    session,
    frame_bgr: np.ndarray,
    input_size: tuple[int, int],
    conf: float,
    iou_nms: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    _ensure_code_files_path()
    from postprocessing import postprocess_and_log_outputs  # noqa: WPS433
    from preprocessing import preprocess_frame  # noqa: WPS433

    preprocessed_frame, original_frame, original_frame_shape, scale, pad = preprocess_frame(
        frame_bgr, input_size, feed_dtype=session.input.dtype
    )
    preds = session.infer(preprocessed_frame)

    outputs = preds
    if session.input.dtype != np.dtype(np.float32):
        outputs = [o.astype(np.float32, copy=False) for o in preds]

    _img_u, filtered_detections = postprocess_and_log_outputs(
        original_frame,
        outputs,
        original_frame_shape,
        scale,
        pad,
        conf_threshold=conf,
        iou_threshold=iou_nms,
        person_class_id=PERSON_CLASS_ID,
    )
    hw = original_frame_shape
    if not filtered_detections:
        return np.zeros((0, 4)), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64), hw
    boxes = np.asarray([det[:4] for det in filtered_detections], dtype=np.float32)
    scores = np.asarray([det[4] for det in filtered_detections], dtype=np.float32)
    class_ids = np.asarray([det[5] for det in filtered_detections], dtype=np.int64)
    return boxes, scores, class_ids, hw


def run_trt_benchmark(
    session,
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    input_size: tuple[int, int],
    conf: float,
    iou_nms: float,
    *,
    engine_path: str,
    device_line: str = "device=gpu",
) -> None:
    _ensure_code_files_path()

    from inference import draw_predictions, format_model_disk_size  # noqa: WPS433
    from metrics_map import load_yolo_ground_truth, mean_average_precision_person  # noqa: WPS433

    labels_path = Path(labels_dir)
    total_time = 0.0
    total_frames = 0

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

        boxes, scores, class_ids, (h, w) = infer_boxes_trt(
            session,
            input_frame,
            input_size,
            conf,
            iou_nms,
        )
        result_image = draw_predictions(input_frame, boxes, scores, class_ids)

        total_time += time.time() - t0
        total_frames += 1

        list_gt.append(load_yolo_ground_truth(labels_path, filename_no_ext, h, w, PERSON_CLASS_ID))
        list_preds.append((boxes, scores))

        out_path = os.path.join(output_dir, f"{filename_no_ext}.jpg")
        cv2.imwrite(out_path, result_image)

    size_str = format_model_disk_size(engine_path)

    if total_frames == 0:
        print("FPS: N/A (no images processed)")
        print(device_line)
        print(f"Model size (on disk): {size_str}")
        print(f"mAP@IoU={MAP_IOU:.2f} : N/A (no images processed)")
        return

    fps = total_frames / total_time
    ap, n_gt, _ = mean_average_precision_person(list_gt, list_preds, iou_match=MAP_IOU)

    print(f"FPS: {fps:.4f}")
    print(device_line)
    print(f"Model size (on disk): {size_str}")
    if n_gt == 0:
        print(f"mAP@IoU={MAP_IOU:.2f} : undefined (zero person boxes in GT labels)")
    elif np.isnan(ap):
        print(f"mAP@IoU={MAP_IOU:.2f} : nan")
    else:
        print(f"mAP@IoU={MAP_IOU:.2f} : {ap:.6f}")
