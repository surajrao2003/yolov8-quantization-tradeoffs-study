"""
Benchmark loop for TFLite models (FPS, file size, person mAP@0.5).
Use tflite_benchmarking/main_execution.py to run.
"""
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


def _as_yolo_outputs(raw: list[np.ndarray]) -> list[np.ndarray]:
    """
    Convert common TFLite YOLO output layouts into the shape expected by code_files/postprocessing.py:
      outputs[0][0] is (84, 8400) so that `.transpose()` becomes (8400, 84).
    """
    if not raw:
        raise RuntimeError("No outputs from TFLite interpreter.")
    y = raw[0]
    y = np.asarray(y)
    if y.ndim != 3 or y.shape[0] != 1:
        raise RuntimeError(f"Unexpected output shape: {y.shape} (expected [1,*,*])")

    a, b = int(y.shape[1]), int(y.shape[2])
    # Heuristic: YOLOv8 detect head is typically 84 x 8400 (for 640), or 84 x N generally.
    if a == 84:
        return [y]
    if b == 84:
        return [np.transpose(y, (0, 2, 1))]
    return [y]


def _is_end2end_nms_output(y: np.ndarray) -> bool:
    # Common Ultralytics end2end output: [1, max_det, 6] (xyxy, score, cls)
    return y.ndim == 3 and y.shape[0] == 1 and y.shape[-1] == 6


def _infer_boxes_from_end2end(
    y: np.ndarray,
    *,
    original_img_shape: tuple[int, int],
    letterbox_hw: tuple[int, int],
    scale: tuple[float, float],
    pad: tuple[float, float],
    conf_threshold: float,
    iou_threshold: float,
    person_class_id: int,
) -> list[list[float]]:
    _ensure_code_files_path()
    from postprocessing import non_max_suppression, scale_boxes  # noqa: WPS433

    det = np.asarray(y[0], dtype=np.float32)
    if det.size == 0:
        return []

    boxes = det[:, 0:4].copy()
    scores = det[:, 4]
    class_ids = det[:, 5].astype(np.int64)
    # Ultralytics TFLite (nms=True) returns xyxy in [0,1] relative to the square input.
    lb_h, lb_w = int(letterbox_hw[0]), int(letterbox_hw[1])
    if boxes.size and float(np.max(boxes)) <= 1.5:
        boxes *= np.array([lb_w, lb_h, lb_w, lb_h], dtype=np.float32)

    keep = (class_ids == int(person_class_id)) & (scores >= float(conf_threshold))
    boxes = boxes[keep]
    scores = scores[keep]
    if boxes.size == 0:
        return []

    nms_boxes, nms_scores = non_max_suppression(boxes, scores, float(iou_threshold))
    if len(nms_boxes) == 0:
        return []

    new_unpad_shape = (
        int(original_img_shape[0] * scale[0]),
        int(original_img_shape[1] * scale[1]),
    )
    new_shape_with_pad = (
        new_unpad_shape[0] + 2 * int(pad[1]),
        new_unpad_shape[1] + 2 * int(pad[0]),
    )
    scaled = scale_boxes(
        np.asarray(nms_boxes, dtype=np.float32),
        new_shape_with_pad,
        original_img_shape,
        scale,
        pad,
    )
    return [[*b.tolist(), float(s), int(person_class_id)] for b, s in zip(scaled, nms_scores)]


def infer_boxes_tflite(session, frame_bgr: np.ndarray, input_size: tuple[int, int], conf: float, iou_nms: float):
    _ensure_code_files_path()
    from preprocessing import preprocess_frame  # noqa: WPS433

    pre, orig, orig_shape, scale, pad = preprocess_frame(frame_bgr, input_size, feed_dtype=np.float32)

    raw = session.infer(pre)
    y0 = np.asarray(raw[0])
    if _is_end2end_nms_output(y0):
        filtered = _infer_boxes_from_end2end(
            y0,
            original_img_shape=orig_shape,
            letterbox_hw=input_size,
            scale=scale,
            pad=pad,
            conf_threshold=conf,
            iou_threshold=iou_nms,
            person_class_id=PERSON_CLASS_ID,
        )
    else:
        from postprocessing import postprocess_and_log_outputs  # noqa: WPS433

        outputs = _as_yolo_outputs(raw)
        _img_u, filtered = postprocess_and_log_outputs(
            orig,
            outputs,
            orig_shape,
            scale,
            pad,
            conf_threshold=conf,
            iou_threshold=iou_nms,
            person_class_id=PERSON_CLASS_ID,
        )
    hw = orig_shape
    if not filtered:
        return np.zeros((0, 4)), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64), hw
    boxes = np.asarray([d[:4] for d in filtered], dtype=np.float32)
    scores = np.asarray([d[4] for d in filtered], dtype=np.float32)
    class_ids = np.asarray([d[5] for d in filtered], dtype=np.int64)
    return boxes, scores, class_ids, hw


def run_tflite_benchmark(
    session,
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    input_size: tuple[int, int],
    conf: float,
    iou_nms: float,
    *,
    model_path: str,
) -> None:
    _ensure_code_files_path()
    from inference import draw_predictions, format_model_disk_size  # noqa: WPS433
    from metrics_map import load_yolo_ground_truth, mean_average_precision_person  # noqa: WPS433

    labels_path = Path(labels_dir)
    total_time = 0.0
    total_frames = 0

    files = sorted(f for f in os.listdir(images_dir) if f.lower().endswith(IMAGE_EXTENSIONS))

    list_gt: list[np.ndarray] = []
    list_preds: list[tuple[np.ndarray, np.ndarray]] = []

    for image_file in files:
        image_path = os.path.join(images_dir, image_file)
        filename_no_ext = Path(image_file).stem
        t0 = time.time()
        frame = cv2.imread(image_path)
        if frame is None:
            continue

        boxes, scores, class_ids, (h, w) = infer_boxes_tflite(session, frame, input_size, conf, iou_nms)
        out_img = draw_predictions(frame, boxes, scores, class_ids)

        total_time += time.time() - t0
        total_frames += 1

        list_gt.append(load_yolo_ground_truth(labels_path, filename_no_ext, h, w, PERSON_CLASS_ID))
        list_preds.append((boxes, scores))

        cv2.imwrite(os.path.join(output_dir, f"{filename_no_ext}.jpg"), out_img)

    size_str = format_model_disk_size(model_path)
    if total_frames == 0:
        print("FPS: N/A (no images processed)")
        print(f"device={getattr(session, 'device', 'cpu')}")
        print(f"Model size (on disk): {size_str}")
        print(f"mAP@IoU={MAP_IOU:.2f} : N/A (no images processed)")
        return

    fps = total_frames / total_time
    ap, n_gt, _ = mean_average_precision_person(list_gt, list_preds, iou_match=MAP_IOU)
    print(f"FPS: {fps:.4f}")
    print(f"device={getattr(session, 'device', 'cpu')}")
    print(f"Model size (on disk): {size_str}")
    if n_gt == 0:
        print(f"mAP@IoU={MAP_IOU:.2f} : undefined (zero person boxes in GT labels)")
    elif np.isnan(ap):
        print(f"mAP@IoU={MAP_IOU:.2f} : nan")
    else:
        print(f"mAP@IoU={MAP_IOU:.2f} : {ap:.6f}")

