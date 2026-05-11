import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from metrics_map import load_yolo_ground_truth, mean_average_precision_person
from postprocessing import display_people_count_patch, postprocess_and_log_outputs
from preprocessing import preprocess_frame

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
PERSON_CLASS_ID = 0
MAP_IOU = 0.5  # COCO AP50-style matching threshold (separate from NMS IoU)


def _ort_input_feed_dtype(inp: ort.NodeArg) -> np.dtype:
    """Map ORT declared input type string to NumPy dtype; default float32 (FP32 models)."""
    t = inp.type.lower()
    if "float16" in t:
        return np.dtype(np.float16)
    return np.dtype(np.float32)


_GPU_OR_CUDA_SETUP = (
    "GPU requested but CUDA is not active. For pip installs use "
    "onnxruntime-gpu[cuda,cudnn]; DLLs live under site-packages/nvidia/ (not on PATH). "
    "This process calls onnxruntime.preload_dlls() before the session. Otherwise install "
    "CUDA 12.x (toolkit or conda) and cuDNN 9.x. See "
    "https://onnxruntime.ai/docs/execution-providers/CUDA-Execution-Provider.html"
)


def initialize_model(model_path, device):
    dev = (device or "").lower()
    if dev == "gpu":
        # Pip CUDA/cuDNN wheels live under site-packages/nvidia/* — use preload_dlls() before session.
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
        model = ort.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        if "CUDAExecutionProvider" not in model.get_providers():
            raise RuntimeError(_GPU_OR_CUDA_SETUP)
    elif dev == "cpu":
        model = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    else:
        print("Invalid device type specified. Choose from options 'cpu','CPU','gpu','GPU'.")
        sys.exit(1)

    return model


def infer_boxes(
    model,
    frame_bgr: np.ndarray,
    input_size: tuple[int, int],
    conf: float,
    iou_nms: float,
    *,
    feed_dtype: np.dtype | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """
    Forward pass only: person boxes XYXY pixels, scores, class IDs, (H,W) shape.
    """
    if feed_dtype is None:
        feed_dtype = _ort_input_feed_dtype(model.get_inputs()[0])
    preprocessed_frame, original_frame, original_frame_shape, scale, pad = preprocess_frame(
        frame_bgr, input_size, feed_dtype=feed_dtype
    )
    outputs = model.run(None, {"images": preprocessed_frame})
    _img, filtered_detections = postprocess_and_log_outputs(
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
        return np.zeros((0, 4)), np.zeros((0,), dtype=np.float32), np.zeros(
            (0,), dtype=np.int64
        ), hw
    boxes = np.array([det[:4] for det in filtered_detections])
    scores = np.array([det[4] for det in filtered_detections], dtype=np.float32)
    class_ids = np.array([det[5] for det in filtered_detections], dtype=np.int64)
    return boxes, scores, class_ids, hw


def draw_predictions(frame_bgr: np.ndarray, boxes, scores, class_ids) -> np.ndarray:
    result_frame = frame_bgr.copy()
    people_count = 0
    if boxes is None or len(boxes) == 0:
        display_people_count_patch(result_frame, people_count)
        return result_frame
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        score = float(scores[i])
        cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        draw_text(
            result_frame,
            f"P: {score:.2f}",
            (x1, y1 - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            (0, 0, 0),
        )
        people_count += 1
    display_people_count_patch(result_frame, people_count)
    return result_frame


def draw_text(frame, text, position, font, font_scale, text_color, bg_color):
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness=2)
    x, y = position
    cv2.rectangle(
        frame, (x, y - text_height - 10), (x + text_width, y + 10), bg_color, -1
    )
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, 2)


def format_model_disk_size(model_path: str) -> str:
    p = Path(model_path)
    n = p.stat().st_size
    mib = n / (1024**2)
    return f"{mib:.2f} MiB ({n:,} bytes)"


def reported_device_from_session(model: ort.InferenceSession) -> str:
    """
    Summarize where ONNX Runtime runs the graph using the session's provider order.
    Primary provider index 0: CPUExecutionProvider -> cpu; any other (e.g. CUDA) -> gpu.
    Per-node CPU fallback within a CUDA session is not distinguished here.
    """
    primary = model.get_providers()[0]
    return "cpu" if primary == "CPUExecutionProvider" else "gpu"


def run_image_inference(
    model,
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

    files = sorted(
        f
        for f in os.listdir(images_dir)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    )

    feed_dtype = _ort_input_feed_dtype(model.get_inputs()[0])

    list_gt: list[np.ndarray] = []
    list_preds: list[tuple[np.ndarray, np.ndarray]] = []

    for image_file in files:
        image_path = os.path.join(images_dir, image_file)
        filename_no_ext = Path(image_file).stem
        t0 = time.time()
        input_frame = cv2.imread(image_path)
        if input_frame is None:
            continue
        boxes, scores, class_ids, (h, w) = infer_boxes(
            model,
            input_frame,
            input_size,
            conf,
            iou_nms,
            feed_dtype=feed_dtype,
        )
        result_image = draw_predictions(input_frame, boxes, scores, class_ids)

        inference_time = time.time() - t0
        total_time += inference_time
        total_frames += 1

        list_gt.append(
            load_yolo_ground_truth(labels_path, filename_no_ext, h, w, PERSON_CLASS_ID)
        )
        list_preds.append((boxes, scores))

        out_path = os.path.join(output_dir, f"{filename_no_ext}.jpg")
        cv2.imwrite(out_path, result_image)

    # --- Printed summary (requested outputs only) ---
    size_str = format_model_disk_size(model_path)

    device_tag = reported_device_from_session(model)

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
        print(
            f"mAP@IoU={MAP_IOU:.2f} : undefined (zero person boxes in GT labels)"
        )
    elif np.isnan(ap):
        print(f"mAP@IoU={MAP_IOU:.2f} : nan")
    else:
        print(f"mAP@IoU={MAP_IOU:.2f} : {ap:.6f}")
