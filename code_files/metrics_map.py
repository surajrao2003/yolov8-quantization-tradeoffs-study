"""Person-class detection metrics from YOLO-format labels."""

from pathlib import Path

import numpy as np

from postprocessing import iou

PERSON_CLASS_ID = 0


def load_yolo_ground_truth(
    labels_dir: Path, stem: str, img_h: int, img_w: int, class_id: int = PERSON_CLASS_ID
) -> np.ndarray:
    """
    Load gt boxes from YOLO label file (<class cx cy w h> normalized).

    Returns (N, 4) float32 XYXY pixels; empty array if missing or unreadable.
    """
    path = Path(labels_dir) / f"{stem}.txt"
    if not path.is_file():
        return np.zeros((0, 4), dtype=np.float32)
    xyxy_list = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            if cls != class_id:
                continue
            cx, cy, bw, bh = map(float, parts[1:5])
            x1 = (cx - bw / 2.0) * img_w
            y1 = (cy - bh / 2.0) * img_h
            x2 = (cx + bw / 2.0) * img_w
            y2 = (cy + bh / 2.0) * img_h
            xyxy_list.append([x1, y1, x2, y2])
    if not xyxy_list:
        return np.zeros((0, 4), dtype=np.float32)
    return np.asarray(xyxy_list, dtype=np.float32)


def _voc_ap(rec: np.ndarray, prec: np.ndarray) -> float:
    """VOC-style AP with precision envelope (interpolated)."""
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def mean_average_precision_person(
    images_gts: list[np.ndarray],
    images_preds: list[tuple[np.ndarray, np.ndarray]],
    iou_match: float = 0.5,
) -> tuple[float, int, int]:
    """
    Person-class mAP (same as AP for one class): sort all predictions by score,
    greedy match to ground truth at ``iou_match`` (default 0.5, COCO AP50-style).

    Returns (ap, total_gt_count, total_pred_count).
    """
    n_img = len(images_gts)
    if n_img != len(images_preds):
        raise ValueError("Ground truth and prediction lists must align per image.")

    total_gt = int(sum(g.shape[0] for g in images_gts))
    if total_gt > 0 and not any(boxes.shape[0] > 0 for boxes, _s in images_preds):
        return 0.0, total_gt, 0

    all_entries: list[tuple[float, int, np.ndarray]] = []
    for img_idx, (boxes, scores) in enumerate(images_preds):
        if boxes.size == 0:
            continue
        for row in range(boxes.shape[0]):
            all_entries.append((float(scores[row]), img_idx, boxes[row].astype(np.float64)))

    all_entries.sort(key=lambda x: x[0], reverse=True)

    gt_matched: list[set[int]] = [set() for _ in range(n_img)]
    tp = np.zeros(len(all_entries), dtype=np.float64)
    fp = np.zeros(len(all_entries), dtype=np.float64)

    for i, (_score, img_idx, pred_box) in enumerate(all_entries):
        gts = images_gts[img_idx]
        if gts.shape[0] == 0:
            fp[i] = 1.0
            continue
        best_j = -1
        best_iou = 0.0
        for j in range(gts.shape[0]):
            if j in gt_matched[img_idx]:
                continue
            iou_v = float(iou(pred_box, gts[j]))
            if iou_v > best_iou:
                best_iou = iou_v
                best_j = j
        if best_j >= 0 and best_iou >= iou_match:
            tp[i] = 1.0
            gt_matched[img_idx].add(best_j)
        else:
            fp[i] = 1.0

    if total_gt > 0 and len(all_entries) == 0:
        return 0.0, total_gt, 0

    if total_gt == 0:
        return float("nan"), 0, len(all_entries)

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall = cum_tp / total_gt
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
    ap = _voc_ap(recall, precision)
    return ap, total_gt, len(all_entries)
