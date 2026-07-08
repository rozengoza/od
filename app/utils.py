"""Shared drawing / violation-logging helpers used by the Streamlit demo
across its image, video, and webcam modes."""
import csv
from datetime import datetime
from pathlib import Path

import cv2

# Class indices must match data/prepare_dataset.py's CLASSES list.
CLASS_NAMES = {0: "helmet", 1: "head", 2: "person"}
VIOLATION_CLASS_NAME = "head"  # a bare head == rider/worker without a helmet

COLORS_BGR = {
    "helmet": (0, 200, 0),     # green = compliant
    "head": (0, 0, 255),       # red = violation
    "person": (255, 160, 0),   # blue = context only
}

LOG_FIELDS = ["timestamp", "source", "class", "confidence", "x1", "y1", "x2", "y2", "snapshot_path"]


def draw_detections(frame_bgr, result, conf_thres: float):
    """Draws boxes on a copy of frame_bgr and returns (annotated_frame, violations).

    violations is a list of dicts with class/confidence/bbox for every
    detection of VIOLATION_CLASS_NAME above conf_thres.
    """
    annotated = frame_bgr.copy()
    violations = []

    boxes = result.boxes
    if boxes is None:
        return annotated, violations

    for box in boxes:
        conf = float(box.conf[0])
        if conf < conf_thres:
            continue
        cls_id = int(box.cls[0])
        cls_name = CLASS_NAMES.get(cls_id, str(cls_id))
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        color = COLORS_BGR.get(cls_name, (200, 200, 200))

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if cls_name == VIOLATION_CLASS_NAME:
            violations.append({
                "class": cls_name, "confidence": conf,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            })

    return annotated, violations


def log_violations(snapshot_dir: Path, log_csv_path: Path, annotated_frame_bgr, violations, source_label: str):
    """Appends one CSV row per violation and saves a single annotated snapshot.
    Returns the list of rows written (for immediate display in the UI)."""
    if not violations:
        return []

    snapshot_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    snapshot_path = snapshot_dir / f"violation_{timestamp}.jpg"
    cv2.imwrite(str(snapshot_path), annotated_frame_bgr)

    write_header = not log_csv_path.exists()
    log_csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(log_csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if write_header:
            writer.writeheader()
        for v in violations:
            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "source": source_label,
                "class": v["class"],
                "confidence": f"{v['confidence']:.4f}",
                "x1": v["x1"], "y1": v["y1"], "x2": v["x2"], "y2": v["y2"],
                "snapshot_path": str(snapshot_path),
            }
            writer.writerow(row)
            rows.append(row)
    return rows
