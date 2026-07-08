"""
Real-world demo: Automated Safety-Helmet Compliance Monitor.

Simulates the kind of camera feed a construction-site entry gate or a
traffic intersection would use: it runs the trained one-stage detector on
an uploaded image, an uploaded video, or a live webcam feed, draws
detections, and keeps a running violation log (timestamped snapshot +
CSV row) whenever a bare head (no helmet) is detected -- the artifact you
point to for the "real-world implementation" part of the evaluation.

Run with:
    streamlit run app/streamlit_app.py
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.utils import draw_detections, log_violations, LOG_FIELDS  # noqa: E402

SNAPSHOT_DIR = REPO_ROOT / "violations" / "snapshots"
LOG_CSV_PATH = REPO_ROOT / "violations" / "violations_log.csv"

st.set_page_config(page_title="Helmet Compliance Monitor", layout="wide")


@st.cache_resource(show_spinner="Loading detector...")
def load_model(weights_path: str):
    if "cbam" in Path(weights_path).stem.lower() or "cbam" in str(Path(weights_path).parent).lower():
        from models import register_modules  # noqa: F401
    from ultralytics import YOLO
    return YOLO(weights_path)


def discover_checkpoints():
    runs_dir = REPO_ROOT / "runs" / "detect"
    if not runs_dir.exists():
        return []
    return sorted(str(p) for p in runs_dir.glob("*/weights/best.pt"))


def sidebar_controls():
    st.sidebar.title("Settings")

    checkpoints = discover_checkpoints()
    default_path = checkpoints[0] if checkpoints else ""
    weights_path = st.sidebar.selectbox(
        "Trained weights", options=checkpoints or ["<none found, type path below>"],
        index=0,
    )
    manual_path = st.sidebar.text_input("...or enter a weights path manually", value="" if checkpoints else default_path)
    if manual_path:
        weights_path = manual_path

    conf_thres = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.35, 0.05)
    iou_thres = st.sidebar.slider("NMS IoU threshold", 0.1, 0.9, 0.45, 0.05)
    source = st.sidebar.radio("Input source", ["Image", "Video file", "Webcam"])

    return weights_path, conf_thres, iou_thres, source


def show_violation_log():
    st.subheader("Violation log")
    if LOG_CSV_PATH.exists():
        df = pd.read_csv(LOG_CSV_PATH)
        st.metric("Total logged violations", len(df))
        st.dataframe(df.sort_values("timestamp", ascending=False), use_container_width=True, height=250)
        with open(LOG_CSV_PATH, "rb") as f:
            st.download_button("Download full log (CSV)", f, file_name="violations_log.csv")
    else:
        st.info("No violations logged yet.")


def run_image_mode(model, conf_thres, iou_thres):
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        return

    image = Image.open(uploaded).convert("RGB")
    frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    results = model.predict(frame_bgr, conf=conf_thres, iou=iou_thres, verbose=False)
    annotated, violations = draw_detections(frame_bgr, results[0], conf_thres)

    col1, col2 = st.columns(2)
    col1.image(image, caption="Input", use_container_width=True)
    col2.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Detections", use_container_width=True)

    if violations:
        st.error(f"{len(violations)} helmet violation(s) detected.")
        log_violations(SNAPSHOT_DIR, LOG_CSV_PATH, annotated, violations, source_label=uploaded.name)
    else:
        st.success("No violations detected.")


def run_video_mode(model, conf_thres, iou_thres):
    uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded is None:
        return

    tmp_path = REPO_ROOT / "violations" / f"_tmp_{uploaded.name}"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_bytes(uploaded.read())

    cap = cv2.VideoCapture(str(tmp_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    frame_window = st.image([])
    progress = st.progress(0)
    violation_count = 0
    frame_idx = 0

    stop = st.button("Stop")
    while cap.isOpened() and not stop:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        results = model.predict(frame, conf=conf_thres, iou=iou_thres, verbose=False)
        annotated, violations = draw_detections(frame, results[0], conf_thres)
        frame_window.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

        if violations:
            violation_count += len(violations)
            log_violations(SNAPSHOT_DIR, LOG_CSV_PATH, annotated, violations, source_label=uploaded.name)

        progress.progress(min(frame_idx / total_frames, 1.0))

    cap.release()
    tmp_path.unlink(missing_ok=True)
    st.success(f"Processed {frame_idx} frames, logged {violation_count} violation(s).")


def run_webcam_mode(model, conf_thres, iou_thres):
    st.warning("Webcam mode reads from the local camera of the machine running this Streamlit process.")
    run = st.checkbox("Start webcam")
    frame_window = st.image([])

    if not run:
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open webcam (device 0).")
        return

    fps_placeholder = st.empty()
    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to read frame from webcam.")
            break

        t0 = time.time()
        results = model.predict(frame, conf=conf_thres, iou=iou_thres, verbose=False)
        annotated, violations = draw_detections(frame, results[0], conf_thres)
        dt = time.time() - t0

        frame_window.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        fps_placeholder.caption(f"~{1 / dt:.1f} FPS")

        if violations:
            log_violations(SNAPSHOT_DIR, LOG_CSV_PATH, annotated, violations, source_label="webcam")

    cap.release()


def main():
    st.title("Automated Safety-Helmet Compliance Monitor")
    st.caption(
        "One-stage detector (YOLOv8, optionally with a CBAM attention head) flags "
        "riders/workers without a helmet and keeps a timestamped violation log — "
        "a stand-in for a construction-site gate camera or a traffic enforcement camera."
    )

    weights_path, conf_thres, iou_thres, source = sidebar_controls()

    if not weights_path or not Path(weights_path).exists():
        st.warning("Select or enter a valid trained weights (.pt) path in the sidebar to begin.")
        return

    model = load_model(weights_path)

    if source == "Image":
        run_image_mode(model, conf_thres, iou_thres)
    elif source == "Video file":
        run_video_mode(model, conf_thres, iou_thres)
    else:
        run_webcam_mode(model, conf_thres, iou_thres)

    st.divider()
    show_violation_log()


if __name__ == "__main__":
    main()
