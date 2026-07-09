"""
Evaluate one or more trained runs on the held-out test split and print/save
a comparison table (mAP50, mAP50-95, precision, recall per class, and
inference speed) -- the numbers used to justify the CBAM modification in
the project report.

Examples:
    # Evaluate a single run
    python evaluate.py --weights runs/detect/helmet-baseline-yolov8s/weights/best.pt \\
                        --data data/dataset/data.yaml

    # Compare baseline vs CBAM side by side
    python evaluate.py \\
        --weights runs/detect/helmet-baseline-yolov8s/weights/best.pt \\
                  runs/detect/helmet-cbam-yolov8s/weights/best.pt \\
        --names baseline cbam \\
        --data data/dataset/data.yaml \\
        --out docs/results_comparison.csv
"""
import argparse
from pathlib import Path

import pandas as pd
from ultralytics import YOLO


def evaluate_one(weights: str, data: str, imgsz: int, split: str) -> dict:
    # CBAM checkpoints need the module registered before loading.
    from models import register_modules  # noqa: F401

    model = YOLO(weights)
    metrics = model.val(data=data, imgsz=imgsz, split=split, plots=True)

    names = metrics.names
    row = {
        "weights": weights,
        "mAP50": metrics.box.map50,
        "mAP50-95": metrics.box.map,
        "precision": metrics.box.mp,
        "recall": metrics.box.mr,
    }
    for cls_idx, cls_name in names.items():
        if cls_idx < len(metrics.box.ap50):
            row[f"AP50_{cls_name}"] = metrics.box.ap50[cls_idx]

    speed = metrics.speed  # dict: preprocess/inference/postprocess ms per image
    total_ms = sum(speed.values())
    row["inference_ms_per_image"] = total_ms
    row["fps"] = 1000.0 / total_ms if total_ms > 0 else float("nan")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", nargs="+", required=True, help="One or more .pt checkpoints")
    parser.add_argument("--names", nargs="+", default=None, help="Friendly labels, one per --weights entry")
    parser.add_argument("--data", type=str, default="data/dataset/data.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--out", type=str, default="docs/results_comparison.csv")
    args = parser.parse_args()

    labels = args.names or [Path(w).parent.parent.name for w in args.weights]
    if len(labels) != len(args.weights):
        raise ValueError("--names must have the same length as --weights")

    # See train.py for why this must be absolute (Ultralytics' relative-path
    # fallback resolves against its DATASETS_DIR setting, not the yaml's folder).
    data_path = str(Path(args.data).resolve())

    rows = []
    for label, weights in zip(labels, args.weights):
        print(f"\n=== Evaluating '{label}' ({weights}) on '{args.split}' split ===")
        row = evaluate_one(weights, data_path, args.imgsz, args.split)
        row["run"] = label
        rows.append(row)

    df = pd.DataFrame(rows).set_index("run")
    cols = ["mAP50", "mAP50-95", "precision", "recall", "inference_ms_per_image", "fps"]
    cols = [c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols and c != "weights"]
    df = df[cols]

    print("\n=== Comparison ===")
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path)
    print(f"\nSaved comparison table to {out_path}")


if __name__ == "__main__":
    main()
