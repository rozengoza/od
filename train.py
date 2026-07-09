"""
Train the one-stage helmet-compliance detector.

Two variants share this entry point:
  --variant baseline  -> stock YOLOv8 (transfer-learned from COCO weights)
  --variant cbam      -> YOLOv8 backbone/neck with CBAM attention inserted
                         before each detection head (models/yolov8s-cbam.yaml),
                         still transfer-learned from the matching COCO weights
                         wherever tensor shapes line up (backbone is identical).

Defaults are tuned for a free-tier Colab T4 GPU (yolov8s, 640px, batch 16).
Drop --model-size to "n" and/or lower --batch if you're on CPU or a smaller GPU.

Examples:
    python train.py --variant baseline --data data/dataset/data.yaml --epochs 60
    python train.py --variant cbam     --data data/dataset/data.yaml --epochs 60
"""
import argparse
import sys
from pathlib import Path

# Block Ultralytics' optional Weights & Biases integration before it's ever imported.
# We never asked for W&B logging, but Ultralytics auto-enables it whenever the wandb
# package happens to be installed (it's pre-bundled in Colab's default image), and it
# reuses our --project value as a wandb *project name*, which rejects "/" -- crashing
# on the absolute path we deliberately pass (see the data/project path-resolution
# fixes above/nearby). Poisoning sys.modules makes any `import wandb` fail cleanly,
# regardless of whether the package is actually installed.
sys.modules["wandb"] = None

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent


def build_model(variant: str, model_size: str) -> YOLO:
    if variant == "baseline":
        return YOLO(f"yolov8{model_size}.pt")

    if variant == "cbam":
        # Registering CBAM must happen before the YAML is parsed.
        from models import register_modules  # noqa: F401

        cbam_yaml = REPO_ROOT / "models" / f"yolov8{model_size}-cbam.yaml"
        if not cbam_yaml.exists():
            raise FileNotFoundError(
                f"{cbam_yaml} not found. models/yolov8s-cbam.yaml is provided; "
                f"copy it to yolov8{model_size}-cbam.yaml for other scales "
                "(the 'scales' block already supports n/s/m/l/x)."
            )
        model = YOLO(str(cbam_yaml))
        model.load(f"yolov8{model_size}.pt")  # transfer weights for matching layers
        return model

    raise ValueError(f"Unknown variant: {variant}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=["baseline", "cbam"], required=True)
    parser.add_argument("--data", type=str, default="data/dataset/data.yaml")
    parser.add_argument("--model-size", type=str, default="s", choices=["n", "s", "m", "l", "x"])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", type=str, default="")  # "" = let Ultralytics auto-pick
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", type=str, default="runs/detect")
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()

    run_name = args.name or f"helmet-{args.variant}-yolov8{args.model_size}"

    # Ultralytics only resolves relative train/val/test paths against the yaml's own
    # directory when the --data path itself is absolute; otherwise it falls back to
    # its DATASETS_DIR setting, which is wrong here. Same issue with --project: a
    # relative value gets joined onto Ultralytics' own RUNS_DIR setting instead of
    # the current directory, silently doubling the path (e.g. runs/detect/runs/detect).
    # Always pass absolute paths to sidestep both.
    data_path = str(Path(args.data).resolve())
    project_path = str(Path(args.project).resolve())

    model = build_model(args.variant, args.model_size)
    model.train(
        data=data_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device or None,
        workers=args.workers,
        seed=args.seed,
        project=project_path,
        name=run_name,
        pretrained=True,
        plots=True,
    )

    print(f"\nTraining finished. Best weights at: {project_path}/{run_name}/weights/best.pt")


if __name__ == "__main__":
    main()
