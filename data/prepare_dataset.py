"""
Downloads the Kaggle "Hard Hat Detection" dataset (andrewmvd/hard-hat-detection)
and converts it from Pascal-VOC XML annotations into YOLO-format labels, split
into train/val/test folders ready for Ultralytics training.

Classes:
    0: helmet
    1: head        (person's head, no helmet -> the violation class)
    2: person

Usage:
    python data/prepare_dataset.py --out data/dataset --val-frac 0.1 --test-frac 0.1

Requires a Kaggle account (kagglehub will prompt for API credentials on first
run, or read ~/.kaggle/kaggle.json / KAGGLE_USERNAME+KAGGLE_KEY env vars).

If kagglehub can't reach api.kaggle.com (e.g. a local SSL-intercepting
antivirus/proxy breaks the download), download the dataset zip manually
from https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection, extract
it, and pass its path via --raw-dir to skip the network download entirely:

    python data/prepare_dataset.py --raw-dir "C:/Users/you/Downloads/hard-hat-detection" --out data/dataset
"""
import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from tqdm import tqdm

CLASSES = ["helmet", "head", "person"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}


def download_raw_dataset() -> Path:
    import kagglehub

    path = kagglehub.dataset_download("andrewmvd/hard-hat-detection")
    return Path(path)


def voc_to_yolo_line(obj, img_w, img_h):
    name = obj.find("name").text.strip().lower()
    if name not in CLASS_TO_ID:
        return None
    cls_id = CLASS_TO_ID[name]

    bbox = obj.find("bndbox")
    xmin = float(bbox.find("xmin").text)
    ymin = float(bbox.find("ymin").text)
    xmax = float(bbox.find("xmax").text)
    ymax = float(bbox.find("ymax").text)

    xmin, xmax = sorted((max(0.0, xmin), min(img_w, xmax)))
    ymin, ymax = sorted((max(0.0, ymin), min(img_h, ymax)))
    if xmax <= xmin or ymax <= ymin:
        return None

    cx = (xmin + xmax) / 2.0 / img_w
    cy = (ymin + ymax) / 2.0 / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    return f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def find_images_and_annotations(raw_root: Path):
    images_dir = next(p for p in raw_root.rglob("images") if p.is_dir())
    annots_dir = next(p for p in raw_root.rglob("annotations") if p.is_dir())
    xml_files = sorted(annots_dir.glob("*.xml"))
    pairs = []
    for xml_path in xml_files:
        stem = xml_path.stem
        img_path = None
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = images_dir / f"{stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        if img_path is not None:
            pairs.append((img_path, xml_path))
    return pairs


def convert_and_split(pairs, out_dir: Path, val_frac: float, test_frac: float, seed: int):
    random.Random(seed).shuffle(pairs)
    n = len(pairs)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    splits = {
        "val": pairs[:n_val],
        "test": pairs[n_val:n_val + n_test],
        "train": pairs[n_val + n_test:],
    }

    for split, split_pairs in splits.items():
        img_out = out_dir / split / "images"
        lbl_out = out_dir / split / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path, xml_path in tqdm(split_pairs, desc=f"Converting {split}"):
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find("size")
            img_w = float(size.find("width").text)
            img_h = float(size.find("height").text)

            lines = []
            for obj in root.findall("object"):
                line = voc_to_yolo_line(obj, img_w, img_h)
                if line:
                    lines.append(line)

            shutil.copy2(img_path, img_out / img_path.name)
            (lbl_out / f"{img_path.stem}.txt").write_text("\n".join(lines), encoding="utf-8")

        print(f"{split}: {len(split_pairs)} images")

    return {k: len(v) for k, v in splits.items()}


def write_data_yaml(out_dir: Path):
    yaml_content = (
        f"path: {out_dir.resolve().as_posix()}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        "names:\n"
        + "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES))
        + "\n"
    )
    (out_dir / "data.yaml").write_text(yaml_content, encoding="utf-8")
    print(f"Wrote {out_dir / 'data.yaml'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default="data/dataset")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--raw-dir", type=str, default=None,
        help="Path to an already-downloaded+extracted copy of the dataset "
             "(must contain images/ and annotations/ folders somewhere under "
             "it). Skips the kagglehub network download entirely.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.raw_dir:
        raw_root = Path(args.raw_dir)
        if not raw_root.exists():
            raise FileNotFoundError(f"--raw-dir does not exist: {raw_root}")
        print(f"Using local raw dataset at: {raw_root}")
    else:
        print("Downloading raw dataset from Kaggle (andrewmvd/hard-hat-detection)...")
        raw_root = download_raw_dataset()
        print(f"Raw dataset at: {raw_root}")

    pairs = find_images_and_annotations(raw_root)
    print(f"Found {len(pairs)} image/annotation pairs")
    if not pairs:
        raise RuntimeError("No image/annotation pairs found - check dataset download.")

    convert_and_split(pairs, out_dir, args.val_frac, args.test_frac, args.seed)
    write_data_yaml(out_dir)
    print("Done. Dataset ready at:", out_dir.resolve())


if __name__ == "__main__":
    main()
