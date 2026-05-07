#!/usr/bin/env python3
import argparse
import random
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Split YOLO images/labels into train and val folders")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/roboflow/dataset"))
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 < args.train_ratio < 1.0:
        raise SystemExit("--train-ratio must be between 0.0 and 1.0")

    images_dir = args.dataset_dir / "images"
    labels_dir = args.dataset_dir / "labels"

    out_img_train = images_dir / "train"
    out_img_val = images_dir / "val"
    out_lbl_train = labels_dir / "train"
    out_lbl_val = labels_dir / "val"

    for directory in [out_img_train, out_img_val, out_lbl_train, out_lbl_val]:
        directory.mkdir(parents=True, exist_ok=True)

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    pairs = []
    for img_path in images_dir.iterdir():
        if img_path.is_file() and img_path.suffix.lower() in image_exts:
            label_path = labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                pairs.append((img_path, label_path))

    random.seed(args.seed)
    random.shuffle(pairs)

    n_train = int(len(pairs) * args.train_ratio)
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:]

    for img_path, label_path in train_pairs:
        shutil.move(str(img_path), str(out_img_train / img_path.name))
        shutil.move(str(label_path), str(out_lbl_train / label_path.name))

    for img_path, label_path in val_pairs:
        shutil.move(str(img_path), str(out_img_val / img_path.name))
        shutil.move(str(label_path), str(out_lbl_val / label_path.name))

    print(f"Train: {len(train_pairs)}")
    print(f"Val:   {len(val_pairs)}")


if __name__ == "__main__":
    main()
