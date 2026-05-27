import argparse
import csv
import json
import sys
from pathlib import Path

import cv2

sys.path.append(str(Path(__file__).resolve().parents[2]))

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"

from capstone_robot.vision.bell_trigger_features import (
    FEATURE_NAMES,
    BellFeatureConfig,
    RoiConfig,
    build_feature_masks,
    extract_bell_features,
    config_to_dict,
    image_paths,
)


def add_roi_args(parser):
    parser.add_argument("--roi-x", type=float, default=0.0, help="ROI left as fraction of image width")
    parser.add_argument("--roi-y", type=float, default=0.0, help="ROI top as fraction of image height")
    parser.add_argument("--roi-w", type=float, default=1.0, help="ROI width as fraction of image width")
    parser.add_argument("--roi-h", type=float, default=1.0, help="ROI height as fraction of image height")


def add_feature_args(parser):
    parser.add_argument("--warm-h-min", type=int, default=5)
    parser.add_argument("--warm-h-max", type=int, default=40)
    parser.add_argument("--warm-s-min", type=int, default=35)
    parser.add_argument("--warm-v-min", type=int, default=40)
    parser.add_argument("--bright-v-min", type=int, default=170)
    parser.add_argument("--bright-s-max", type=int, default=130)
    parser.add_argument("--dark-v-max", type=int, default=55)
    parser.add_argument("--high-s-min", type=int, default=120)
    parser.add_argument("--canny-low", type=int, default=60)
    parser.add_argument("--canny-high", type=int, default=140)
    parser.add_argument("--blur-kernel", type=int, default=5)
    parser.add_argument("--morph-kernel", type=int, default=9)


def config_from_args(args):
    return BellFeatureConfig(
        warm_h_min=args.warm_h_min,
        warm_h_max=args.warm_h_max,
        warm_s_min=args.warm_s_min,
        warm_v_min=args.warm_v_min,
        bright_v_min=args.bright_v_min,
        bright_s_max=args.bright_s_max,
        dark_v_max=args.dark_v_max,
        high_s_min=args.high_s_min,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        blur_kernel=args.blur_kernel,
        morph_kernel=args.morph_kernel,
    )


def roi_from_args(args):
    return RoiConfig(x=args.roi_x, y=args.roi_y, w=args.roi_w, h=args.roi_h)


def save_debug_masks(debug_dir, image_path, label, masks):
    rel_name = f"{label}_{image_path.stem}"
    label_dir = debug_dir / label
    label_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(label_dir / f"{rel_name}_warm.png"), masks["warm_mask"])
    cv2.imwrite(str(label_dir / f"{rel_name}_bright.png"), masks["bright_mask"])
    cv2.imwrite(str(label_dir / f"{rel_name}_edges.png"), masks["edge_mask"])
    cv2.imwrite(str(label_dir / f"{rel_name}_candidate.png"), masks["candidate_mask"])


def main():
    parser = argparse.ArgumentParser(description="Extract OpenCV bell trigger features from labeled frames.")
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATA_DIR / "datasets" / "bell_dataset",
        help="Folder containing strike/ and no_strike/",
    )
    parser.add_argument("--out", default=DEFAULT_DATA_DIR / "bell_features.csv", help="Output CSV path")
    parser.add_argument("--debug_dir", help="Optional folder for debug masks")
    add_roi_args(parser)
    add_feature_args(parser)
    args = parser.parse_args()

    dataset = Path(args.dataset)
    out_path = Path(args.out)
    debug_dir = Path(args.debug_dir) if args.debug_dir else None
    roi_config = roi_from_args(args)
    feature_config = config_from_args(args)

    rows = []
    for label_name, label_value in (("strike", 1), ("no_strike", 0)):
        label_dir = dataset / label_name
        if not label_dir.exists():
            raise FileNotFoundError(f"Missing label folder: {label_dir}")

        for image_path in image_paths(label_dir):
            frame = cv2.imread(str(image_path))
            if frame is None:
                print(f"SKIP {image_path}: could not read")
                continue

            features, masks = extract_bell_features(frame, roi_config=roi_config, config=feature_config)
            row = {"path": str(image_path), "label": label_value}
            row.update(features)
            rows.append(row)

            if debug_dir:
                save_debug_masks(debug_dir, image_path, label_name, masks)

    if not rows:
        raise RuntimeError(f"No readable images found under {dataset}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["path", "label"] + FEATURE_NAMES)
        writer.writeheader()
        writer.writerows(rows)

    metadata_path = out_path.with_suffix(".json")
    metadata = {
        "roi": config_to_dict(roi_config),
        "feature_config": config_to_dict(feature_config),
        "feature_names": FEATURE_NAMES,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    strike_count = sum(row["label"] == 1 for row in rows)
    print(f"Wrote {len(rows)} rows to {out_path}")
    print(f"Wrote metadata to {metadata_path}")
    print(f"strike={strike_count} no_strike={len(rows) - strike_count}")


if __name__ == "__main__":
    main()
