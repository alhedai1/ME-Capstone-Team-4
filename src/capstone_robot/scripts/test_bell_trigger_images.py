import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"

from capstone_robot.vision.bell_trigger_features import (
    FEATURE_NAMES,
    config_from_dict,
    draw_roi,
    extract_bell_features,
    feature_vector,
    image_paths,
    put_lines,
    roi_from_dict,
)


def load_payload(path):
    import joblib

    payload = joblib.load(path)
    if hasattr(payload, "predict"):
        return {"model": payload, "feature_names": FEATURE_NAMES, "roi": {}, "feature_config": {}}
    return payload


def important_feature_names(model, limit=5):
    importances = getattr(model, "feature_importances_", np.zeros(len(FEATURE_NAMES)))
    names = [name for name, importance in sorted(zip(FEATURE_NAMES, importances), key=lambda item: item[1], reverse=True) if importance > 0]
    return names[:limit] if names else FEATURE_NAMES[:limit]


def draw_prediction(frame, prediction, probability, features, masks, important_names):
    out = frame.copy()
    color = (0, 0, 255) if prediction else (0, 180, 0)
    draw_roi(out, masks["roi_box"], color=color, thickness=2)
    status = "STRIKE" if prediction else "no_strike"
    lines = [f"{status} p={probability:.2f}"]
    lines.extend(f"{name}={features[name]:.3f}" for name in important_names)
    put_lines(out, lines, color=color)
    return out


def main():
    parser = argparse.ArgumentParser(description="Run a trained bell trigger on a folder of images.")
    parser.add_argument("--model", default="models/bell_trigger.joblib")
    parser.add_argument("--images", required=True, help="Folder containing images")
    parser.add_argument("--out", default=DEFAULT_DATA_DIR / "debug_bell_trigger_images", help="Output folder for annotated images")
    parser.add_argument("--roi-x", type=float, help="Override model ROI x")
    parser.add_argument("--roi-y", type=float, help="Override model ROI y")
    parser.add_argument("--roi-w", type=float, help="Override model ROI w")
    parser.add_argument("--roi-h", type=float, help="Override model ROI h")
    args = parser.parse_args()

    payload = load_payload(args.model)
    model = payload["model"]
    roi_config = roi_from_dict(payload.get("roi", {}))
    for attr, value in (("x", args.roi_x), ("y", args.roi_y), ("w", args.roi_w), ("h", args.roi_h)):
        if value is not None:
            setattr(roi_config, attr, value)
    feature_config = config_from_dict(payload.get("feature_config", {}))
    important_names = important_feature_names(model)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths(args.images):
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"SKIP {image_path}: could not read")
            continue

        features, masks = extract_bell_features(frame, roi_config=roi_config, config=feature_config)
        x = np.array([feature_vector(features)], dtype=np.float32)
        prediction = int(model.predict(x)[0])
        probability = float(model.predict_proba(x)[0][1]) if hasattr(model, "predict_proba") else float(prediction)
        annotated = draw_prediction(frame, prediction, probability, features, masks, important_names)
        cv2.imwrite(str(out_dir / image_path.name), annotated)
        print(f"{image_path.name}: {'STRIKE' if prediction else 'no_strike'} p={probability:.2f}")


if __name__ == "__main__":
    main()
