import argparse
import sys
from collections import deque
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
    put_lines,
    roi_from_dict,
)
from capstone_robot.scripts.test_bell_trigger_images import important_feature_names


def main():
    parser = argparse.ArgumentParser(description="Run a trained bell trigger on video with 3-of-5 temporal filtering.")
    parser.add_argument("--model", default="models/bell_trigger.joblib")
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", default=DEFAULT_DATA_DIR / "debug_bell_trigger_video.mp4")
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--min-hits", type=int, default=3)
    args = parser.parse_args()

    import joblib

    payload = joblib.load(args.model)
    if hasattr(payload, "predict"):
        payload = {"model": payload, "feature_names": FEATURE_NAMES, "roi": {}, "feature_config": {}}
    model = payload["model"]
    roi_config = roi_from_dict(payload.get("roi", {}))
    feature_config = config_from_dict(payload.get("feature_config", {}))
    important_names = important_feature_names(model)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    history = deque(maxlen=args.window)
    frame_count = 0
    trigger_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        features, masks = extract_bell_features(frame, roi_config=roi_config, config=feature_config)
        x = np.array([feature_vector(features)], dtype=np.float32)
        raw_prediction = int(model.predict(x)[0])
        probability = float(model.predict_proba(x)[0][1]) if hasattr(model, "predict_proba") else float(raw_prediction)
        history.append(raw_prediction)
        triggered = sum(history) >= args.min_hits
        trigger_count += int(triggered)

        out = frame.copy()
        color = (0, 0, 255) if triggered else (0, 180, 0)
        draw_roi(out, masks["roi_box"], color=color, thickness=2)
        lines = [
            f"{'TRIGGER' if triggered else 'hold'} raw={raw_prediction} p={probability:.2f}",
            f"hits={sum(history)}/{len(history)}",
        ]
        lines.extend(f"{name}={features[name]:.3f}" for name in important_names[:4])
        put_lines(out, lines, color=color)
        writer.write(out)
        frame_count += 1

    cap.release()
    writer.release()
    print(f"Wrote {frame_count} frames to {out_path}")
    print(f"Temporal trigger frames: {trigger_count}/{frame_count}")


if __name__ == "__main__":
    main()
