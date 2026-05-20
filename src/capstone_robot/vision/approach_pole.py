#!/usr/bin/env python3
import argparse
from pathlib import Path

from capstone_robot.utils import AiCamera, find_repo_root


REPO_ROOT = find_repo_root(__file__)
DEFAULT_MODEL = REPO_ROOT / "src/capstone_robot/models/pole_imx/network.rpk"
DEFAULT_LABELS = REPO_ROOT / "src/capstone_robot/models/pole_imx/labels.txt"


def load_labels(path):
    if path is None or not path.exists():
        return None
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Center robot on pole using Raspberry Pi AI Camera")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--deadband", type=float, default=30.0, help="center tolerance in pixels")
    parser.add_argument("--target-label", default="pole")
    parser.add_argument("--bbox-order", choices=["xy", "yx"], default="xy")
    parser.add_argument("--bbox-normalization", action="store_true", default=True)
    parser.add_argument("--no-bbox-normalization", dest="bbox_normalization", action="store_false")
    return parser.parse_args()


def choose_pole(detections, target_label):
    if target_label:
        poles = [det for det in detections if det.label.lower() == target_label.lower()]
        if poles:
            return max(poles, key=lambda det: det.confidence)

    return max(detections, key=lambda det: det.confidence) if detections else None


def steering_from_error(error_x, deadband):
    if error_x < -deadband:
        return "LEFT"
    if error_x > deadband:
        return "RIGHT"
    return "STRAIGHT"


def main():
    args = parse_args()
    labels = load_labels(args.labels)

    camera = AiCamera(
        model_path=args.model,
        width=args.width,
        height=args.height,
        fps=args.fps,
        bbox_normalization=args.bbox_normalization,
        bbox_order=args.bbox_order,
    )

    try:
        while True:
            ok, frame, metadata = camera.read()
            if not ok:
                print("No camera frame")
                break

            detections = camera.get_detections(metadata, labels=labels, threshold=args.conf)
            pole = choose_pole(detections, args.target_label)

            if pole is None:
                print("NO POLE")
                continue

            x, y, w, h = pole.box
            pole_center_x = x + w / 2.0
            frame_center_x = frame.shape[1] / 2.0
            error_x = pole_center_x - frame_center_x
            steering = steering_from_error(error_x, args.deadband)

            print(
                f"{steering}: error_x={error_x:.1f}px, "
                f"center_x={pole_center_x:.1f}, conf={pole.confidence:.2f}, box=({x},{y},{w},{h})"
            )

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        camera.release()


if __name__ == "__main__":
    main()
