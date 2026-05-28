#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2

from capstone_robot.utils import find_repo_root, rotate_frame
from capstone_robot.vision.bell_circle_climb import BellCircle


REPO_ROOT = find_repo_root(__file__)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "src/capstone_robot/data/presentation_visuals"
DEFAULT_FRAME_DIR = REPO_ROOT / "src/capstone_robot/data/extracted_frames/may28/train_ai_bell2"


GREEN = (70, 230, 80)
RED = (60, 80, 255)
BLUE = (255, 120, 40)
YELLOW = (40, 220, 255)
WHITE = (245, 245, 245)
GRAY = (120, 120, 120)
DARK_BLUE = (130, 55, 0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Draw bell-circle tracking annotations on one image or one video frame."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="input image/video path, or a frame filename from --frame-dir",
    )
    parser.add_argument("--output", type=Path, default=None, help="annotated output image path")
    parser.add_argument(
        "--frame-dir",
        type=Path,
        default=DEFAULT_FRAME_DIR,
        help="directory used when input is only a frame filename",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help="video frame index to annotate; if omitted, a detected frame near the middle is used",
    )
    parser.add_argument("--rotate", choices=["none", "cw", "ccw", "180"], default="none")
    parser.add_argument("--no-show", action="store_true", help="save/print only; do not open OpenCV window")
    parser.add_argument("--min-radius", type=int, default=10)
    parser.add_argument("--startup-max-radius", type=int, default=50)
    parser.add_argument("--tracking-max-radius", type=int, default=130)
    parser.add_argument("--param1", type=int, default=50)
    parser.add_argument("--param2", type=int, default=50)
    parser.add_argument(
        "--startup-confirm-threshold",
        type=int,
        default=1,
        help="consistent startup frames required before reporting a new circle",
    )
    return parser.parse_args()


def resolve_input_path(input_path, frame_dir):
    if input_path.exists():
        return input_path

    if input_path.parent == Path("."):
        candidate = frame_dir / input_path.name
        if candidate.exists():
            return candidate

    raise SystemExit(
        f"Input does not exist: {input_path}\n"
        f"Tried frame directory: {frame_dir / input_path.name}"
    )


def put_label(frame, text, org, scale=1.0, color=WHITE, thickness=2):
    x, y = org
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_PLAIN, scale, color, thickness, cv2.LINE_AA)


def make_detector(args):
    return BellCircle(
        color_format="bgr",
        dp=1.5,
        min_dist=5,
        param1=args.param1,
        param2=args.param2,
        min_radius=args.min_radius,
        max_radius=args.startup_max_radius,
        startup_max_radius=args.startup_max_radius,
        tracking_max_radius=args.tracking_max_radius,
        lost_after_frames=8,
        startup_confirm_threshold=args.startup_confirm_threshold,
    )


def is_video(path):
    return path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def read_image(path, rotation):
    frame = cv2.imread(str(path))
    if frame is None:
        raise SystemExit(f"Could not read image: {path}")
    return rotate_frame(frame, rotation)


def read_video_frame(path, rotation, requested_frame, detector):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if requested_frame is None:
        requested_frame = max(0, frame_count // 2)

    selected_frame = None
    selected_detection = None
    selected_index = None
    fallback = None

    for frame_index in range(max(0, requested_frame) + 1):
        ok, frame = cap.read()
        if not ok:
            break
        frame = rotate_frame(frame, rotation)
        detection = detector.detect(frame)
        if detection is not None:
            fallback = (frame.copy(), detection, frame_index)
        if frame_index == requested_frame:
            selected_frame = frame
            selected_detection = detection
            selected_index = frame_index
            break

    cap.release()

    if selected_frame is None and fallback is not None:
        return fallback
    if selected_frame is None:
        raise SystemExit(f"Could not read frame {requested_frame} from video: {path}")
    if selected_detection is None and fallback is not None:
        return fallback
    return selected_frame, selected_detection, selected_index


def command_from_detection(detection):
    if detection is None:
        return "DESCEND", RED
    return "CLIMB", GREEN


def annotate(frame, detection, detector=None):
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    image_center_x = width / 2.0

    cv2.line(annotated, (int(image_center_x), 0), (int(image_center_x), height), GRAY, 1)
    put_label(annotated, "BELL TRACK", (24, 48), 1.5, DARK_BLUE)

    command, command_color = command_from_detection(detection)

    if detection is None:
        put_label(annotated, "NO BELL", (24, height - 72), 1.5, RED)
        put_label(annotated, command, (width - 160, height - 40), 1.7, command_color)
        return annotated, None

    x, y, radius = detection.circle
    error_x = x - image_center_x

    cv2.circle(annotated, (x, y), radius, BLUE, 2)
    cv2.circle(annotated, (x, y), 5, GREEN, -1)
    cv2.line(annotated, (int(image_center_x), y), (x, y), YELLOW, 2)
    cv2.line(annotated, (x, max(0, y - radius)), (x, min(height, y + radius)), BLUE, 1)

    put_label(annotated, f"r={radius}", (24, height - 72), 1.5, BLUE)
    put_label(annotated, f"ERROR: {error_x:+.0f} px", (24, height - 32), 1.5, YELLOW)
    put_label(annotated, command, (width - 150, height - 40), 1.7, command_color)

    info = {
        "x": x,
        "y": y,
        "radius": radius,
        "error_x": error_x,
        "command": command,
        "stable_frames": getattr(detector, "stable_frames", None),
        "missed_frames": getattr(detector, "missed_frames", None),
    }
    return annotated, info


def default_output_path(input_path):
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_OUTPUT_DIR / f"{input_path.stem}_bell_tracking_annotated.png"


def main():
    args = parse_args()

    input_path = resolve_input_path(args.input, args.frame_dir)

    detector = make_detector(args)

    if is_video(input_path):
        frame, detection, frame_index = read_video_frame(input_path, args.rotate, args.frame, detector)
    else:
        frame = read_image(input_path, args.rotate)
        detection = detector.detect(frame)
        frame_index = None

    annotated, info = annotate(frame, detection, detector)
    output = args.output if args.output is not None else default_output_path(input_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), annotated)

    if info is None:
        print("No bell circle detected")
    else:
        frame_text = "" if frame_index is None else f"frame={frame_index}, "
        print(
            f"{info['command']}: {frame_text}"
            f"x={info['x']}, y={info['y']}, r={info['radius']}, "
            f"error_x={info['error_x']:+.1f}px, "
            f"stable={info['stable_frames']}, missed={info['missed_frames']}"
        )
    print(f"Saved annotated image to: {output}")

    if not args.no_show:
        cv2.imshow("bell circle tracking annotation", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
