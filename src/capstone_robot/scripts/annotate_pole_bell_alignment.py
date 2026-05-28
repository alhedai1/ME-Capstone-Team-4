#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2

from capstone_robot.utils import find_repo_root, rotate_frame
from capstone_robot.vision.pole_bell2 import PoleBellTracker, line_x_at_y


REPO_ROOT = find_repo_root(__file__)
DEFAULT_FRAME_DIR = REPO_ROOT / "src/capstone_robot/data/extracted_frames/may25/may25_align"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "src/capstone_robot/data/presentation_visuals"


GREEN = (70, 230, 80)
RED = (60, 80, 255)
BLUE = (255, 120, 40)
YELLOW = (40, 220, 255)
WHITE = (245, 245, 245)
GRAY = (120, 120, 120)
DARK_BLUE = (130, 55, 0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Draw concise pole-bell alignment annotations on one image or video frame."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="input image/video path, or a frame filename from --frame-dir",
    )
    parser.add_argument("--output", type=Path, default=None, help="annotated output image path")
    parser.add_argument("--frame-dir", type=Path, default=DEFAULT_FRAME_DIR)
    parser.add_argument("--frame", type=int, default=None, help="video frame index to annotate")
    parser.add_argument("--rotate", choices=["none", "cw", "ccw", "180"], default="none")
    parser.add_argument("--color-format", choices=["bgr", "rgb"], default="bgr")
    parser.add_argument(
        "--aligned-threshold",
        type=float,
        default=30.0,
        help="absolute error in pixels treated as aligned",
    )
    parser.add_argument("--no-show", action="store_true")
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


def is_video(path):
    return path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def put_label(frame, text, org, scale=1.0, color=WHITE, thickness=2):
    x, y = org
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_PLAIN, scale, color, thickness, cv2.LINE_AA)


def draw_line(frame, line, color=GREEN, thickness=3):
    vx, vy, x0, y0 = line
    t = 1000
    x1 = int(x0 - vx * t)
    y1 = int(y0 - vy * t)
    x2 = int(x0 + vx * t)
    y2 = int(y0 + vy * t)
    cv2.line(frame, (x1, y1), (x2, y2), color, thickness)


def read_image(path, rotation):
    frame = cv2.imread(str(path))
    if frame is None:
        raise SystemExit(f"Could not read image: {path}")
    return rotate_frame(frame, rotation)


def read_video_frame(path, rotation, requested_frame, tracker):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if requested_frame is None:
        requested_frame = max(0, frame_count // 2)

    selected = None
    fallback = None
    for frame_index in range(max(0, requested_frame) + 1):
        ok, frame = cap.read()
        if not ok:
            break

        frame = rotate_frame(frame, rotation)
        alignment = tracker.detect(frame)
        if alignment is not None:
            fallback = (frame.copy(), alignment, frame_index)
        if frame_index == requested_frame:
            selected = (frame, alignment, frame_index)
            break

    cap.release()

    if selected is None and fallback is not None:
        return fallback
    if selected is None:
        raise SystemExit(f"Could not read frame {requested_frame} from video: {path}")
    if selected[1] is None and fallback is not None:
        return fallback
    return selected


def alignment_status(alignment, threshold):
    if alignment is None:
        return "NO ALIGN", RED
    if abs(alignment.error_px) <= threshold:
        return "ALIGNED", GREEN
    if alignment.side == "left":
        return "LEFT", YELLOW
    return "RIGHT", YELLOW


def annotate(frame, alignment, threshold):
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    image_center_x = width / 2.0

    cv2.line(annotated, (int(image_center_x), 0), (int(image_center_x), height), GRAY, 1)
    put_label(annotated, "POLE-BELL", (24, 48), 1.5, DARK_BLUE)

    status, status_color = alignment_status(alignment, threshold)
    if alignment is None:
        put_label(annotated, status, (width - 190, height - 40), 1.7, status_color)
        return annotated, None

    draw_line(annotated, alignment.pole_line, GREEN, 3)
    bx, by, br = alignment.bell
    cv2.circle(annotated, (bx, by), br, BLUE, 2)
    cv2.circle(annotated, (bx, by), 5, RED, -1)

    pole_x = line_x_at_y(alignment.pole_line, by)
    if pole_x is not None:
        pole_point = (int(round(pole_x)), by)
        cv2.circle(annotated, pole_point, 4, GREEN, -1)
        cv2.line(annotated, pole_point, (bx, by), YELLOW, 2)

    put_label(annotated, f"ERROR: {alignment.error_px:+.0f} px", (24, height - 32), 1.5, YELLOW)
    put_label(annotated, status, (width - 190, height - 40), 1.7, status_color)

    info = {
        "error_px": alignment.error_px,
        "side": alignment.side,
        "status": status,
        "bell": alignment.bell,
    }
    return annotated, info


def default_output_path(input_path):
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source = input_path.parent.name
    return DEFAULT_OUTPUT_DIR / f"{source}_{input_path.stem}_pole_bell_alignment_annotated.png"


def main():
    args = parse_args()
    input_path = resolve_input_path(args.input, args.frame_dir)
    tracker = PoleBellTracker(color_format=args.color_format)

    if is_video(input_path):
        frame, alignment, frame_index = read_video_frame(input_path, args.rotate, args.frame, tracker)
    else:
        frame = read_image(input_path, args.rotate)
        alignment = tracker.detect(frame)
        frame_index = None

    annotated, info = annotate(frame, alignment, args.aligned_threshold)
    output = args.output if args.output is not None else default_output_path(input_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), annotated)

    if info is None:
        print("No pole-bell alignment detected")
    else:
        frame_text = "" if frame_index is None else f"frame={frame_index}, "
        print(
            f"{info['status']}: {frame_text}"
            f"side={info['side']}, error={info['error_px']:+.1f}px, bell={info['bell']}"
        )
    print(f"Saved annotated image to: {output}")

    if not args.no_show:
        cv2.imshow("pole-bell alignment annotation", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
