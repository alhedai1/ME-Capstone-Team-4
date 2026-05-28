#!/usr/bin/env python3
import argparse
import contextlib
import io
from pathlib import Path

import cv2

from capstone_robot.utils import find_repo_root, rotate_frame
from capstone_robot.vision.bell_circle_climb import BellCircle


REPO_ROOT = find_repo_root(__file__)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "src/capstone_robot/data/presentation_visuals/bell_tracking_sequence"


GREEN = (70, 230, 80)
RED = (60, 80, 255)
BLUE = (255, 120, 40)
YELLOW = (40, 220, 255)
WHITE = (245, 245, 245)
BLACK = (20, 20, 20)
GRAY = (120, 120, 120)
DARK_BLUE = (130, 55, 0)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Scan a climbing video and save presentation images for bell visible, "
            "bell disappeared, and bell reacquired."
        )
    )
    parser.add_argument("video", type=Path, help="input climbing video")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rotate", choices=["none", "cw", "ccw", "180"], default="none")
    parser.add_argument("--save-raw", action="store_true", help="also save unannotated frames")
    parser.add_argument("--min-stable-frames", type=int, default=3)
    parser.add_argument("--sample-step", type=int, default=1, help="process every Nth frame")
    parser.add_argument("--min-radius", type=int, default=10)
    parser.add_argument("--startup-max-radius", type=int, default=50)
    parser.add_argument("--tracking-max-radius", type=int, default=130)
    parser.add_argument("--param1", type=int, default=50)
    parser.add_argument("--param2", type=int, default=50)
    return parser.parse_args()


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
        startup_confirm_threshold=2,
    )


def put_label(frame, text, org, scale=1.0, color=WHITE, thickness=2):
    x, y = org
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_PLAIN, scale, color, thickness, cv2.LINE_AA)


def put_box_label(frame, text, org, scale=1.25, color=WHITE, bg=BLACK):
    x, y = org
    (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, scale, 2)
    cv2.rectangle(frame, (x - 8, y - h - 8), (x + w + 8, y + baseline + 8), bg, -1)
    put_label(frame, text, (x, y), scale, color, 2)


def draw_detection(frame, detection):
    if detection is None:
        return
    x, y, radius = detection.circle
    cv2.circle(frame, (x, y), radius, BLUE, 2)
    cv2.circle(frame, (x, y), 5, GREEN, -1)
    cv2.line(frame, (x, max(0, y - radius)), (x, min(frame.shape[0], y + radius)), BLUE, 1)


def annotate(frame, detection, title, command, color, frame_index, stable_frames, missed_frames):
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    center_x = width // 2

    cv2.line(annotated, (center_x, 0), (center_x, height), GRAY, 1)
    put_box_label(annotated, title, (24, 42), 1.45, WHITE, DARK_BLUE)

    if detection is None:
        x, y = width // 2, height // 2
        cv2.line(annotated, (x - 45, y - 45), (x + 45, y + 45), RED, 5)
        cv2.line(annotated, (x + 45, y - 45), (x - 45, y + 45), RED, 5)
        put_box_label(annotated, "NO CIRCLE DETECTED", (24, height - 92), 1.35, RED)
    else:
        draw_detection(annotated, detection)
        x, y, radius = detection.circle
        error_x = x - center_x
        cv2.line(annotated, (center_x, y), (x, y), YELLOW, 2)
        put_box_label(annotated, f"x={x} y={y} r={radius}", (24, height - 112), 1.25, BLUE)
        put_box_label(annotated, f"center error={error_x:+.0f}px", (24, height - 72), 1.25, YELLOW)

    put_box_label(annotated, command, (24, height - 32), 1.15, color)
    put_box_label(
        annotated,
        f"frame={frame_index} stable={stable_frames} missed={missed_frames}",
        (width - 360, 42),
        1.05,
        WHITE,
        BLACK,
    )
    return annotated


def save_frame(output_dir, name, frame, annotated, save_raw):
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = output_dir / f"{name}.png"
    cv2.imwrite(str(annotated_path), annotated)
    raw_path = None
    if save_raw:
        raw_path = output_dir / f"{name}_raw.png"
        cv2.imwrite(str(raw_path), frame)
    return annotated_path, raw_path


def better_visible(current, candidate):
    if current is None:
        return candidate
    return candidate["detection"].radius < current["detection"].radius


def better_reacquired(current, candidate):
    if current is None:
        return candidate
    return candidate["detection"].radius > current["detection"].radius


def scan_video(args):
    if not args.video.exists():
        raise SystemExit(f"Video does not exist: {args.video}")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    detector = make_detector(args)
    sample_step = max(1, args.sample_step)

    visible = None
    disappeared = None
    reacquired = None
    seen_visible = False
    seen_disappeared = False

    frame_index = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_index += 1
        if frame_index % sample_step != 0:
            continue

        frame = rotate_frame(frame, args.rotate)
        with contextlib.redirect_stdout(io.StringIO()):
            detection = detector.detect(frame)

        stable = detector.stable_frames >= args.min_stable_frames
        actively_detected = detection is not None and detector.missed_frames == 0 and stable
        fully_lost = seen_visible and detection is None

        if actively_detected and not seen_disappeared:
            candidate = {
                "frame": frame.copy(),
                "detection": detection,
                "index": frame_index,
                "stable_frames": detector.stable_frames,
                "missed_frames": detector.missed_frames,
            }
            visible = candidate if better_visible(visible, candidate) else visible
            seen_visible = True

        if fully_lost and disappeared is None:
            disappeared = {
                "frame": frame.copy(),
                "detection": None,
                "index": frame_index,
                "stable_frames": detector.stable_frames,
                "missed_frames": detector.missed_frames,
            }
            seen_disappeared = True

        if seen_disappeared and actively_detected:
            candidate = {
                "frame": frame.copy(),
                "detection": detection,
                "index": frame_index,
                "stable_frames": detector.stable_frames,
                "missed_frames": detector.missed_frames,
            }
            reacquired = candidate if better_reacquired(reacquired, candidate) else reacquired

    cap.release()
    return detector, visible, disappeared, reacquired


def main():
    args = parse_args()
    detector, visible, disappeared, reacquired = scan_video(args)

    outputs = []
    if visible is not None:
        annotated = annotate(
            visible["frame"],
            visible["detection"],
            "1. BELL VISIBLE",
            "CLIMB / STRIKE WINDOW",
            GREEN,
            visible["index"],
            visible["stable_frames"],
            visible["missed_frames"],
        )
        outputs.append(save_frame(args.output_dir, "01_bell_visible", visible["frame"], annotated, args.save_raw))

    if disappeared is not None:
        annotated = annotate(
            disappeared["frame"],
            None,
            "2. BELL DISAPPEARED",
            "STOP MOTORS / SLIP DOWN",
            RED,
            disappeared["index"],
            disappeared["stable_frames"],
            disappeared["missed_frames"],
        )
        outputs.append(
            save_frame(args.output_dir, "02_bell_disappeared", disappeared["frame"], annotated, args.save_raw)
        )

    if reacquired is not None:
        annotated = annotate(
            reacquired["frame"],
            reacquired["detection"],
            "3. BELL REACQUIRED",
            "WAIT 3 SECONDS / STRIKE AGAIN",
            YELLOW,
            reacquired["index"],
            reacquired["stable_frames"],
            reacquired["missed_frames"],
        )
        outputs.append(save_frame(args.output_dir, "03_bell_reacquired", reacquired["frame"], annotated, args.save_raw))

    if not outputs:
        raise SystemExit("No usable bell-tracking frames found. Try different video or Hough parameters.")

    print(f"Saved bell tracking sequence to: {args.output_dir}")
    if visible is None:
        print("Warning: bell visible frame not found")
    if disappeared is None:
        print("Warning: bell disappeared frame not found")
    if reacquired is None:
        print("Warning: bell reacquired frame not found")
    for annotated_path, raw_path in outputs:
        print(f"- {annotated_path}")
        if raw_path is not None:
            print(f"- {raw_path}")


if __name__ == "__main__":
    main()
