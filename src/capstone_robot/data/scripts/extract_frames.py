#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2

from capstone_robot.utils import find_repo_root

REPO_ROOT = find_repo_root(__file__)
DEFAULT_INPUT_DIR = REPO_ROOT / "src/capstone_robot/data/videos/may15/trimmed"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "src/capstone_robot/data/extracted_frames/may15"
DEFAULT_FRAME_STEP = 15
VIDEO_EXTENSIONS = {".avi", ".mov", ".mp4", ".m4v", ".mkv", ".webm"}
# VIDEO_ROTATIONS = {
#     "20260504_181046": "cw",
#     "20260504_181112": "cw",
# }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract one frame every N frames from all videos in a folder."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--frame-step", type=int, default=DEFAULT_FRAME_STEP)
    parser.add_argument("--image-ext", choices=["jpg", "png"], default="jpg")
    parser.add_argument(
        "--default-rotate",
        choices=["none", "cw", "ccw", "180"],
        default="none",
        help="rotation to apply to videos that are not listed in VIDEO_ROTATIONS",
    )
    return parser.parse_args()


def rotate_frame(frame, rotate):
    if rotate == "cw":
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotate == "ccw":
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotate == "180":
        return cv2.rotate(frame, cv2.ROTATE_180)
    return frame


def extract_video_frames(video_path, output_dir, frame_step, image_ext, default_rotate):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Skipping {video_path}: could not open video")
        return 0

    video_output_dir = output_dir / video_path.stem
    video_output_dir.mkdir(parents=True, exist_ok=True)

    # rotate = VIDEO_ROTATIONS.get(video_path.stem, default_rotate)
    # print(f"{video_path.name}: rotation={rotate}")

    saved_count = 0
    frame_index = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        if frame_index % frame_step == 0:
            # frame = rotate_frame(frame, rotate)
            output_path = video_output_dir / f"frame_{frame_index:06d}.{image_ext}"
            if not cv2.imwrite(str(output_path), frame):
                print(f"Failed to save frame: {output_path}")
            else:
                saved_count += 1

        frame_index += 1

    cap.release()
    print(f"{video_path.name}: saved {saved_count} frames")
    return saved_count


def main():
    args = parse_args()

    if args.frame_step <= 0:
        raise SystemExit("--frame-step must be greater than 0")

    if not args.input_dir.is_dir():
        raise SystemExit(f"Input folder does not exist: {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(
        path
        for path in args.input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )

    if not videos:
        raise SystemExit(f"No videos found in: {args.input_dir}")

    total_saved = 0
    for video_path in videos:
        total_saved += extract_video_frames(
            video_path=video_path,
            output_dir=args.output_dir,
            frame_step=args.frame_step,
            image_ext=args.image_ext,
            default_rotate=args.default_rotate,
        )

    print(f"Done. Saved {total_saved} frames to: {args.output_dir}")


if __name__ == "__main__":
    main()
