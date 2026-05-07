#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Extract frames from a video file")
    parser.add_argument("video", type=Path, help="input video path")
    parser.add_argument("--output-dir", type=Path, default=Path("data/extracted_frames"))
    parser.add_argument("--frame-skip", type=int, default=15)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.frame_skip <= 0:
        raise SystemExit("--frame-skip must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    frame_count = 0
    saved_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % args.frame_skip == 0:
            filename = f"{args.video.stem}_{saved_count:05d}.jpg"
            cv2.imwrite(str(args.output_dir / filename), frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Saved {saved_count} images to {args.output_dir}")


if __name__ == "__main__":
    main()
