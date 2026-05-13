#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from capstone_robot.utils import *

REPO_ROOT = find_repo_root(__file__)
DEFAULT_MODEL = REPO_ROOT / "src/capstone_robot/train/runs/detect/runs/upward_2/yolo11n_upward_2_640/weights/best.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="Run a trained YOLO model on a video")
    parser.add_argument("video", type=Path, help="input video path")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="trained YOLO model path")
    parser.add_argument("--conf", type=float, default=0.4, help="confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--output", type=Path, default=None, help="optional annotated output video path")
    parser.add_argument("--rotate", choices=["none", "cw", "ccw", "180"], default="none")
    parser.add_argument("--window-name", default="YOLO detection")
    return parser.parse_args()


def rotate_frame(frame, rotation):
    if rotation == "cw":
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation == "ccw":
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotation == "180":
        return cv2.rotate(frame, cv2.ROTATE_180)
    return frame


def open_writer(output_path, fps, frame_shape):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame_shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video for writing: {output_path}")
    return writer


def draw_fps(frame, fps):
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def resolve_model_path(model_path):
    if model_path.exists():
        return model_path

    candidates = sorted(
        (REPO_ROOT / "runs").rglob("best.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates and model_path == DEFAULT_MODEL:
        return candidates[0]

    raise SystemExit(f"Model does not exist: {model_path}")


def main():
    args = parse_args()

    if not args.video.exists():
        raise SystemExit(f"Video does not exist: {args.video}")

    model_path = resolve_model_path(args.model)
    print(f"Using model: {model_path}")
    model = YOLO(str(model_path))
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    writer = None
    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            frame = rotate_frame(frame, args.rotate)
            infer_start = time.time()
            result = model(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
            infer_time = time.time() - infer_start

            annotated = result.plot()
            draw_fps(annotated, 1.0 / infer_time if infer_time > 0 else 0.0)

            if args.output is not None:
                if writer is None:
                    writer = open_writer(args.output, fps, annotated.shape)
                writer.write(annotated)

            cv2.imshow(args.window_name, annotated)
            cv2.waitKey(0)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

            frame_count += 1

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"Saved annotated video to: {args.output}")
        cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    avg_fps = frame_count / elapsed if elapsed > 0 else 0.0
    print(f"Frames processed: {frame_count}")
    print(f"Average FPS: {avg_fps:.2f}")


if __name__ == "__main__":
    main()
