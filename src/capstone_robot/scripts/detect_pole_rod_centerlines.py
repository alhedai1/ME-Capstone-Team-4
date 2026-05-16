#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from capstone_robot.utils import find_repo_root, rotate_frame
from capstone_robot.vision_centerlines import CenterlineConfig, process_frame


REPO_ROOT = find_repo_root(__file__)
DEFAULT_VIDEO_PATH = REPO_ROOT / "src/capstone_robot/data/videos/may15/trimmed/test1_trim.mp4"
DEFAULT_MODEL = REPO_ROOT / "src/capstone_robot/train/runs/detect/runs/upward_2/yolo11n_upward_2_640/weights/best.pt"
IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Detect pole and rod centerlines with YOLO-guided OpenCV.")
    parser.add_argument("--path", type=Path, default=DEFAULT_VIDEO_PATH, help="input image or video path")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="trained YOLO pole/rod model path")
    parser.add_argument("--output", type=Path, help="optional annotated output image/video path")
    parser.add_argument("--rotate", choices=["none", "cw", "ccw", "180"], default="none")
    parser.add_argument("--no-display", action="store_true", help="process without opening preview windows")
    parser.add_argument("--show-edges", action="store_true", help="show the edge image used for line detection")
    parser.add_argument("--show-dark-mask", action="store_true", help="show the black-material support mask")
    parser.add_argument("--window-name", default="pole_rod_centerlines")

    yolo = parser.add_argument_group("YOLO")
    yolo.add_argument("--conf", type=float, default=0.5, help="YOLO confidence threshold")
    yolo.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    yolo.add_argument("--no-yolo", action="store_true", help="use full-frame OpenCV only")
    yolo.add_argument("--full-frame-fallback", action="store_true", help="run full-frame OpenCV if YOLO misses a class")
    yolo.add_argument("--roi-pad", type=float, default=0.25, help="fractional padding around YOLO boxes")

    cv_args = parser.add_argument_group("OpenCV refinement")
    cv_args.add_argument("--resize-width", type=int, default=0, help="resize frames to this width before processing")
    cv_args.add_argument("--blur", type=int, default=5, help="odd Gaussian blur kernel size; use 0 to disable")
    cv_args.add_argument("--dark-threshold", type=int, default=95, help="HSV value threshold for black pixels")
    cv_args.add_argument("--dark-max-saturation", type=int, default=120, help="max saturation for black-material support")
    cv_args.add_argument("--very-dark-threshold", type=int, default=55, help="value threshold accepted regardless of saturation")
    cv_args.add_argument("--pole-min-dark-support", type=float, default=0.10)
    cv_args.add_argument("--rod-min-dark-support", type=float, default=0.08)
    cv_args.add_argument("--canny-low", type=int, default=45)
    cv_args.add_argument("--canny-high", type=int, default=130)
    cv_args.add_argument("--hough-threshold", type=int, default=20)
    cv_args.add_argument("--min-line-length", type=int, default=25)
    cv_args.add_argument("--max-line-gap", type=int, default=20)
    cv_args.add_argument("--pole-min-length-ratio", type=float, default=0.12)
    cv_args.add_argument("--pole-angle-tolerance", type=float, default=35.0)
    cv_args.add_argument("--pole-min-width", type=float, default=10.0)
    cv_args.add_argument("--pole-max-width-ratio", type=float, default=0.28)
    cv_args.add_argument("--rod-min-length", type=int, default=20)
    cv_args.add_argument("--rod-search-radius-ratio", type=float, default=0.28)
    cv_args.add_argument("--rod-min-pole-angle", type=float, default=25.0)
    return parser.parse_args()


def config_from_args(args):
    return CenterlineConfig(
        conf=args.conf,
        imgsz=args.imgsz,
        roi_pad=args.roi_pad,
        full_frame_fallback=args.full_frame_fallback,
        resize_width=args.resize_width,
        blur=args.blur,
        dark_threshold=args.dark_threshold,
        dark_max_saturation=args.dark_max_saturation,
        very_dark_threshold=args.very_dark_threshold,
        pole_min_dark_support=args.pole_min_dark_support,
        rod_min_dark_support=args.rod_min_dark_support,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        hough_threshold=args.hough_threshold,
        min_line_length=args.min_line_length,
        max_line_gap=args.max_line_gap,
        pole_min_length_ratio=args.pole_min_length_ratio,
        pole_angle_tolerance=args.pole_angle_tolerance,
        pole_min_width=args.pole_min_width,
        pole_max_width_ratio=args.pole_max_width_ratio,
        rod_min_length=args.rod_min_length,
        rod_search_radius_ratio=args.rod_search_radius_ratio,
        rod_min_pole_angle=args.rod_min_pole_angle,
    )


def resolve_model_path(model_path):
    if model_path.exists():
        return model_path

    candidates = sorted(
        (REPO_ROOT / "src/capstone_robot").rglob("weights/best.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates and model_path == DEFAULT_MODEL:
        return candidates[0]

    raise SystemExit(f"Model does not exist: {model_path}")


def open_writer(output_path, fps, frame_shape):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame_shape[:2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video for writing: {output_path}")
    return writer


def annotate_frame(frame, args, model, config):
    frame = rotate_frame(frame, args.rotate)
    return process_frame(frame, model, config)


def show_debug_windows(args, annotated, edges, dark_mask, wait_ms):
    cv2.imshow(args.window_name, annotated)
    if args.show_edges:
        cv2.imshow(f"{args.window_name}_edges", edges)
    if args.show_dark_mask:
        cv2.imshow(f"{args.window_name}_dark_mask", dark_mask)
    key = cv2.waitKey(wait_ms) & 0xFF
    return key == ord("q") or key == 27


def process_image(path, args, model, config):
    frame = cv2.imread(str(path))
    if frame is None:
        raise SystemExit(f"Could not read image: {path}")

    annotated, edges, dark_mask = annotate_frame(frame, args, model, config)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.output), annotated)
        print(f"Saved annotated image to: {args.output}")

    if not args.no_display:
        show_debug_windows(args, annotated, edges, dark_mask, wait_ms=0)
        cv2.destroyAllWindows()


def process_video(path, args, model, config):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = None
    frame_count = 0
    start = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            annotated, edges, dark_mask = annotate_frame(frame, args, model, config)
            # cv2.imshow("annotated", annotated)
            # cv2.waitKey(0)
            # cv2.imshow("edges", edges)
            # cv2.waitKey(0)
            # cv2.imshow("dark_mask", dark_mask)
            # cv2.waitKey(0)
            # import sys
            # sys.exit()
            if args.output is not None:
                if writer is None:
                    writer = open_writer(args.output, fps, annotated.shape)
                writer.write(annotated)

            if not args.no_display and show_debug_windows(args, annotated, edges, dark_mask, wait_ms=100):
                break

            frame_count += 1
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"Saved annotated video to: {args.output}")
        if not args.no_display:
            cv2.destroyAllWindows()

    elapsed = time.time() - start
    print(f"Frames processed: {frame_count}")
    print(f"Average FPS: {frame_count / elapsed if elapsed > 0 else 0.0:.2f}")


def main():
    args = parse_args()
    if not args.path.exists():
        raise SystemExit(f"Input does not exist: {args.path}")

    config = config_from_args(args)
    model = None
    if not args.no_yolo:
        model_path = resolve_model_path(args.model)
        print(f"Using YOLO model: {model_path}")
        model = YOLO(str(model_path))

    if args.path.suffix.lower() in IMAGE_SUFFIXES:
        process_image(args.path, args, model, config)
    else:
        process_video(args.path, args, model, config)


if __name__ == "__main__":
    main()
