#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time

import cv2
import gc

import sys
# sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from capstone_robot.utils import *

REPO_ROOT = find_repo_root(__file__)
DEFAULT_MODEL = REPO_ROOT / "src/capstone_robot/models/pole_imx_new/network.rpk"
DEFAULT_LABELS = REPO_ROOT / "src/capstone_robot/models/pole_imx_new/labels.txt"


@dataclass
class Detection:
    label: str
    confidence: float
    box: tuple  # (x, y, w, h)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run YOLO11n IMX500 detection using Raspberry Pi AI Camera"
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="IMX500 .rpk model path")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS, help="labels.txt path")
    parser.add_argument("--width", type=int, default=640, help="camera output width")
    parser.add_argument("--height", type=int, default=480, help="camera output height")
    parser.add_argument("--fps", type=float, default=30, help="camera framerate")
    parser.add_argument("--conf", type=float, default=0.5, help="confidence threshold")
    parser.add_argument("--rotate", choices=["none", "cw", "ccw", "180"], default="none")
    parser.add_argument("--show", action="store_true", help="show local OpenCV preview window")
    parser.add_argument("--preview-port", type=int, default=1234, help="serve browser preview on this port; use 0 to disable")
    parser.add_argument("--preview-host", default="0.0.0.0", help="host/interface for browser preview")
    parser.add_argument("--preview-width", type=int, default=640, help="browser preview width")
    parser.add_argument("--jpeg-quality", type=int, default=75, help="MJPEG preview JPEG quality from 1 to 100")

    # These match the usual Raspberry Pi YOLOv8n/YOLO11n IMX500 object-detection examples.
    parser.add_argument("--bbox-normalization", action="store_true", default=True, help="normalize bbox coordinates")
    parser.add_argument("--no-bbox-normalization", dest="bbox_normalization", action="store_false")
    parser.add_argument("--bbox-order", choices=["xy", "yx"], default="xy", help="bbox output order")
    return parser.parse_args()


def resolve_existing_path(path, description):
    if path is None:
        return None
    if path.exists():
        return path
    raise SystemExit(f"{description} does not exist: {path}")


def load_labels(labels_path):
    if labels_path is None:
        return None
    if not labels_path.exists():
        print(f"Labels file not found, using numeric class IDs: {labels_path}")
        return None

    labels = []
    with labels_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(line)

    return labels if labels else None


def resize_preview(frame, preview_width):
    if preview_width <= 0 or frame.shape[1] == preview_width:
        return frame

    scale = preview_width / frame.shape[1]
    preview_height = int(frame.shape[0] * scale)
    return cv2.resize(frame, (preview_width, preview_height))


def draw_detections(frame, detections):
    for det in detections:
        x, y, w, h = det.box

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        text = f"{det.label} {det.confidence:.2f}"
        cv2.putText(
            frame,
            text,
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


def draw_status(frame, loop_fps, average_fps, frame_count, num_detections):
    cv2.putText(
        frame,
        f"loop FPS: {loop_fps:.1f}  avg FPS: {average_fps:.1f}  frame: {frame_count}  det: {num_detections}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def print_best_detection(detections, frame_width):
    if not detections:
        return

    best = max(detections, key=lambda d: d.confidence)
    x, y, w, h = best.box
    cx = x + w / 2.0
    image_center_x = frame_width / 2.0
    error_x = cx - image_center_x

    print(
        f"{best.label}: conf={best.confidence:.2f}, "
        f"box=({x},{y},{w},{h}), center_x={cx:.1f}, error_x={error_x:.1f}, "
        f"WIDTH RATIO: {w/frame_width}"
    )


def main():
    args = parse_args()

    model_path = resolve_existing_path(args.model, "RPK model")
    labels = load_labels(args.labels)

    print(f"Using RPK model: {model_path}")
    if labels:
        print(f"Using labels: {args.labels} -> {labels}")

    try:
        camera = AiCamera(
            model_path=model_path,
            width=args.width,
            height=args.height,
            fps=args.fps,
            bbox_normalization=args.bbox_normalization,
            bbox_order=args.bbox_order,
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc))

    preview_server = None
    if args.preview_port > 0:
        preview_server = MjpegPreview(args.preview_host, args.preview_port, args.jpeg_quality)
        preview_server.start()
        host, port = preview_server.address
        display_host = "localhost" if host == "0.0.0.0" else host
        print(f"Preview stream: http://{display_host}:{port}")

    last_detections = []
    last_detection_time = 0.0
    detection_hold_time = 0.0  # seconds

    frame_count = 0
    start_time = time.time()
    last_loop_time = start_time

    smoothed_box = None
    alpha = 0.3

    try:
        while True:
            ok, frame, metadata = camera.read()
            if not ok or frame is None or metadata is None:
                print("No frame/metadata received; stopping.")
                break

            frame = rotate_frame(frame, args.rotate)

            new_detections = camera.get_detections(
                metadata=metadata,
                labels=labels,
                threshold=args.conf,
            )

            now = time.time()

            if new_detections:
                last_detections = new_detections
                last_detection_time = now
            
            if now - last_detection_time <= detection_hold_time:
                detections = last_detections
            else:
                detections = []

            if len(detections) > 0:
                best = max(detections, key=lambda d: d.confidence)
                if smoothed_box is None:
                    smoothed_box = best.box
                else:
                    x, y, w, h = best.box
                    sx, sy, sw, sh = smoothed_box

                    smoothed_box = (
                        int(alpha * x + (1 - alpha) * sx),
                        int(alpha * y + (1 - alpha) * sy),
                        int(alpha * w + (1 - alpha) * sw),
                        int(alpha * h + (1 - alpha) * sh),
                    )

            # If you want to use this for control, this is the important part:
            # best.box gives x,y,w,h and its center tells you where the pole is.
            print_best_detection(detections, frame.shape[1])

            annotated = frame.copy()
            draw_detections(annotated, detections)

            now = time.time()
            loop_dt = now - last_loop_time
            last_loop_time = now

            elapsed = now - start_time
            loop_fps = 1.0 / loop_dt if loop_dt > 0 else 0.0
            average_fps = (frame_count + 1) / elapsed if elapsed > 0 else 0.0
            draw_status(annotated, loop_fps, average_fps, frame_count, len(detections))

            if preview_server is not None:
                preview_server.update(resize_preview(annotated, args.preview_width))

            if args.show:
                cv2.imshow("ai_camera_imx500_detection", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break

            frame_count += 1

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        if preview_server is not None:
            preview_server.stop()
        if args.show:
            cv2.destroyAllWindows()

        camera.release()
        gc.collect()
        time.sleep(2.0)

    elapsed = time.time() - start_time
    average_fps = frame_count / elapsed if elapsed > 0 else 0.0
    print(f"Frames processed: {frame_count}")
    print(f"Average loop FPS: {average_fps:.2f}")


if __name__ == "__main__":
    main()