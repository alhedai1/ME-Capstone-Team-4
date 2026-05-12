#!/usr/bin/env python3
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time

import cv2
from ultralytics import YOLO

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import *

REPO_ROOT = find_repo_root(__file__)

def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO detection on a live Raspberry Pi Camera feed")
    # parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="trained YOLO .pt or exported model path")
    parser.add_argument("--width", type=int, default=640, help="camera capture width")
    parser.add_argument("--height", type=int, default=480, help="camera capture height")
    parser.add_argument("--fps", type=float, default=15, help="camera framerate")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--conf", type=float, default=0.4, help="confidence threshold")
    parser.add_argument("--rotate", choices=["none", "cw", "ccw", "180"], default="none")
    parser.add_argument("--show", action="store_true", help="show local OpenCV preview window")
    parser.add_argument("--preview-port", type=int, default=1234, help="serve browser preview on this port; use 0 to disable")
    parser.add_argument("--preview-host", default="0.0.0.0", help="host/interface for browser preview")
    parser.add_argument("--preview-width", type=int, default=320, help="browser preview width")
    parser.add_argument("--jpeg-quality", type=int, default=50, help="MJPEG preview JPEG quality from 1 to 100")
    return parser.parse_args()


def main():
    args = parse_args()
    # model_path = resolve_model_path(args.model)
    # print(f"Using model: {model_path}")

    try:
        camera = PiCamera(args.width, args.height, args.fps)
    except RuntimeError as exc:
        raise SystemExit(str(exc))

    preview_server = None
    if args.preview_port > 0:
        preview_server = MjpegPreview(args.preview_host, args.preview_port, args.jpeg_quality)
        preview_server.start()
        host, port = preview_server.address
        display_host = "localhost" if host == "0.0.0.0" else host
        print(f"Preview stream: http://{display_host}:{port}")

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                print("No frame received; stopping.")
                break

            frame = cv2.resize(frame, (args.width, args.height))
            frame = rotate_frame(frame, args.rotate)

            infer_start = time.time()
            infer_time = time.time() - infer_start

            elapsed = time.time() - start_time
            inference_fps = 1.0 / infer_time if infer_time > 0 else 0.0
            average_fps = (frame_count + 1) / elapsed if elapsed > 0 else 0.0
            draw_status(frame, inference_fps, average_fps, frame_count)

            if preview_server is not None:
                preview_server.update(resize_preview(frame, args.preview_width))

            if args.show:
                cv2.imshow("picamera_yolo_detection", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break

            frame_count += 1

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        camera.release()
        if preview_server is not None:
            preview_server.stop()
        if args.show:
            cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    average_fps = frame_count / elapsed if elapsed > 0 else 0.0
    print(f"Frames processed: {frame_count}")
    print(f"Average FPS: {average_fps:.2f}")


if __name__ == "__main__":
    main()
