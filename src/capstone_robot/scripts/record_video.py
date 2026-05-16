#!/usr/bin/env python3
import argparse
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading

import cv2

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import *

REPO_ROOT = find_repo_root(__file__)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "src/capstone_robot/data/test_videos"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "recording.mp4"


def output_path_from_name(name):
    output = Path(name)
    if output.suffix == "":
        output = output.with_suffix(".mp4")
    if output.parent == Path("."):
        output = DEFAULT_OUTPUT_DIR / output
    return output


def parse_args():
    parser = argparse.ArgumentParser(description="Record video from the Raspberry Pi Camera")
    parser.add_argument(
        "name",
        nargs="?",
        help="output filename saved under data/test_videos; .mp4 is added if omitted",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="full output video path; overrides positional filename",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--rotate", choices=["none", "cw", "ccw", "180"], default="none")
    parser.add_argument("--show", action="store_true", help="show preview window")
    parser.add_argument("--preview-port", type=int, default=1234, help="serve browser preview on this port")
    parser.add_argument("--preview-host", default="0.0.0.0", help="host/interface for browser preview")
    parser.add_argument("--preview-width", type=int, default=640, help="browser preview width")
    args = parser.parse_args()
    if args.output is None:
        args.output = output_path_from_name(args.name) if args.name else DEFAULT_OUTPUT
    return args


def writer_size(width, height, rotation):
    if rotation in {"cw", "ccw"}:
        return height, width
    return width, height


def draw_overlay(frame, frame_number):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    text = f"{timestamp}  frame={frame_number}"
    cv2.putText(
        frame,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def make_preview_frame(frame, frame_number, preview_width):
    preview = frame.copy()
    if preview_width > 0 and preview.shape[1] != preview_width:
        scale = preview_width / preview.shape[1]
        preview_height = int(preview.shape[0] * scale)
        preview = cv2.resize(preview, (preview_width, preview_height))
    draw_overlay(preview, frame_number)
    return preview


def main():
    args = parse_args()

    try:
        camera = PiCamera(args.width, args.height, args.fps)
    except RuntimeError as exc:
        raise SystemExit(str(exc))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_width, out_height = writer_size(args.width, args.height, args.rotate)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.output), fourcc, args.fps, (out_width, out_height))
    if not writer.isOpened():
        camera.release()
        raise SystemExit(f"Could not open output video for writing: {args.output}")

    preview_server = None
    if args.preview_port is not None:
        preview_server = MjpegPreview(args.preview_host, args.preview_port)
        preview_server.start()
        host, port = preview_server.address
        display_host = "localhost" if host == "0.0.0.0" else host
        print(f"Preview stream: http://{display_host}:{port}")

    frame_number = 0
    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                print("No frame received; stopping.")
                break

            frame = cv2.resize(frame, (args.width, args.height))
            frame = rotate_frame(frame, args.rotate)
            writer.write(frame)

            if args.show or preview_server is not None:
                preview = make_preview_frame(frame, frame_number, args.preview_width)

            if preview_server is not None:
                preview_server.update(preview)

            if args.show:
                cv2.imshow("record_video", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_number += 1

    except KeyboardInterrupt:
        print("Stopping recording...")
    finally:
        writer.release()
        camera.release()
        if preview_server is not None:
            preview_server.stop()
        if args.show:
            cv2.destroyAllWindows()

    print(f"Saved video to: {args.output}")


if __name__ == "__main__":
    main()
