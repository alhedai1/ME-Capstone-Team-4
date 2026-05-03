#!/usr/bin/env python3
import argparse
from datetime import datetime
from pathlib import Path

import cv2


def parse_source(value):
    if value.lower() in {"picamera", "picam", "pi"}:
        return "picamera"
    return int(value) if value.isdigit() else value


def parse_args():
    parser = argparse.ArgumentParser(description="Record video from an OpenCV camera source")
    parser.add_argument("--source", default="picamera", help="camera index or video source")
    parser.add_argument("--output", type=Path, default=Path("data/test_videos/recording.mp4"))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--rotate", choices=["none", "cw", "ccw", "180"], default="none")
    parser.add_argument("--show", action="store_true", help="show preview window")
    return parser.parse_args()


def rotate_frame(frame, rotation):
    if rotation == "cw":
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation == "ccw":
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotation == "180":
        return cv2.rotate(frame, cv2.ROTATE_180)
    return frame


def writer_size(width, height, rotation):
    if rotation in {"cw", "ccw"}:
        return height, width
    return width, height


class OpenCVCamera:
    def __init__(self, source, width, height, fps):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()


class PiCamera:
    def __init__(self, width, height, fps):
        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            raise RuntimeError("Picamera2 is not installed. Install python3-picamera2 on the Raspberry Pi.") from exc

        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"},
            controls={"FrameRate": fps},
        )
        self.picam2.configure(config)
        self.picam2.start()

    def read(self):
        frame = self.picam2.capture_array()
        if frame is None:
            return False, None
        return True, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def release(self):
        self.picam2.stop()


def open_camera(source, width, height, fps):
    if source == "picamera":
        return PiCamera(width, height, fps)
    return OpenCVCamera(source, width, height, fps)


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


def main():
    args = parse_args()
    source = parse_source(str(args.source))

    try:
        camera = open_camera(source, args.width, args.height, args.fps)
    except RuntimeError as exc:
        raise SystemExit(str(exc))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_width, out_height = writer_size(args.width, args.height, args.rotate)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.output), fourcc, args.fps, (out_width, out_height))
    if not writer.isOpened():
        camera.release()
        raise SystemExit(f"Could not open output video for writing: {args.output}")

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

            if args.show:
                preview = frame.copy()
                draw_overlay(preview, frame_number)
                cv2.imshow("record_video", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_number += 1

    except KeyboardInterrupt:
        print("Stopping recording...")
    finally:
        writer.release()
        camera.release()
        if args.show:
            cv2.destroyAllWindows()

    print(f"Saved video to: {args.output}")


if __name__ == "__main__":
    main()
