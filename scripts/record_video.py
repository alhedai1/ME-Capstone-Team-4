#!/usr/bin/env python3
import argparse
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Record video from the Raspberry Pi Camera")
    # parser.add_argument("--output", type=Path, default=Path("data/test_videos/recording.mp4"))
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data/test_videos/recording.mp4",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--rotate", choices=["none", "cw", "ccw", "180"], default="none")
    parser.add_argument("--show", action="store_true", help="show preview window")
    parser.add_argument("--preview-port", type=int, default=None, help="serve browser preview on this port")
    parser.add_argument("--preview-host", default="0.0.0.0", help="host/interface for browser preview")
    parser.add_argument("--preview-width", type=int, default=640, help="browser preview width")
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
        return True, frame

    def release(self):
        self.picam2.stop()


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


class MjpegPreview:
    def __init__(self, host, port):
        self.frame = None
        self.condition = threading.Condition()

        preview = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path in {"/", "/index.html"}:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(
                        b"<html><body><img src='/stream' style='max-width:100%;'></body></html>"
                    )
                    return

                if self.path != "/stream":
                    self.send_error(404)
                    return

                self.send_response(200)
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Pragma", "no-cache")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()

                while True:
                    with preview.condition:
                        preview.condition.wait(timeout=1.0)
                        jpg = preview.frame

                    if jpg is None:
                        continue

                    try:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode("ascii"))
                        self.wfile.write(jpg)
                        self.wfile.write(b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        break

            def log_message(self, format, *args):
                return

        self.httpd = ThreadingHTTPServer((host, port), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def address(self):
        host, port = self.httpd.server_address
        return host, port

    def start(self):
        self.thread.start()

    def update(self, frame):
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ok:
            return

        with self.condition:
            self.frame = encoded.tobytes()
            self.condition.notify_all()

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


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
