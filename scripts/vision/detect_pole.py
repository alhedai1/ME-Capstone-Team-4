#!/usr/bin/env python3
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading

import cv2
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO pole detection from the Pi Camera")
    parser.add_argument("--model", type=Path, default=REPO_ROOT / "models/best_ncnn_model_sz320")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--show", action="store_true", help="show local OpenCV preview")
    parser.add_argument("--preview-port", type=int, default=None, help="serve browser preview on this port")
    parser.add_argument("--preview-host", default="0.0.0.0", help="host/interface for browser preview")
    parser.add_argument("--preview-width", type=int, default=640, help="browser preview width")
    return parser.parse_args()


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


def resize_preview(frame, preview_width):
    if preview_width <= 0 or frame.shape[1] == preview_width:
        return frame

    scale = preview_width / frame.shape[1]
    preview_height = int(frame.shape[0] * scale)
    return cv2.resize(frame, (preview_width, preview_height))


def main():
    args = parse_args()
    model = YOLO(str(args.model))

    try:
        camera = PiCamera(args.width, args.height, args.fps)
    except RuntimeError as exc:
        raise SystemExit(str(exc))

    preview_server = None
    if args.preview_port is not None:
        preview_server = MjpegPreview(args.preview_host, args.preview_port)
        preview_server.start()
        host, port = preview_server.address
        display_host = "localhost" if host == "0.0.0.0" else host
        print(f"Preview stream: http://{display_host}:{port}")

    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                print("No frame received; stopping.")
                break

            results = model(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)
            annotated = results[0].plot()
            preview = resize_preview(annotated, args.preview_width)

            if preview_server is not None:
                preview_server.update(preview)

            if args.show:
                cv2.imshow("detect_pole", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        camera.release()
        if preview_server is not None:
            preview_server.stop()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
