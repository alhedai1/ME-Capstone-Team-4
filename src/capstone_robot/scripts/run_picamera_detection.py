#!/usr/bin/env python3
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time

import cv2
from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "runs/detect/runs/pole/yolo26n_640/weights/best_ncnn_model"


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO detection on a live Raspberry Pi Camera feed")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="trained YOLO .pt or exported model path")
    parser.add_argument("--width", type=int, default=640, help="camera capture width")
    parser.add_argument("--height", type=int, default=480, help="camera capture height")
    parser.add_argument("--fps", type=float, default=30, help="camera framerate")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--conf", type=float, default=0.4, help="confidence threshold")
    parser.add_argument("--rotate", choices=["none", "cw", "ccw", "180"], default="none")
    parser.add_argument("--show", action="store_true", help="show local OpenCV preview window")
    parser.add_argument("--preview-port", type=int, default=1234, help="serve browser preview on this port; use 0 to disable")
    parser.add_argument("--preview-host", default="0.0.0.0", help="host/interface for browser preview")
    parser.add_argument("--preview-width", type=int, default=640, help="browser preview width")
    parser.add_argument("--jpeg-quality", type=int, default=75, help="MJPEG preview JPEG quality from 1 to 100")
    return parser.parse_args()


def rotate_frame(frame, rotation):
    if rotation == "cw":
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation == "ccw":
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotation == "180":
        return cv2.rotate(frame, cv2.ROTATE_180)
    return frame


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
    def __init__(self, host, port, jpeg_quality):
        self.frame = None
        self.jpeg_quality = jpeg_quality
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
        quality = max(1, min(100, self.jpeg_quality))
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
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


def draw_status(frame, inference_fps, average_fps, frame_count):
    cv2.putText(
        frame,
        f"infer FPS: {inference_fps:.1f}  avg FPS: {average_fps:.1f}  frame: {frame_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def main():
    args = parse_args()
    model_path = resolve_model_path(args.model)
    print(f"Using model: {model_path}")

    model = YOLO(str(model_path), task="detect")

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
            result = model(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
            infer_time = time.time() - infer_start

            annotated = result.plot()
            elapsed = time.time() - start_time
            inference_fps = 1.0 / infer_time if infer_time > 0 else 0.0
            average_fps = (frame_count + 1) / elapsed if elapsed > 0 else 0.0
            draw_status(annotated, inference_fps, average_fps, frame_count)

            if preview_server is not None:
                preview_server.update(resize_preview(annotated, args.preview_width))

            if args.show:
                cv2.imshow("picamera_yolo_detection", annotated)
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
