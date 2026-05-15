"""
Utility functions for the capstone project.
"""

import cv2
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading


def find_repo_root(start_path):
    """
    Walk up the directory tree from start_path until finding a .git directory,
    which indicates the repository root. Raises ValueError if not found.
    """
    current = Path(start_path).resolve()
    while current != current.parent:  # Stop at filesystem root
        if (current / '.git').exists():
            return current
        current = current.parent
    raise ValueError("Could not find repository root (.git directory not found)")


def rotate_frame(frame, rotation):
    if rotation == "cw":
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation == "ccw":
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotation == "180":
        return cv2.rotate(frame, cv2.ROTATE_180)
    return frame

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
    def __init__(self, host, port, jpeg_quality=75):
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

def get_detections(result, frame):
    detections = []
    if result.boxes is None:
        return detections

    names = result.names
    for box in result.boxes:
        cls_id = int(box.cls.item())
        name = str(names.get(cls_id, cls_id)).lower()
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append(
            {
                "class_id": cls_id,
                "name": name,
                "conf": float(box.conf.item()),
                "box": clamp_box((x1, y1, x2, y2), frame.shape),
            }
        )
    return detections

def clamp_box(box, frame_shape):
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, int(round(x1))))
    y1 = max(0, min(height - 1, int(round(y1))))
    x2 = max(x1 + 1, min(width, int(round(x2))))
    y2 = max(y1 + 1, min(height, int(round(y2))))
    return (x1, y1, x2, y2)