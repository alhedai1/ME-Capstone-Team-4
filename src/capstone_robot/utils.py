"""
Utility functions for the capstone project.
"""

import cv2
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time
from dataclasses import dataclass



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
            from libcamera import controls
        except ImportError as exc:
            raise RuntimeError("Picamera2 is not installed. Install python3-picamera2 on the Raspberry Pi.") from exc

        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"},
            controls={"FrameRate": fps},
        )
        self.picam2.configure(config)
        self.picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
        self.picam2.start()

    def read(self):
        frame = self.picam2.capture_array()
        if frame is None:
            return False, None
        return True, frame

    def release(self):
        self.picam2.stop()

@dataclass
class Detection:
    label: str
    confidence: float
    box: tuple  # (x, y, w, h)

class AiCamera:
    def __init__(self, model_path, width, height, fps, bbox_normalization=True, bbox_order="xy"):
        try:
            from picamera2 import Picamera2
            from picamera2.devices import IMX500
            from picamera2.devices.imx500 import NetworkIntrinsics
        except ImportError as exc:
            raise RuntimeError(
                "Picamera2 IMX500 support is not installed. Try:\n"
                "  sudo apt update\n"
                "  sudo apt install -y imx500-all python3-picamera2 python3-opencv"
            ) from exc

        self.imx500 = IMX500(str(model_path))

        self.intrinsics = self.imx500.network_intrinsics
        if not self.intrinsics:
            self.intrinsics = NetworkIntrinsics()
            self.intrinsics.task = "object detection"

        self.intrinsics.bbox_normalization = bbox_normalization
        self.intrinsics.bbox_order = bbox_order
        self.intrinsics.update_with_defaults()

        self.picam2 = Picamera2(self.imx500.camera_num)

        # Use RGB888 because OpenCV display/encoding is simple and predictable.
        # The IMX500 inference runs on the AI camera; the Pi only reads metadata.
        config = self.picam2.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"},
            controls={"FrameRate": fps},
            buffer_count=3,
        )

        print("Loading .rpk model onto Raspberry Pi AI Camera...")
        self.imx500.show_network_fw_progress_bar()

        self.picam2.configure(config)

        if getattr(self.intrinsics, "preserve_aspect_ratio", False):
            self.imx500.set_auto_aspect_ratio()

        print("Starting camera...")
        self.picam2.start()

        print("Waiting for IMX500 inference metadata...")
        for i in range(60):
            metadata = self.picam2.capture_metadata()
            outputs = self.imx500.get_outputs(metadata, add_batch=True)

            if outputs is not None:
                print(f"IMX500 outputs ready after {i + 1} metadata frames")
                break

            time.sleep(0.1)
        else:
            print("WARNING: IMX500 outputs never became ready")

    def read(self):
        request = self.picam2.capture_request()
        try:
            frame = request.make_array("main")
            metadata = request.get_metadata()
        finally:
            request.release()

        if frame is None or metadata is None:
            return False, None, None

        return True, frame, metadata

    def get_detections(self, metadata, labels=None, threshold=0.4):
        outputs = self.imx500.get_outputs(metadata, add_batch=True)

        if outputs is None:
            return []

        # For Ultralytics YOLO11n/YOOv8n IMX export with NMS/postprocess included,
        # the usual output order is: boxes, scores, classes.
        if len(outputs) < 3:
            return []

        boxes = outputs[0][0]
        scores = outputs[1][0]
        classes = outputs[2][0]

        if boxes is None or scores is None or classes is None:
            return []

        input_w, input_h = self.imx500.get_input_size()

        if self.intrinsics.bbox_normalization:
            boxes = boxes / input_h

        if self.intrinsics.bbox_order == "xy":
            # Convert xyxy -> yxyx before convert_inference_coords(), matching
            # Raspberry Pi's official IMX500 object-detection example pattern.
            boxes = boxes[:, [1, 0, 3, 2]]

        detections = []
        for box, score, cls in zip(boxes, scores, classes):
            confidence = float(score)
            if confidence < threshold:
                continue

            class_id = int(cls)
            if labels is not None and 0 <= class_id < len(labels):
                label = labels[class_id]
            else:
                label = str(class_id)

            x, y, w, h = self.imx500.convert_inference_coords(box, metadata, self.picam2)
            detections.append(
                Detection(
                    label=label,
                    confidence=confidence,
                    box=(int(x), int(y), int(w), int(h)),
                )
            )

        return detections

    def release(self):
        print("Releasing camera...")

        try:
            self.picam2.stop()
            time.sleep(0.5)
        except Exception as e:
            print(f"picam2.stop() error: {e}")

        try:
            self.picam2.close()
            time.sleep(1.0)
        except Exception as e:
            print(f"picam2.close() error: {e}")

        try:
            del self.picam2
        except Exception:
            pass

        try:
            del self.imx500
        except Exception:
            pass

        print("Camera released.")

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

# def get_detections(result, frame):
#     detections = []
#     if result.boxes is None:
#         return detections

#     names = result.names
#     for box in result.boxes:
#         cls_id = int(box.cls.item())
#         name = str(names.get(cls_id, cls_id)).lower()
#         x1, y1, x2, y2 = box.xyxy[0].tolist()
#         detections.append(
#             {
#                 "class_id": cls_id,
#                 "name": name,
#                 "conf": float(box.conf.item()),
#                 "box": clamp_box((x1, y1, x2, y2), frame.shape),
#             }
#         )
#     return detections

# def clamp_box(box, frame_shape):
#     height, width = frame_shape[:2]
#     x1, y1, x2, y2 = box
#     x1 = max(0, min(width - 1, int(round(x1))))
#     y1 = max(0, min(height - 1, int(round(y1))))
#     x2 = max(x1 + 1, min(width, int(round(x2))))
#     y2 = max(y1 + 1, min(height, int(round(y2))))
#     return (x1, y1, x2, y2)