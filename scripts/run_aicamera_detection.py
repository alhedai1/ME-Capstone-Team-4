#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time

import cv2


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "models/pole_yolo11n.rpk"
DEFAULT_LABELS = REPO_ROOT / "models/labels.txt"


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
    parser.add_argument("--conf", type=float, default=0.4, help="confidence threshold")
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


def rotate_frame(frame, rotation):
    if rotation == "cw":
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation == "ccw":
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotation == "180":
        return cv2.rotate(frame, cv2.ROTATE_180)
    return frame


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
            buffer_count=12,
        )

        print("Loading .rpk model onto Raspberry Pi AI Camera...")
        self.imx500.show_network_fw_progress_bar()

        self.picam2.start(config, show_preview=False)

        if getattr(self.intrinsics, "preserve_aspect_ratio", False):
            self.imx500.set_auto_aspect_ratio()

    def read(self):
        frame = self.picam2.capture_array()
        metadata = self.picam2.capture_metadata()

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
        f"box=({x},{y},{w},{h}), center_x={cx:.1f}, error_x={error_x:.1f}"
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

    frame_count = 0
    start_time = time.time()
    last_loop_time = start_time

    try:
        while True:
            ok, frame, metadata = camera.read()
            if not ok or frame is None or metadata is None:
                print("No frame/metadata received; stopping.")
                break

            frame = rotate_frame(frame, args.rotate)

            detections = camera.get_detections(
                metadata=metadata,
                labels=labels,
                threshold=args.conf,
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
        camera.release()
        if preview_server is not None:
            preview_server.stop()
        if args.show:
            cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    average_fps = frame_count / elapsed if elapsed > 0 else 0.0
    print(f"Frames processed: {frame_count}")
    print(f"Average loop FPS: {average_fps:.2f}")


if __name__ == "__main__":
    main()