#!/usr/bin/env python3
"""
Detect a pole with a forward-facing Pi Camera and send drive commands to Arduino.

This is a first field-test draft. Keep the robot raised or on blocks for initial
tests, verify motor directions with low power, then tune the thresholds outdoors.
"""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from capstone_robot.utils import *

REPO_ROOT = find_repo_root(__file__)
DEFAULT_MODEL = REPO_ROOT / "src/capstone_robot/models/pole/yolo11n_640/weights/best_ncnn_model"


def clamp(value, low, high):
    return max(low, min(high, value))


def parse_args():
    parser = argparse.ArgumentParser(description="Approach a detected pole using YOLO and Arduino serial motors")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="YOLO .pt or exported model path")
    parser.add_argument("--width", type=int, default=640, help="camera capture width")
    parser.add_argument("--height", type=int, default=480, help="camera capture height")
    parser.add_argument("--fps", type=float, default=30, help="camera framerate")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--conf", type=float, default=0.5, help="YOLO confidence threshold")
    parser.add_argument("--show", action="store_true", help="show local OpenCV preview")
    parser.add_argument("--preview-port", type=int, default=1234, help="serve browser preview on this port; use 0 to disable")
    parser.add_argument("--preview-host", default="0.0.0.0", help="host/interface for browser preview")
    parser.add_argument("--preview-width", type=int, default=640, help="browser preview width")
    parser.add_argument("--armed", action="store_true", help="required before motor commands are sent")
    parser.add_argument("--serial-port", default="/dev/ttyACM0", help="Arduino serial port")
    parser.add_argument("--baud", type=int, default=115200, help="Arduino serial baud rate")
    parser.add_argument("--serial-timeout", type=float, default=1.0, help="serial open/read timeout")
    parser.add_argument("--timeout", type=float, default=45.0, help="maximum autonomous run time in seconds")
    parser.add_argument("--base-speed", type=float, default=0.22, help="forward speed when centered")
    parser.add_argument("--search-speed", type=float, default=0.16, help="turn speed while searching")
    parser.add_argument("--turn-gain", type=float, default=0.55, help="steering gain from normalized x error")
    parser.add_argument("--max-turn", type=float, default=0.22, help="maximum steering correction")
    parser.add_argument("--center-deadband", type=float, default=0.08, help="ignore small normalized x errors")
    parser.add_argument("--stop-area-ratio", type=float, default=0.18, help="stop when pole box area/frame area exceeds this")
    parser.add_argument("--stop-height-ratio", type=float, default=0.78, help="stop when pole box height/frame height exceeds this")
    parser.add_argument("--lost-stop-frames", type=int, default=8, help="stop after this many missing frames while approaching")
    parser.add_argument("--detect-start-frames", type=int, default=3, help="detections required before forward motion")
    parser.add_argument("--left-trim", type=float, default=0.0, help="additive trim for left motor speed")
    parser.add_argument("--right-trim", type=float, default=0.0, help="additive trim for right motor speed")
    parser.add_argument("--invert-left", action="store_true", help="invert left motor command")
    parser.add_argument("--invert-right", action="store_true", help="invert right motor command")
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


class MotorDriver:
    def __init__(self, args):
        self.armed = args.armed
        self.left_trim = args.left_trim
        self.right_trim = args.right_trim
        self.invert_left = args.invert_left
        self.invert_right = args.invert_right
        self.serial = None

        if not self.armed:
            print("Motor serial disabled. Re-run with --armed after the robot is secured.")
            return

        try:
            import serial
        except ImportError as exc:
            raise RuntimeError("pyserial is not installed. Install with: python3 -m pip install pyserial") from exc

        self.serial = serial.Serial(args.serial_port, args.baud, timeout=args.serial_timeout)
        time.sleep(2.0)
        self.stop()
        print(f"Arduino motor serial armed on {args.serial_port} at {args.baud} baud")

    def set_wheel_speeds(self, left, right):
        left = clamp(left + self.left_trim, -1.0, 1.0)
        right = clamp(right + self.right_trim, -1.0, 1.0)

        if self.invert_left:
            left = -left
        if self.invert_right:
            right = -right

        if not self.armed:
            print(f"DRY RUN motors: left={left:+.2f} right={right:+.2f}")
            return

        self._send(f"M {left:.3f} {right:.3f}")

    def stop(self):
        if self.armed and self.serial is not None:
            self._send("S")

    def close(self):
        self.stop()
        if self.serial is not None:
            self.serial.close()

    def _send(self, command):
        self.serial.write((command + "\n").encode("ascii"))
        self.serial.flush()


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


def center_x(box):
    x1, _, x2, _ = box
    return (x1 + x2) / 2.0


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_height(box):
    _, y1, _, y2 = box
    return max(0.0, y2 - y1)


def pick_best_pole(result):
    best = None

    for box in result.boxes:
        cls_id = int(box.cls[0])
        class_name = result.names.get(cls_id, str(cls_id)).lower()
        if class_name != "pole":
            continue

        conf = float(box.conf[0])
        xyxy = [float(v) for v in box.xyxy[0].tolist()]
        detection = {
            "conf": conf,
            "box": xyxy,
            "center_x": center_x(xyxy),
            "area": box_area(xyxy),
            "height": box_height(xyxy),
        }

        if best is None or detection["conf"] > best["conf"]:
            best = detection

    return best


def compute_command(pole, frame_width, args):
    error_px = pole["center_x"] - (frame_width / 2.0)
    normalized_error = error_px / (frame_width / 2.0)

    if abs(normalized_error) < args.center_deadband:
        normalized_error = 0.0

    turn = clamp(args.turn_gain * normalized_error, -args.max_turn, args.max_turn)

    # Positive image error means pole is right of center, so slow the right wheel
    # and speed up the left wheel to turn right. Calibrate this sign on your robot.
    left = clamp(args.base_speed + turn, -1.0, 1.0)
    right = clamp(args.base_speed - turn, -1.0, 1.0)
    return left, right, error_px, normalized_error


def draw_overlay(frame, pole, status, left, right, error_px):
    annotated = frame.copy()

    if pole is not None:
        x1, y1, x2, y2 = [int(v) for v in pole["box"]]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.line(annotated, (int(pole["center_x"]), 0), (int(pole["center_x"]), frame.shape[0]), (0, 255, 0), 2)

    cv2.line(annotated, (frame.shape[1] // 2, 0), (frame.shape[1] // 2, frame.shape[0]), (255, 255, 255), 1)
    cv2.putText(annotated, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(
        annotated,
        f"L={left:+.2f} R={right:+.2f} err={error_px:+.1f}px",
        (10, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )
    return annotated


def resize_preview(frame, preview_width):
    if preview_width <= 0 or frame.shape[1] == preview_width:
        return frame

    scale = preview_width / frame.shape[1]
    preview_height = int(frame.shape[0] * scale)
    return cv2.resize(frame, (preview_width, preview_height))


def main():
    args = parse_args()

    if args.base_speed < 0 or args.search_speed < 0:
        raise SystemExit("Use positive --base-speed and --search-speed values.")

    if not args.model.exists():
        raise SystemExit(f"Model does not exist: {args.model}")

    model = YOLO(str(args.model), task="detect")
    motors = MotorDriver(args)

    try:
        camera = PiCamera(args.width, args.height, args.fps)
    except RuntimeError as exc:
        raise SystemExit(str(exc))

    frame_area = args.width * args.height
    detected_frames = 0
    lost_frames = 0
    start_time = time.monotonic()

    preview_server = None
    if args.preview_port > 0:
        preview_server = MjpegPreview(args.preview_host, args.preview_port)
        preview_server.start()
        host, port = preview_server.address
        display_host = "localhost" if host == "0.0.0.0" else host
        print(f"Preview stream: http://{display_host}:{port}")

    try:
        while True:
            elapsed = time.monotonic() - start_time
            if elapsed > args.timeout:
                print("Timeout reached; stopping.")
                break

            ok, frame = camera.read()
            if not ok or frame is None:
                print("No frame received; stopping.")
                break

            result = model(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
            pole = pick_best_pole(result)

            left = 0.0
            right = 0.0
            error_px = 0.0

            if pole is None:
                detected_frames = 0
                lost_frames += 1

                if lost_frames >= args.lost_stop_frames:
                    status = "POLE LOST - STOP"
                    motors.stop()
                else:
                    status = "SEARCH"
                    left = args.search_speed
                    right = -args.search_speed
                    motors.set_wheel_speeds(left, right)
            else:
                lost_frames = 0
                detected_frames += 1

                area_ratio = pole["area"] / frame_area
                height_ratio = pole["height"] / args.height

                if area_ratio >= args.stop_area_ratio or height_ratio >= args.stop_height_ratio:
                    status = f"NEAR POLE - STOP area={area_ratio:.2f} height={height_ratio:.2f}"
                    motors.stop()
                    annotated = draw_overlay(frame, pole, status, left, right, error_px)
                    if preview_server is not None:
                        preview_server.update(resize_preview(annotated, args.preview_width))
                    if args.show:
                        cv2.imshow("approach_pole_yolo", annotated)
                        cv2.waitKey(500)
                    print(status)
                    break

                if detected_frames < args.detect_start_frames:
                    status = f"CONFIRMING {detected_frames}/{args.detect_start_frames}"
                    motors.stop()
                else:
                    left, right, error_px, normalized_error = compute_command(pole, frame.shape[1], args)
                    status = f"APPROACH err={normalized_error:+.2f}"
                    motors.set_wheel_speeds(left, right)

            print(f"{status} left={left:+.2f} right={right:+.2f}")

            if args.show or preview_server is not None:
                annotated = draw_overlay(frame, pole, status, left, right, error_px)

            if preview_server is not None:
                preview_server.update(resize_preview(annotated, args.preview_width))

            if args.show:
                cv2.imshow("approach_pole_yolo", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("Keyboard interrupt; stopping.")
    finally:
        motors.close()
        camera.release()
        if preview_server is not None:
            preview_server.stop()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
