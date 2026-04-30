#!/usr/bin/env python3
"""
pole_track_cv_usb.py

Detect and track a black vertical pole outdoors using OpenCV with a USB camera.

Features:
- Uses OpenCV VideoCapture with a USB camera device or index.
- Detects dark/black vertical objects by combining HSV (low V) and grayscale thresholds.
- Filters contours by area and aspect ratio to prefer tall thin poles.
- Simple exponential smoothing for centroid to reduce jitter.
- Optional serial output (simple single-character commands) for robot control.

Usage examples:
    python3 pole_track_cv_usb.py --display --serial /dev/ttyACM0 --baud 115200
    python3 pole_track_cv_usb.py --display --camera 0
    python3 pole_track_cv_usb.py --display --camera /dev/video0 --backend v4l2

Dependencies:
    pip install opencv-python numpy pyserial

Notes / tuning:
 - Outdoors, lighting varies: tune --v-thresh and --gray-thresh.
 - Adjust --min-area and --aspect-ratio for your pole size and camera mounting.
 - The serial protocol is intentionally minimal: 'L','R' (steer), 'F' (forward), 'S' (stop), 'N' (no target)
"""

import argparse
import time
import sys

import numpy as np
import cv2

try:
    import serial
    SERIAL_AVAILABLE = True
except Exception:
    SERIAL_AVAILABLE = False


class CameraSource:
    """USB camera wrapper that yields BGR frames via OpenCV VideoCapture."""

    def __init__(self, width=640, height=480, camera=0, backend="auto"):
        self.width = int(width)
        self.height = int(height)
        self.camera = self._normalize_camera(camera)
        self.backend = self._resolve_backend(backend)
        self.cap = None

    @staticmethod
    def _normalize_camera(camera):
        if isinstance(camera, str):
            camera = camera.strip()
            if camera.isdigit():
                return int(camera)
        return camera

    @staticmethod
    def _resolve_backend(backend):
        backend = backend.lower()
        backend_map = {
            "auto": cv2.CAP_ANY,
            "v4l2": getattr(cv2, "CAP_V4L2", cv2.CAP_ANY),
            "gstreamer": getattr(cv2, "CAP_GSTREAMER", cv2.CAP_ANY),
            "ffmpeg": getattr(cv2, "CAP_FFMPEG", cv2.CAP_ANY),
        }
        if backend not in backend_map:
            raise ValueError(f"Unsupported backend '{backend}'. Use one of: {', '.join(backend_map)}")
        return backend_map[backend]

    def open(self):
        self.cap = cv2.VideoCapture(self.camera, self.backend)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open USB camera {self.camera!r}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        print(f"Using USB camera {self.camera!r} with OpenCV VideoCapture")

    def read(self):
        return self.cap.read()

    def release(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass


def detect_black_pole(frame, v_thresh=60, gray_thresh=80, min_area=1500, aspect_ratio_min=3.0):
    """Detect candidate black vertical pole in frame.

    Returns dict with keys: found (bool), bbox (x,y,w,h), centroid (cx,cy), area, mask
    """
    h, w = frame.shape[:2]

    # Optional resize already handled by camera; convert
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    # Mask for low V (dark) in HSV
    # V channel is hsv[...,2]
    hsv_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, int(v_thresh)))

    # Mask for dark in grayscale
    _, gray_mask = cv2.threshold(gray, int(gray_thresh), 255, cv2.THRESH_BINARY_INV)

    # Combine masks
    mask = cv2.bitwise_and(hsv_mask, gray_mask)

    # Morphological ops to clean
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bh / (bw + 1e-6)
        if aspect < aspect_ratio_min:
            continue
        # prefer the largest by area
        if area > best_area:
            best_area = area
            best = (x, y, bw, bh, area)

    if best is None:
        return {"found": False, "mask": mask}

    x, y, bw, bh, area = best
    cx = x + bw / 2.0
    cy = y + bh / 2.0
    return {"found": True, "bbox": (int(x), int(y), int(bw), int(bh)), "centroid": (int(cx), int(cy)), "area": int(area), "mask": mask}


def make_command_from_detection(detection, frame_width, x_tol=0.08, forward_area=15000):
    """Generate a simple char command based on detection.

    - 'L' turn left
    - 'R' turn right
    - 'F' forward
    - 'S' stop (target close enough)
    - 'N' no detection
    """
    if not detection["found"]:
        return 'N'
    cx, _ = detection["centroid"]
    # normalize error in [-0.5, 0.5]
    err = (cx - frame_width / 2.0) / frame_width
    # steering decision
    if abs(err) > x_tol:
        return 'L' if err < 0 else 'R'
    # distance decision using area (larger area = closer)
    if detection["area"] < forward_area:
        return 'F'
    return 'S'


def run(args):
    cam = CameraSource(width=args.width, height=args.height, camera=args.camera, backend=args.backend)
    cam.open()

    ser = None
    if args.serial:
        if not SERIAL_AVAILABLE:
            print("pyserial not installed; serial output disabled")
        else:
            try:
                ser = serial.Serial(args.serial, args.baud, timeout=0.1)
                print(f"Opened serial port {args.serial} @ {args.baud}")
            except Exception as e:
                print("Failed to open serial port:", e)
                ser = None

    # smoothing state
    smoothed_cx = None
    smoothed_cy = None

    try:
        while True:
            ok, frame = cam.read()
            if not ok or frame is None:
                print("Failed to read frame from camera")
                time.sleep(0.1)
                continue

            # Optionally resize to target dims (ensure consistent processing)
            frame = cv2.resize(frame, (args.width, args.height))

            det = detect_black_pole(frame, v_thresh=args.v_thresh, gray_thresh=args.gray_thresh,
                                     min_area=args.min_area, aspect_ratio_min=args.aspect_ratio)

            if det.get("found"):
                cx, cy = det["centroid"]
                if smoothed_cx is None:
                    smoothed_cx = cx
                    smoothed_cy = cy
                else:
                    alpha = args.alpha
                    smoothed_cx = alpha * cx + (1 - alpha) * smoothed_cx
                    smoothed_cy = alpha * cy + (1 - alpha) * smoothed_cy

                cmd = make_command_from_detection(det, frame.shape[1], x_tol=args.x_tol, forward_area=args.forward_area)
            else:
                smoothed_cx = None
                smoothed_cy = None
                cmd = 'N'

            # Send command if serial open
            if ser is not None:
                try:
                    ser.write((cmd + "\n").encode('ascii'))
                except Exception:
                    pass

            # Display annotated frame
            if args.display:
                out = frame.copy()
                if det.get("found"):
                    x, y, bw, bh = det["bbox"]
                    cv2.rectangle(out, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                    # draw smoothed centroid
                    if smoothed_cx is not None:
                        cv2.circle(out, (int(smoothed_cx), int(smoothed_cy)), 6, (0, 0, 255), -1)
                    cx, cy = det["centroid"]
                    cv2.circle(out, (int(cx), int(cy)), 4, (255, 0, 0), -1)
                    cv2.putText(out, f"area={det['area']}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(out, "No detection", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                # draw center line
                fh, fw = out.shape[:2]
                cv2.line(out, (fw // 2, 0), (fw // 2, fh), (200, 200, 200), 1)
                cv2.putText(out, f"cmd={cmd}", (10, fh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                cv2.imshow('pole_tracker', out)
                # small mask window for debugging
                if det.get("mask") is not None:
                    cv2.imshow('mask', det['mask'])

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

            else:
                # when not displaying, print simple telemetry occasionally
                print(f"cmd={cmd} area={det.get('area',0)} found={det.get('found',False)}")
                time.sleep(0.05)

    except KeyboardInterrupt:
        print("Exiting on user interrupt")
    finally:
        cam.release()
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass
        if args.display:
            cv2.destroyAllWindows()


def parse_args():
    p = argparse.ArgumentParser(description="Pole tracker for a USB camera with OpenCV")
    p.add_argument('--width', type=int, default=640, help='capture width')
    p.add_argument('--height', type=int, default=480, help='capture height')
    p.add_argument('--camera', type=str, default='0', help='USB camera index or device path (for example: 0 or /dev/video0)')
    p.add_argument('--backend', type=str, default='auto', choices=['auto', 'v4l2', 'gstreamer', 'ffmpeg'],
                   help='OpenCV capture backend for the USB camera')
    p.add_argument('--display', action='store_true', help='show annotated display windows')
    p.add_argument('--serial', type=str, default=None, help='serial port to send commands to (e.g. /dev/ttyACM0)')
    p.add_argument('--baud', type=int, default=115200, help='baud for serial')
    p.add_argument('--v-thresh', type=int, default=60, help='HSV V channel threshold for "black"')
    p.add_argument('--gray-thresh', type=int, default=80, help='grayscale threshold for "black"')
    p.add_argument('--min-area', type=int, default=1500, help='minimum contour area to consider')
    p.add_argument('--aspect-ratio', type=float, default=3.0, help='minimum height/width for a pole-like object')
    p.add_argument('--alpha', type=float, default=0.6, help='smoothing alpha for centroid (0-1)')
    p.add_argument('--x-tol', type=float, default=0.08, help='normalized x tolerance for steering (0-0.5)')
    p.add_argument('--forward-area', type=int, default=15000, help='area threshold above which we stop (close to pole)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    try:
        run(args)
    except Exception as e:
        print('Error running pole tracker:', e)
        sys.exit(1)
