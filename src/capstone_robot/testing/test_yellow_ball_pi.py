#!/usr/bin/env python3
import argparse
import time

import cv2
import numpy as np

from capstone_robot.utils import MjpegPreview, PiCamera, resize_preview, rotate_frame


def detect_yellow_ball(frame_bgr, min_area=10000, max_area=100000):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, np.array([0, 40, 40]), np.array([35, 255, 255]))
    _, s, v = cv2.split(hsv)
    mask = cv2.bitwise_and(mask, cv2.inRange(s, 50, 255))
    mask = cv2.bitwise_and(mask, cv2.inRange(v, 50, 255))

    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect = w / h if h else 0
        if not 0.75 <= aspect <= 1.33:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        circle_area = np.pi * radius * radius
        fill_ratio = area / circle_area if circle_area > 0 else 0

        hull_area = cv2.contourArea(cv2.convexHull(contour))
        solidity = area / hull_area if hull_area > 0 else 0
        if solidity < 0.75:
            continue

        score = (
            2.0 * fill_ratio
            + 1.5 * solidity
            + 1.0 * (1.0 - abs(aspect - 1.0))
            - 0.00001 * area
        )

        if circularity > 0.45 and fill_ratio > 0.45 and radius > 20:
            candidates.append((score, contour, int(cx), int(cy), int(radius), area))

    if not candidates:
        return None, mask

    return max(candidates, key=lambda item: item[0]), mask


def draw_detection(frame_bgr, detection, fps):
    vis = frame_bgr.copy()

    if detection is None:
        status = f"NO BALL  fps={fps:.1f}"
    else:
        _, contour, x, y, radius, area = detection
        overlay = vis.copy()
        cv2.drawContours(overlay, [contour], -1, (0, 255, 255), -1)
        cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)
        cv2.circle(vis, (x, y), radius, (0, 0, 255), 3)
        cv2.circle(vis, (x, y), 4, (0, 0, 255), -1)
        status = f"BALL x={x} y={y} r={radius} area={area:.0f} fps={fps:.1f}"

    cv2.putText(vis, status, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return vis


def build_preview(frame_bgr, detection, mask, fps, debug_mask):
    vis = draw_detection(frame_bgr, detection, fps)
    if not debug_mask:
        return vis

    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.putText(mask_bgr, "MASK", (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return np.hstack((vis, mask_bgr))


def main():
    parser = argparse.ArgumentParser(description="Track a yellow ball from the Pi camera in a browser preview.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--rotate", choices=["none", "cw", "ccw", "180"], default="none")
    parser.add_argument("--min-area", type=int, default=2000)
    parser.add_argument("--max-area", type=int, default=100000)
    parser.add_argument(
        "--camera-format",
        choices=["rgb", "bgr"],
        default="rgb",
        help="Picamera2 RGB888 normally returns rgb; try bgr if the mask misses obvious yellow.",
    )
    parser.add_argument("--debug-mask", action="store_true", help="show the threshold mask beside the camera view")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=1234)
    parser.add_argument("--preview-width", type=int, default=640)
    parser.add_argument("--jpeg-quality", type=int, default=75)
    parser.add_argument("--show", action="store_true", help="also show a local OpenCV window")
    args = parser.parse_args()

    camera = PiCamera(0, args.width, args.height, args.fps)
    preview = MjpegPreview(args.host, args.port, args.jpeg_quality)
    preview.start()
    print(f"Preview stream: http://<RPI_IP_ADDRESS>:{args.port}")
    print("Press Ctrl+C to stop.")

    frame_count = 0
    started_at = time.time()

    try:
        while True:
            ok, frame_rgb = camera.read()
            if not ok or frame_rgb is None:
                time.sleep(0.05)
                continue

            frame = rotate_frame(frame_rgb, args.rotate)
            if args.camera_format == "rgb":
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame

            detection, mask = detect_yellow_ball(frame_bgr, args.min_area, args.max_area)
            frame_count += 1
            fps = frame_count / max(1e-6, time.time() - started_at)

            vis = build_preview(frame_bgr, detection, mask, fps, args.debug_mask)
            preview.update(resize_preview(vis, args.preview_width))

            if args.show:
                cv2.imshow("Yellow ball Pi camera test", vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

    except KeyboardInterrupt:
        print("\nStopping yellow ball test.")
    finally:
        preview.stop()
        camera.release()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
