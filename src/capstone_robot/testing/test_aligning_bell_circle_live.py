import argparse
import time

import cv2

from capstone_robot.utils import MjpegPreview, PiCamera, resize_preview, rotate_frame
from capstone_robot.vision.pole_bell2 import PoleBellTracker

try:
    from libcamera import controls
except ImportError:
    controls = None


def aligning_controls():
    if controls is None:
        return {}
    return {
        "AfMode": controls.AfModeEnum.Manual,
        "LensPosition": 0.0,
        "AwbMode": controls.AwbModeEnum.Daylight,
        "ExposureValue": -1.5,
    }


def draw_line(img, line, color=(0, 255, 0), thickness=2):
    out = img.copy()
    vx, vy, x0, y0 = line
    t = 1000
    x1 = int(x0 - vx * t)
    y1 = int(y0 - vy * t)
    x2 = int(x0 + vx * t)
    y2 = int(y0 + vy * t)
    cv2.line(out, (x1, y1), (x2, y2), color, thickness)
    return out


def orbit_seconds_from_error(error_px, px_per_second, min_seconds, max_seconds):
    seconds = abs(error_px) / max(1.0, px_per_second)
    return max(min_seconds, min(max_seconds, seconds))


def draw_alignment(frame, alignment, status, threshold):
    vis = frame.copy()
    if alignment is not None:
        vis = draw_line(vis, alignment.pole_line, (0, 255, 0), 3)
        bx, by, br = alignment.bell
        cv2.circle(vis, (bx, by), br, (255, 0, 0), 2)
        cv2.circle(vis, (bx, by), 3, (0, 0, 255), -1)

        aligned = abs(alignment.error_px) <= threshold
        color = (0, 255, 0) if aligned else (0, 165, 255)
        cv2.putText(
            vis,
            f"error={alignment.error_px:.1f}px side={alignment.side}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.putText(vis, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
    return vis


def parse_args():
    parser = argparse.ArgumentParser(
        description="Live no-motor pole/bell alignment test using the upward Pi camera."
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--rotation", choices=["none", "cw", "ccw", "180"], default="180")
    parser.add_argument("--threshold", type=float, default=20.0)
    parser.add_argument("--stable-frames", type=int, default=4)
    parser.add_argument("--px-per-second", type=float, default=80.0)
    parser.add_argument("--min-seconds", type=float, default=0.2)
    parser.add_argument("--max-seconds", type=float, default=1.2)
    parser.add_argument("--preview-width", type=int, default=640)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=1234)
    parser.add_argument("--jpeg-quality", type=int, default=75)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-camera-controls", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    camera = PiCamera(idx=args.camera_index, width=args.width, height=args.height, fps=args.fps)
    if not args.no_camera_controls:
        controls_dict = aligning_controls()
        if controls_dict:
            camera.picam2.set_controls(controls_dict)

    tracker = PoleBellTracker(color_format="rgb")
    preview = None
    if not args.no_preview:
        preview = MjpegPreview(host=args.host, port=args.port, jpeg_quality=args.jpeg_quality)
        preview.start()
        print(f"Preview stream: http://<RPI_IP_ADDRESS>:{args.port}")

    stable_count = 0
    frame_count = 0
    started_at = time.time()
    print("Running live alignment test with NO motor commands. Press Ctrl+C to stop.")
    if args.show:
        print("OpenCV window enabled. Press q or Esc in the window to stop.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                print("[ALIGN LIVE] No camera frame")
                time.sleep(0.05)
                continue

            if args.rotation != "none":
                frame = rotate_frame(frame, args.rotation)

            alignment = tracker.detect(frame)
            frame_count += 1
            fps = frame_count / max(1e-6, time.time() - started_at)

            if alignment is None:
                stable_count = 0
                status = f"NO POLE/BELL fps={fps:.1f}"
                orbit_seconds = None
            else:
                aligned = abs(alignment.error_px) <= args.threshold
                stable_count = min(args.stable_frames, stable_count + 1) if aligned else 0
                orbit_seconds = orbit_seconds_from_error(
                    alignment.error_px,
                    args.px_per_second,
                    args.min_seconds,
                    args.max_seconds,
                )
                status = (
                    f"{'ALIGNED' if aligned else alignment.side.upper()} "
                    f"stable={stable_count}/{args.stable_frames} "
                    f"orbit={orbit_seconds:.2f}s fps={fps:.1f}"
                )

            vis = draw_alignment(frame, alignment, status, args.threshold)
            if preview is not None:
                preview.update(resize_preview(vis, args.preview_width))

            if args.show:
                cv2.imshow("live pole/bell alignment no motors", vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("r"):
                    tracker.reset()
                    stable_count = 0
                    print("tracker reset")

            if frame_count % max(1, args.fps) == 0:
                if alignment is None:
                    print(status)
                else:
                    print(
                        f"{status} error={alignment.error_px:.1f}px "
                        f"side={alignment.side} bell={alignment.bell}"
                    )

    except KeyboardInterrupt:
        print("\nStopping live alignment test.")
    finally:
        if preview is not None:
            preview.stop()
        if args.show:
            cv2.destroyAllWindows()
        camera.release()


if __name__ == "__main__":
    main()
