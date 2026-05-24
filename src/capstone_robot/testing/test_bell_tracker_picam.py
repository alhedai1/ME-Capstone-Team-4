import argparse
import time

import cv2

from capstone_robot.utils import MjpegPreview, PiCamera, resize_preview, rotate_frame
from capstone_robot.vision.bell import BellTracker


def draw_bell(frame_bgr, bell, status):
    vis = frame_bgr.copy()

    if bell is not None:
        x, y, w, h = bell.box
        cx, cy = bell.center
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(vis, (int(cx), int(cy)), 4, (0, 255, 0), -1)
        cv2.putText(
            vis,
            f"area={bell.area} brass={bell.brass_ratio:.2f}",
            (x, max(25, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(vis, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
    return vis


def main():
    parser = argparse.ArgumentParser(description="Preview BellTracker detections from the Pi camera only.")
    parser.add_argument("--width", type=int, default=640, help="Pi camera frame width")
    parser.add_argument("--height", type=int, default=480, help="Pi camera frame height")
    parser.add_argument("--fps", type=int, default=30, help="Pi camera frame rate")
    parser.add_argument(
        "--rotation",
        choices=["none", "cw", "ccw", "180"],
        default="none",
        help="Rotate frame before detection",
    )
    parser.add_argument("--required-frames", type=int, default=3, help="Stable detection count shown in status")
    parser.add_argument("--preview-width", type=int, default=640, help="MJPEG preview width")
    parser.add_argument("--host", default="0.0.0.0", help="MJPEG preview host")
    parser.add_argument("--port", type=int, default=1235, help="MJPEG preview port")
    parser.add_argument("--jpeg-quality", type=int, default=75, help="MJPEG preview JPEG quality")
    parser.add_argument("--no-preview", action="store_true", help="Disable MJPEG preview server")
    parser.add_argument("--show", action="store_true", help="Also show an OpenCV window if a desktop is available")
    args = parser.parse_args()

    camera = PiCamera(width=args.width, height=args.height, fps=args.fps)
    tracker = BellTracker(color_format="rgb")
    preview = None
    stable_frames = 0
    frame_count = 0
    started_at = time.time()

    if not args.no_preview:
        preview = MjpegPreview(host=args.host, port=args.port, jpeg_quality=args.jpeg_quality)
        preview.start()
        print(f"Preview stream: http://<RPI_IP_ADDRESS>:{args.port}")

    print("Running BellTracker test. Press Ctrl+C to stop.")
    if args.show:
        print("OpenCV window enabled. Press q or Esc in the window to stop.")

    try:
        while True:
            ok, frame_rgb = camera.read()
            if not ok or frame_rgb is None:
                print("[BELL TEST] No camera frame")
                time.sleep(0.05)
                continue

            if args.rotation != "none":
                frame_rgb = rotate_frame(frame_rgb, args.rotation)

            bell = tracker.detect(frame_rgb)
            frame_count += 1

            if bell is None:
                stable_frames = 0
                status = "BELL TEST: NO BELL"
            else:
                stable_frames = min(args.required_frames, stable_frames + 1)
                status = f"BELL TEST: BELL {stable_frames}/{args.required_frames}"

            elapsed = max(1e-6, time.time() - started_at)
            fps = frame_count / elapsed
            status = f"{status}  fps={fps:.1f}"

            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            vis = draw_bell(frame_bgr, bell, status)

            if preview is not None:
                preview.update(resize_preview(vis, args.preview_width))

            if args.show:
                cv2.imshow("BellTracker Pi camera test", vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

            if frame_count % max(1, args.fps) == 0:
                print(status)

    except KeyboardInterrupt:
        print("\nStopping BellTracker test.")
    finally:
        if preview is not None:
            preview.stop()
        if args.show:
            cv2.destroyAllWindows()
        camera.release()


if __name__ == "__main__":
    main()
