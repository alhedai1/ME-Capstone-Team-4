import argparse
import time
from pathlib import Path

import cv2

from capstone_robot.utils import AiCamera, MjpegPreview, resize_preview
from capstone_robot.vision.bell_circle_climb import BellCircle


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PACKAGE_ROOT / "models" / "pole_imx" / "network.rpk"


def draw_detection(frame, detection, status):
    vis = frame.copy()
    height, width = vis.shape[:2]
    cv2.line(vis, (width // 2, 0), (width // 2, height), (80, 80, 80), 1)
    cv2.line(vis, (0, height // 2), (width, height // 2), (80, 80, 80), 1)

    if detection is not None:
        x, y, radius = detection.circle
        error_x = x - width / 2.0
        cv2.circle(vis, (x, y), radius, (255, 0, 0), 2)
        cv2.circle(vis, (x, y), 3, (0, 255, 0), -1)
        cv2.putText(
            vis,
            f"x={x} y={y} r={radius} err={error_x:.1f}px",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(vis, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return vis


def parse_args():
    parser = argparse.ArgumentParser(
        description="Live BellCircle climb detector test using the Raspberry Pi AI Camera frame."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="IMX500 .rpk model path")
    parser.add_argument("--width", type=int, default=640, help="AI camera frame width")
    parser.add_argument("--height", type=int, default=480, help="AI camera frame height")
    parser.add_argument("--fps", type=int, default=15, help="AI camera frame rate")
    parser.add_argument("--dp", type=float, default=1.5)
    parser.add_argument("--min-dist", type=int, default=5)
    parser.add_argument("--param1", type=int, default=50)
    parser.add_argument("--param2", type=int, default=50)
    parser.add_argument("--min-radius", type=int, default=10)
    parser.add_argument("--startup-max-radius", type=int, default=50)
    parser.add_argument("--tracking-max-radius", type=int, default=130)
    parser.add_argument("--lost-after-frames", type=int, default=8)
    parser.add_argument("--startup-confirm-frames", type=int, default=2)
    parser.add_argument("--preview-width", type=int, default=640)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=1234)
    parser.add_argument("--jpeg-quality", type=int, default=75)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--show", action="store_true", help="Also show an OpenCV window if a desktop is available")
    parser.add_argument("--debug-hough", action="store_true", help="Show raw Hough circle candidates in an OpenCV window")
    return parser.parse_args()


def main():
    args = parse_args()
    camera = AiCamera(
        model_path=args.model,
        width=args.width,
        height=args.height,
        fps=args.fps,
        bbox_normalization=True,
        bbox_order="xy",
    )
    detector = BellCircle(
        color_format="rgb",
        dp=args.dp,
        min_dist=args.min_dist,
        param1=args.param1,
        param2=args.param2,
        min_radius=args.min_radius,
        startup_max_radius=args.startup_max_radius,
        tracking_max_radius=args.tracking_max_radius,
        lost_after_frames=args.lost_after_frames,
        startup_confirm_threshold=args.startup_confirm_frames,
        show_debug=args.debug_hough,
    )

    preview = None
    if not args.no_preview:
        preview = MjpegPreview(host=args.host, port=args.port, jpeg_quality=args.jpeg_quality)
        preview.start()
        print(f"Preview stream: http://<RPI_IP_ADDRESS>:{args.port}")

    frame_count = 0
    detected_count = 0
    started_at = time.time()

    print("Running live BellCircle climb test with NO motor commands. Press Ctrl+C to stop.")
    if args.show:
        print("OpenCV window enabled. Press q or Esc in the window to stop. Press r to reset tracker.")

    try:
        while True:
            loop_started = time.time()
            ok, frame, _metadata = camera.read()
            if not ok or frame is None:
                print("[BELL CIRCLE LIVE] No AI camera frame")
                time.sleep(0.05)
                continue

            detection = detector.detect(frame)
            frame_count += 1
            if detection is not None:
                detected_count += 1

            fps_now = 1.0 / max(1e-6, time.time() - loop_started)
            fps_avg = frame_count / max(1e-6, time.time() - started_at)
            hit_rate = detected_count / max(1, frame_count)

            if detection is None:
                status = f"BELL CIRCLE: NONE fps={fps_now:.1f}/{fps_avg:.1f} hit={hit_rate:.2f}"
            else:
                error_x = detection.x - frame.shape[1] / 2.0
                status = (
                    f"BELL CIRCLE: TRACK err={error_x:.1f}px r={detection.radius} "
                    f"stable={detector.stable_frames} missed={detector.missed_frames} "
                    f"fps={fps_now:.1f}/{fps_avg:.1f} hit={hit_rate:.2f}"
                )

            vis = draw_detection(frame, detection, status)

            if preview is not None:
                preview.update(resize_preview(vis, args.preview_width))

            if args.show:
                cv2.imshow("live bell circle climb detector", vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("r"):
                    detector = BellCircle(
                        color_format="rgb",
                        dp=args.dp,
                        min_dist=args.min_dist,
                        param1=args.param1,
                        param2=args.param2,
                        min_radius=args.min_radius,
                        startup_max_radius=args.startup_max_radius,
                        tracking_max_radius=args.tracking_max_radius,
                        lost_after_frames=args.lost_after_frames,
                        startup_confirm_threshold=args.startup_confirm_frames,
                        show_debug=args.debug_hough,
                    )
                    print("tracker reset")

            if frame_count % max(1, args.fps) == 0:
                print(status)

    except KeyboardInterrupt:
        print("\nStopping live BellCircle climb test.")
    finally:
        if preview is not None:
            preview.stop()
        if args.show or args.debug_hough:
            cv2.destroyAllWindows()
        camera.release()


if __name__ == "__main__":
    main()
