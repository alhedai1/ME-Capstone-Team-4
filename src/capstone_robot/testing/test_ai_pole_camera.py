import argparse
import time
from pathlib import Path

import cv2

from capstone_robot.utils import AiCamera, MjpegPreview, resize_preview


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PACKAGE_ROOT / "models" / "pole_imx" / "network.rpk"
DEFAULT_LABELS_PATH = PACKAGE_ROOT / "models" / "pole_imx" / "labels.txt"


def load_labels(path):
    path = Path(path)
    if path is None or not path.exists():
        return None
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def choose_pole(detections, target_label="pole"):
    if target_label:
        poles = [det for det in detections if det.label.lower() == target_label.lower()]
        if poles:
            return max(poles, key=lambda det: det.confidence)

    return max(detections, key=lambda det: det.confidence) if detections else None


def smooth_box(old_box, new_box, alpha):
    if old_box is None:
        return new_box

    return tuple(int(alpha * new + (1.0 - alpha) * old) for old, new in zip(old_box, new_box))


def draw_detection(frame, detections, pole, status, draw_all=False):
    vis = frame.copy()

    if draw_all:
        for det in detections:
            x, y, w, h = det.box
            cv2.rectangle(vis, (x, y), (x + w, y + h), (80, 80, 80), 1)
            cv2.putText(
                vis,
                f"{det.label} {det.confidence:.2f}",
                (x, max(20, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (80, 80, 80),
                1,
                cv2.LINE_AA,
            )

    if pole is not None:
        x, y, w, h = pole.box
        cx = int(x + w / 2)
        cy = int(y + h / 2)
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(vis, (cx, cy), 4, (0, 255, 0), -1)
        cv2.line(vis, (frame.shape[1] // 2, 0), (frame.shape[1] // 2, frame.shape[0]), (255, 0, 0), 1)
        cv2.putText(
            vis,
            f"pole {pole.confidence:.2f}",
            (x, max(25, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(vis, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return vis


def main():
    parser = argparse.ArgumentParser(description="Preview Raspberry Pi AI Camera pole detections without motors.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="IMX500 .rpk model path")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH, help="Labels file path")
    parser.add_argument("--width", type=int, default=640, help="AI camera frame width")
    parser.add_argument("--height", type=int, default=480, help="AI camera frame height")
    parser.add_argument("--fps", type=int, default=30, help="AI camera frame rate")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("--target-label", default="pole", help="Preferred label to track")
    parser.add_argument("--smooth-alpha", type=float, default=0.85, help="Same box smoothing alpha used by states")
    parser.add_argument("--center-deadband", type=float, default=20.0, help="Pole center deadband in pixels")
    parser.add_argument("--stop-width-fraction", type=float, default=0.16, help="Approach stop width fraction")
    parser.add_argument("--draw-all", action="store_true", help="Draw all detections, not only chosen pole")
    parser.add_argument("--preview-width", type=int, default=640, help="MJPEG preview width")
    parser.add_argument("--host", default="0.0.0.0", help="MJPEG preview host")
    parser.add_argument("--port", type=int, default=1236, help="MJPEG preview port")
    parser.add_argument("--jpeg-quality", type=int, default=75, help="MJPEG preview JPEG quality")
    parser.add_argument("--no-preview", action="store_true", help="Disable MJPEG preview server")
    parser.add_argument("--show", action="store_true", help="Also show an OpenCV window if a desktop is available")
    args = parser.parse_args()

    labels = load_labels(args.labels)
    camera = AiCamera(
        model_path=args.model,
        width=args.width,
        height=args.height,
        fps=args.fps,
        bbox_normalization=True,
        bbox_order="xy",
    )

    preview = None
    if not args.no_preview:
        preview = MjpegPreview(host=args.host, port=args.port, jpeg_quality=args.jpeg_quality)
        preview.start()
        print(f"Preview stream: http://<RPI_IP_ADDRESS>:{args.port}")

    smoothed_box = None
    frame_count = 0
    detected_count = 0
    missed_count = 0
    started_at = time.time()

    print("Running AI pole camera test. Press Ctrl+C to stop.")
    if args.show:
        print("OpenCV window enabled. Press q or Esc in the window to stop.")

    try:
        while True:
            loop_started = time.time()
            ok, frame, metadata = camera.read()
            if not ok or frame is None or metadata is None:
                print("[AI POLE TEST] No AI camera frame/metadata")
                time.sleep(0.05)
                continue

            detections = camera.get_detections(
                metadata=metadata,
                labels=labels,
                threshold=args.threshold,
            )
            pole = choose_pole(detections, target_label=args.target_label)
            frame_count += 1

            if pole is None:
                missed_count += 1
                smoothed_box = None
                status = f"AI POLE: NO POLE dets={len(detections)}"
            else:
                detected_count += 1
                smoothed_box = smooth_box(smoothed_box, pole.box, args.smooth_alpha)
                pole.box = smoothed_box

                x, y, w, h = pole.box
                frame_width = frame.shape[1]
                pole_center_x = x + w / 2.0
                error_x = pole_center_x - frame_width / 2.0
                width_fraction = w / frame_width

                if width_fraction >= args.stop_width_fraction:
                    action = "CLOSE/STOP"
                elif abs(error_x) <= args.center_deadband:
                    action = "CENTERED"
                elif error_x < 0:
                    action = "LEFT"
                else:
                    action = "RIGHT"

                status = (
                    f"AI POLE: {action} err={error_x:.1f}px "
                    f"w={width_fraction:.2f} conf={pole.confidence:.2f}"
                )

            elapsed = max(1e-6, time.time() - started_at)
            fps_now = 1.0 / max(1e-6, time.time() - loop_started)
            fps_avg = frame_count / elapsed
            hit_rate = detected_count / max(1, frame_count)
            status = f"{status} fps={fps_now:.1f}/{fps_avg:.1f} hit={hit_rate:.2f}"

            vis = draw_detection(frame, detections, pole, status, draw_all=args.draw_all)

            if preview is not None:
                preview.update(resize_preview(vis, args.preview_width))

            if args.show:
                cv2.imshow("AI pole camera test", vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

            if frame_count % max(1, args.fps) == 0:
                print(status)

    except KeyboardInterrupt:
        print("\nStopping AI pole camera test.")
    finally:
        if preview is not None:
            preview.stop()
        if args.show:
            cv2.destroyAllWindows()
        camera.release()


if __name__ == "__main__":
    main()
