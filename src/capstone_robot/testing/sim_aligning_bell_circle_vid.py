import argparse
import csv
import time

import cv2

from capstone_robot.utils import find_repo_root, rotate_frame
from capstone_robot.vision.pole_bell2 import PoleBellTracker


REPO_ROOT = find_repo_root(__file__)
# DEFAULT_VIDEO = REPO_ROOT / "src/capstone_robot/data/videos/may25/may25_align_trim.mp4"
DEFAULT_VIDEO = REPO_ROOT / "src/capstone_robot/data/videos/may25/may25_alignright.mp4"


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


def draw_alignment(frame, alignment, threshold):
    vis = frame.copy()
    if alignment is None:
        cv2.putText(vis, "NO POLE/BELL", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return vis

    vis = draw_line(vis, alignment.pole_line, (0, 255, 0), 3)
    bx, by, br = alignment.bell
    cv2.circle(vis, (bx, by), br, (255, 0, 0), 2)
    cv2.circle(vis, (bx, by), 3, (0, 0, 255), -1)

    aligned = abs(alignment.error_px) <= threshold
    status = "ALIGNED" if aligned else alignment.side.upper()
    color = (0, 255, 0) if aligned else (0, 165, 255)
    cv2.putText(
        vis,
        f"{status} error={alignment.error_px:.1f}px",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        color,
        2,
    )
    return vis


def orbit_seconds_from_error(error_px, px_per_second, min_seconds, max_seconds):
    seconds = abs(error_px) / max(1.0, px_per_second)
    return max(min_seconds, min(max_seconds, seconds))


def normalize_rotation(rotation):
    if rotation == "90cw":
        return "cw"
    if rotation == "90ccw":
        return "ccw"
    return rotation


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simulate pole/bell alignment on a saved upward-camera video."
    )
    parser.add_argument("--video", type=str, default=str(DEFAULT_VIDEO))
    parser.add_argument("--rotation", default="180", choices=["none", "90cw", "90ccw", "180", "cw", "ccw"])
    parser.add_argument("--color-format", default="bgr", choices=["bgr", "rgb"])
    parser.add_argument("--threshold", type=float, default=20.0)
    parser.add_argument("--stable-frames", type=int, default=4)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--wait", type=int, default=0, help="cv2.waitKey delay when --show is used.")
    parser.add_argument("--csv", type=str, default=None, help="Optional CSV output path.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--px-per-second", type=float, default=80.0)
    parser.add_argument("--min-seconds", type=float, default=0.2)
    parser.add_argument("--max-seconds", type=float, default=1.2)
    return parser.parse_args()


def main():
    args = parse_args()
    tracker = PoleBellTracker(color_format=args.color_format)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    csv_file = open(args.csv, "w", newline="") if args.csv else None
    writer = None
    if csv_file is not None:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "frame",
                "time_s",
                "detected",
                "error_px",
                "side",
                "aligned",
                "suggested_orbit_seconds",
                "bell_x",
                "bell_y",
                "bell_radius",
            ]
        )

    stable_count = 0
    detected_count = 0
    aligned_count = 0
    frame_idx = 0
    started_at = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if args.max_frames is not None and frame_idx >= args.max_frames:
                break
            frame_idx += 1

            rotation = normalize_rotation(args.rotation)
            if rotation != "none":
                frame = rotate_frame(frame, rotation)

            alignment = tracker.detect(frame)
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            if alignment is None:
                stable_count = 0
                print(f"frame={frame_idx:04d} t={timestamp:.2f}s detected=0")
                if writer is not None:
                    writer.writerow([frame_idx, timestamp, 0, "", "", 0, "", "", "", ""])
            else:
                detected_count += 1
                aligned = abs(alignment.error_px) <= args.threshold
                stable_count = stable_count + 1 if aligned else 0
                aligned_count += int(aligned)
                suggested_seconds = orbit_seconds_from_error(
                    alignment.error_px,
                    args.px_per_second,
                    args.min_seconds,
                    args.max_seconds,
                )
                bx, by, br = alignment.bell

                print(
                    f"frame={frame_idx:04d} t={timestamp:.2f}s detected=1 "
                    f"error={alignment.error_px:.1f}px side={alignment.side} "
                    f"aligned={aligned} stable={stable_count}/{args.stable_frames} "
                    f"orbit_seconds={suggested_seconds:.2f}"
                )

                if writer is not None:
                    writer.writerow(
                        [
                            frame_idx,
                            f"{timestamp:.3f}",
                            1,
                            f"{alignment.error_px:.3f}",
                            alignment.side,
                            int(aligned),
                            f"{suggested_seconds:.3f}",
                            bx,
                            by,
                            br,
                        ]
                    )

            if args.show:
                cv2.imshow("pole bell alignment simulation", draw_alignment(frame, alignment, args.threshold))
                key = cv2.waitKey(args.wait) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("r"):
                    tracker.reset()
                    stable_count = 0
                    print("tracker reset")

        elapsed = time.time() - started_at
        print(
            f"summary frames={frame_idx} detected={detected_count} "
            f"aligned={aligned_count} elapsed={elapsed:.1f}s"
        )
    finally:
        cap.release()
        if csv_file is not None:
            csv_file.close()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
