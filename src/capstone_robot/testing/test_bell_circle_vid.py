import argparse

import cv2

from capstone_robot.utils import find_repo_root
from capstone_robot.vision.bell_circle_climb import BellCircle


REPO_ROOT = find_repo_root(__file__)
DEFAULT_VID_PATH = REPO_ROOT / "src/capstone_robot/data/videos/may25/may25_align_trim.mp4"
DEFAULT_VID_PATH = REPO_ROOT / "src/capstone_robot/data/videos/may27/ai_bell_upwards3.mp4"

def draw_detection(img, detection):
    vis = img.copy()
    if detection is None:
        cv2.putText(vis, "No bell", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return vis

    x, y, radius = detection.circle
    cv2.circle(vis, (x, y), radius, (255, 0, 0), 2)
    cv2.circle(vis, (x, y), 2, (0, 255, 0), 3)
    return vis


def main():
    parser = argparse.ArgumentParser(description="Test bell circle detection on a video.")
    parser.add_argument("video", nargs="?", default=DEFAULT_VID_PATH, help="Video path")
    parser.add_argument("--play", action="store_true", help="Play continuously instead of stepping frame by frame")
    args = parser.parse_args()

    detector = BellCircle(color_format="bgr",
        dp=1.5,
        min_dist=5,
        param1=50,
        param2=50,
        min_radius=10,
        max_radius=50,)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    wait_ms = 30 if args.play else 0
    frame_idx = 0
    while True:
        ret, img = cap.read()
        if not ret:
            break

        frame_idx += 1
        detection = detector.detect(img)
        print(f"frame {frame_idx}: {detection.circle if detection else None}")

        cv2.imshow("bell circle", draw_detection(img, detection))
        key = cv2.waitKey(wait_ms) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
