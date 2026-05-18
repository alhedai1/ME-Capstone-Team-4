import cv2
import numpy as np

# CAM_ID = 0

from capstone_robot.utils import find_repo_root

REPO_ROOT = find_repo_root(__file__)

img_path = REPO_ROOT / "src/capstone_robot/data/extracted_frames/may15/test1_trim/frame_000045.jpg"
img_folder = REPO_ROOT / "src/capstone_robot/data/extracted_frames/may15/test1_trim"

# Tune these for your scene
LOWER_POLE = np.array([0, 0, 0])
# UPPER_POLE = np.array([180, 255, 70])
UPPER_POLE = np.array([0, 0, 0])

ALIGN_THRESH_PX = 20


def fit_line_from_mask(mask):
    ys, xs = np.where(mask > 0)

    if len(xs) < 200:
        return None

    pts = np.column_stack((xs, ys)).astype(np.float32)

    vx, vy, x0, y0 = cv2.fitLine(
        pts,
        cv2.DIST_L2,
        0,
        0.01,
        0.01
    )

    return float(vx), float(vy), float(x0), float(y0)

def keep_pole_like_component(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    best_label = None
    best_score = 0

    for label in range(1, num_labels):  # skip background
        x, y, w, h, area = stats[label]

        if area < 300:
            continue

        # Pole should be elongated, not circular like bell
        aspect = max(w, h) / max(1, min(w, h))

        if aspect < 2.5:
            continue

        # Prefer long, large components
        score = area * aspect

        if score > best_score:
            best_score = score
            best_label = label

    if best_label is None:
        return np.zeros_like(mask)

    return np.uint8(labels == best_label) * 255

def signed_distance_to_line(px, py, line):
    vx, vy, x0, y0 = line
    return vx * (py - y0) - vy * (px - x0)


def draw_line(img, line):
    h, w = img.shape[:2]
    vx, vy, x0, y0 = line

    t = 1000
    x1 = int(x0 - vx * t)
    y1 = int(y0 - vy * t)
    x2 = int(x0 + vx * t)
    y2 = int(y0 + vy * t)

    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)


def detect_bell(gray):
    gray = cv2.medianBlur(gray, 5)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)

    # circles = cv2.HoughCircles(
    #     gray,
    #     cv2.HOUGH_GRADIENT,
    #     dp=1.2,
    #     minDist=80,
    #     param1=80,
    #     param2=25,
    #     minRadius=10,
    #     maxRadius=100
    # )
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=150, param2=20, minRadius=0, maxRadius=30)

    if circles is None:
        return None

    circles = np.round(circles[0]).astype(int)

    # For now, choose largest detected circle
    x, y, r = max(circles, key=lambda c: c[2])
    return x, y, r


# cap = cv2.VideoCapture(CAM_ID)

# if not cap.isOpened():
#     raise RuntimeError("Could not open camera")

for img_path in img_folder.iterdir():
    frame = cv2.imread(img_path)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    dark_mask = cv2.inRange(
        hsv,
        np.array([0, 0, 0]),
        np.array([180, 255, 80])
    )

    highlight_mask = cv2.inRange(
        hsv,
        np.array([0, 0, 160]),
        np.array([180, 80, 255])
    )

    # expand dark pole region a bit
    near_dark = cv2.dilate(dark_mask, np.ones((25, 25), np.uint8), iterations=1)

    # only accept highlights close to dark pole
    highlight_near_pole = cv2.bitwise_and(highlight_mask, near_dark)

    pole_mask = cv2.bitwise_or(dark_mask, highlight_near_pole)

    kernel = np.ones((5, 5), np.uint8)
    pole_mask = cv2.morphologyEx(pole_mask, cv2.MORPH_OPEN, kernel)
    pole_mask = cv2.morphologyEx(pole_mask, cv2.MORPH_CLOSE, kernel)

    pole_mask = cv2.erode(pole_mask, np.ones((3, 3), np.uint8), iterations=1)
    pole_mask = keep_pole_like_component(pole_mask)
    pole_mask = cv2.dilate(pole_mask, np.ones((3, 3), np.uint8), iterations=1)

    line = fit_line_from_mask(pole_mask)
    bell = detect_bell(gray)

    status = "NO DETECTION"

    if line is not None:
        draw_line(frame, line)

    if bell is not None:
        bx, by, br = bell
        cv2.circle(frame, (bx, by), br, (255, 0, 0), 2)
        cv2.circle(frame, (bx, by), 3, (255, 0, 0), -1)

    if line is not None and bell is not None:
        bx, by, _ = bell
        error = signed_distance_to_line(bx, by, line)

        if abs(error) < ALIGN_THRESH_PX:
            status = f"ALIGNED error={error:.1f}"
        elif error > 0:
            status = f"BELL SIDE A error={error:.1f}"
        else:
            status = f"BELL SIDE B error={error:.1f}"

    cv2.putText(
        frame,
        status,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2
    )

    cv2.imshow("frame", frame)
    cv2.imshow("pole_mask", pole_mask)

    key = cv2.waitKey(0) & 0xFF
    if key == ord("q"):
        break

# cap.release()
cv2.destroyAllWindows()