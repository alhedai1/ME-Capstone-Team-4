import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

from capstone_robot.utils import *


# SOMETHING IS WRONG WITH COMPUTING THE DISTANCE BETWEEN BELL CENTER AND POLE CENTERLINE


### Tuning Parameters ###
# bright/dark region hsv masks
# near_dark dilation
# kernel open/close morphology mask
# area/aspect (pole component)

# Change this to your saved frame path

REPO_ROOT = find_repo_root(__file__)
IMG_PATH = "../data/extracted_frames/may15/test1_trim/frame_000000.jpg"
IMG_FOLDER = REPO_ROOT / "src/capstone_robot/data/extracted_frames/may15/test1_trim"

img_paths = [img_path for img_path in IMG_FOLDER.iterdir()]
# img_paths = [REPO_ROOT / "src/capstone_robot/data/extracted_frames/may15/test1_trim/frame_001170.jpg"]

# bright regions (hit by sunlight)
LOWER_BRIGHT = np.array([0, 0, 200])
UPPER_BRIGHT = np.array([50, 50, 255])
# dark regions
LOWER_DARK = np.array([100, 0, 0])
UPPER_DARK = np.array([200, 100, 60])

def show(figure, img, title="", cmap=None):
    if img.ndim == 2:
        plt.imshow(img, cmap=cmap or "gray")
    else:
        # OpenCV uses BGR, matplotlib expects RGB
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()

def get_clean_pole_mask(hsv):
    pole_mask_dark = cv2.inRange(hsv, LOWER_DARK, UPPER_DARK)
    pole_mask_bright = cv2.inRange(hsv, LOWER_BRIGHT, UPPER_BRIGHT)

    near_dark = cv2.dilate(
        pole_mask_dark,
        np.ones((50, 50), np.uint8),
        iterations=1
    )

    bright_near_dark = cv2.bitwise_and(pole_mask_bright, near_dark)
    pole_mask = cv2.bitwise_or(pole_mask_dark, bright_near_dark)
    # pole_mask = cv2.bitwise_or(pole_mask_dark, pole_mask_bright)

    # clean mask
    kernel_open = np.ones((3, 3), np.uint8)
    kernel_close = np.ones((20, 20), np.uint8)

    clean_mask = cv2.morphologyEx(pole_mask, cv2.MORPH_OPEN, kernel_open)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel_close)
    return clean_mask

def keep_pole_like_component(mask, min_area=300, min_aspect=1.5):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    best_label = None
    best_score = 0

    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]

        if area < min_area:
            continue

        aspect = max(w, h) / max(1, min(w, h))

        if aspect < min_aspect:
            continue

        score = area * aspect
        # score = aspect

        if score > best_score:
            best_score = score
            best_label = label

    if best_label is None:
        return np.zeros_like(mask)

    return np.uint8(labels == best_label) * 255

def fit_line_from_mask(mask, min_points=100):
    # fit a centerline to the pole_only mask
    ys, xs = np.where(mask > 0)

    if len(xs) < min_points:
        return None

    points = np.column_stack((xs, ys)).astype(np.float32)

    vx, vy, x0, y0 = cv2.fitLine(
        points,
        cv2.DIST_L2,
        0,
        0.01,
        0.01
    )

    return float(vx[0]), float(vy[0]), float(x0[0]), float(y0[0])


def draw_line(img, line, color=(0, 255, 0), thickness=2):
    # draw pole centerline
    out = img.copy()

    vx, vy, x0, y0 = line
    t = 1000

    x1 = int(x0 - vx * t)
    y1 = int(y0 - vy * t)
    x2 = int(x0 + vx * t)
    y2 = int(y0 + vy * t)

    cv2.line(out, (x1, y1), (x2, y2), color, thickness)
    return out

def detect_bell(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=60,
        param1=300,
        param2=20,
        minRadius=10,
        maxRadius=20
    )
    # circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=150, param2=20, minRadius=0, maxRadius=30)

    if circles is None:
        return None, []

    circles = np.round(circles[0]).astype(int)

    # Pick largest circle for now
    best = max(circles, key=lambda c: c[2])
    return tuple(best), circles

def orient_line_toward_bell(line, mask, bell_center):
    vx, vy, x0, y0 = line
    bx, by = bell_center

    ys, xs = np.where(mask > 0)
    pts = np.column_stack((xs, ys)).astype(np.float32)

    # Projection of mask points onto the fitted line
    proj = (pts[:, 0] - x0) * vx + (pts[:, 1] - y0) * vy

    t_min = np.min(proj)
    t_max = np.max(proj)

    # These endpoints are ON the fitted centerline
    p_min = np.array([x0 + t_min * vx, y0 + t_min * vy])
    p_max = np.array([x0 + t_max * vx, y0 + t_max * vy])

    bell = np.array([bx, by])

    # The endpoint closer to the bell should be the "top" direction
    d_min = np.linalg.norm(p_min - bell)
    d_max = np.linalg.norm(p_max - bell)

    if d_max < d_min:
        # current direction points toward bell
        return vx, vy, x0, y0
    else:
        # flip direction, but keep same point on line
        return -vx, -vy, x0, y0

def signed_distance_to_line(px, py, line):
    vx, vy, x0, y0 = line
    return vx * (py - y0) - vy * (px - x0)

def visualize_line(img: np.ndarray):
    vis = img.copy()
    vis = draw_line(vis, line)
    return vis

fig = plt.figure(figsize=(8, 6))

# img_path = IMG_PATH
for img_path in img_paths:
    print(img_path)
    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    clean_mask = get_clean_pole_mask(hsv)
    pole_only = keep_pole_like_component(clean_mask)
    line = fit_line_from_mask(pole_only)

    # vis = img.copy()
    # if line is not None:
    #     vis = draw_line(vis, line)
    # else:
    #     print("No line detected")
    bell, circles = detect_bell(img)
    # vis2 = img.copy()
    # for x, y, r in circles:
    #     cv2.circle(vis2, (x, y), r, (255, 0, 0), 1)
    # if bell is not None:
    #     bx, by, br = bell
    #     cv2.circle(vis2, (bx, by), br, (0, 0, 255), 3)
    # else:
    #     print("No bell detected")

    ALIGN_THRESH_PX = 20
    vis = img.copy()
    if line is not None:
        vis = draw_line(vis, line)
    if bell is not None:
        bx, by, br = bell
        cv2.circle(vis, (bx, by), br, (0, 0, 255), 2)
        # cv2.circle(vis, (bx, by), 3, (0, 0, 255), -1)
    if line is not None and bell is not None:
        bx, by, _ = bell
        line = orient_line_toward_bell(
            line,
            pole_only,
            bell_center=(bx, by)
        )
        error = signed_distance_to_line(bx, by, line)

        if abs(error) < ALIGN_THRESH_PX:
            status = f"ALIGNED, error={error:.1f}"
        elif error > 0:
            status = f"RIGHT SIDE, error={error:.1f}"
        else:
            status = f"LEFT SIDE, error={error:.1f}"
        
        vx, vy, x0, y0 = line
        cv2.putText(vis, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        show(fig, vis)
    else:
        print("Need both pole line and bell detection")
