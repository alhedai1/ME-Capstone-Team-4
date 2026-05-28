import cv2
import numpy as np
import matplotlib.pyplot as plt
from capstone_robot.utils import rotate_frame
from pathlib import Path
import math

# Change this to your saved frame path

IMG_PATH = "../data/extracted_frames/may25/may25_align_trim/frame_000100.jpg"
IMG_FOLDER = "../data/extracted_frames/may25/may25_align_trim"
VID_PATH = "../data/videos/may26/recording.mp4"
VID_PATH = "../data/videos/may25/may25_alignright.mp4"
IMG_PATHS = [path for path in Path(IMG_FOLDER).iterdir()]

def show(img, title="", cmap=None, size=(8, 6)):
    plt.figure(figsize=size)
    if img.ndim == 2:
        plt.imshow(img, cmap=cmap or "gray")
    else:
        # OpenCV uses BGR, matplotlib expects RGB
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()


# show(frame, "Original frame")
# print(frame.shape)

def draw_lines(img, lines):
    line_img = img.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Draw the line on the copied image
            # Parameters: (image, (x1, y1), (x2, y2), (B, G, R), thickness)
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return line_img
# line_img = draw_lines(frame, lines)
# show(line_img)

def near_border(x, y, w, h, margin=5):
    return x < margin or x > w - margin or y < margin or y > h - margin

def line_length(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

def line_angle(x1, y1, x2, y2):
    # angle in degrees, normalized to [0, 180)
    ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
    if ang < 0:
        ang += 180
    if ang >= 180:
        ang -= 180
    return ang

def angle_diff(a, b):
    d = abs(a - b)
    return min(d, 180 - d)


def line_from_points(x1, y1, x2, y2):
    dx = float(x2 - x1)
    dy = float(y2 - y1)
    length = max(1e-6, math.hypot(dx, dy))
    return dx / length, dy / length, (x1 + x2) / 2.0, (y1 + y2) / 2.0


def line_x_at_y(line, y):
    vx, vy, x0, y0 = line
    if abs(vy) < 1e-6:
        return None
    return x0 + vx * ((y - y0) / vy)


def line_from_x_at_y(x_a, y_a, x_b, y_b):
    return line_from_points(x_a, y_a, x_b, y_b)


def draw_full_line(img, line, color=(0, 255, 0), thickness=3):
    out = img.copy()
    vx, vy, x0, y0 = line
    t = 1000
    x1 = int(x0 - vx * t)
    y1 = int(y0 - vy * t)
    x2 = int(x0 + vx * t)
    y2 = int(y0 + vy * t)
    cv2.line(out, (x1, y1), (x2, y2), color, thickness)
    return out


def make_line_candidate(line, w, h, border_margin=5):
    x1, y1, x2, y2 = line[0]
    length = line_length(x1, y1, x2, y2)
    angle = line_angle(x1, y1, x2, y2)
    vertical_error = angle_diff(angle, 90.0)
    touches_border = near_border(x1, y1, w, h, border_margin) or near_border(x2, y2, w, h, border_margin)
    fitted = line_from_points(x1, y1, x2, y2)
    return {
        "raw": line,
        "points": (x1, y1, x2, y2),
        "length": length,
        "angle": angle,
        "vertical_error": vertical_error,
        "touches_border": touches_border,
        "line": fitted,
    }


def choose_pole_centerline(
    all_candidates,
    img_shape,
    min_top_width=6,
    min_bottom_width=10,
    max_bottom_width=110,
    pair_angle_tol=18,
    min_taper_px=2,
    single_offset=28,
):
    h, w = img_shape[:2]
    vertical = [c for c in all_candidates if c["vertical_error"] <= 35 and c["length"] >= 40]
    border_seeds = [c for c in vertical if c["touches_border"]]
    if not border_seeds:
        return None, None, None

    best_pair = None
    best_score = -float("inf")

    for seed in border_seeds:
        seed_bottom = line_x_at_y(seed["line"], h - 1)
        seed_top = line_x_at_y(seed["line"], 0)
        if seed_bottom is None or seed_top is None:
            continue

        for other in vertical:
            if other is seed:
                continue
            if angle_diff(seed["angle"], other["angle"]) > pair_angle_tol:
                continue

            other_bottom = line_x_at_y(other["line"], h - 1)
            other_top = line_x_at_y(other["line"], 0)
            if other_bottom is None or other_top is None:
                continue

            bottom_width = abs(seed_bottom - other_bottom)
            top_width = abs(seed_top - other_top)
            if bottom_width < min_bottom_width or bottom_width > max_bottom_width:
                continue

            if top_width < min_top_width:
                continue

            if bottom_width < top_width + min_taper_px:
                continue

            taper_score = min(30.0, bottom_width - top_width)
            center_bottom = (seed_bottom + other_bottom) / 2.0
            center_score = max(0.0, w / 2.0 - abs(center_bottom - w / 2.0))
            score = seed["length"] + other["length"] + center_score * 0.5 + taper_score * 2.0

            if score > best_score:
                center_top = (seed_top + other_top) / 2.0
                centerline = line_from_x_at_y(center_bottom, h - 1, center_top, 0)
                best_score = score
                best_pair = (centerline, seed, other)

    if best_pair is not None:
        return best_pair

    seed = max(
        border_seeds,
        key=lambda c: c["length"] + max(0.0, w / 2.0 - abs(((c["points"][0] + c["points"][2]) / 2.0) - w / 2.0)) * 0.5,
    )
    seed_bottom = line_x_at_y(seed["line"], h - 1)
    seed_top = line_x_at_y(seed["line"], 0)
    if seed_bottom is None or seed_top is None:
        return None, seed, None

    inward = 1.0 if seed_bottom < w / 2.0 else -1.0
    centerline = line_from_x_at_y(
        seed_bottom + inward * single_offset,
        h - 1,
        seed_top + inward * single_offset,
        0,
    )
    return centerline, seed, None


def average_lines(lines, img_shape):
    """
    Averages multiple line segments into a single line across the image.
    """
    if lines is None or len(lines) == 0:
        return None

    h, w = img_shape[:2]
    rhos = []
    thetas = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1 == x2 and y1 == y2:
            continue # Skip degenerate lines

        angle = line_angle(x1, y1, x2, y2)
        vertical_error = angle_diff(angle, 90.0)
        if vertical_error > 35:
            continue
            
        # Convert segment to polar coordinates (rho, theta)
        # We use standard math because cv2.HoughLinesP outputs segments, not polar representation
        theta = np.arctan2(y2 - y1, x2 - x1) + np.pi / 2
        rho = x1 * np.cos(theta) + y1 * np.sin(theta)
        
        # Ensure theta is bounded consistently to avoid wrapping issues (e.g. close to 0 vs close to pi)
        if rho < 0:
            rho = -rho
            theta -= np.pi

        rhos.append(rho)
        thetas.append(theta)

    if not rhos:
        return None

    # Take the average parameters
    avg_rho = np.mean(rhos)
    avg_theta = np.mean(thetas)

    # Project the average polar line back into pixel coordinates spanning the frame
    a = np.cos(avg_theta)
    b = np.sin(avg_theta)
    x0 = a * avg_rho
    y0 = b * avg_rho

    # Extrapolate points far out so they cross the entire image frame
    x1 = int(x0 + 2000 * (-b))
    y1 = int(y0 + 2000 * (a))
    x2 = int(x0 - 2000 * (-b))
    y2 = int(y0 - 2000 * (a))

    # Clip the line coordinates to the image boundaries safely
    _, p1, p2 = cv2.clipLine((0, 0, w, h), (x1, y1), (x2, y2))
    
    return [[p1[0], p1[1], p2[0], p2[1]]]


def draw_average_line(img, avg_line):
    """
    Draws the calculated average line on the frame in solid green.
    """
    line_img = img.copy()
    if avg_line is not None:
        x1, y1, x2, y2 = avg_line[0]
        # Drawing a thick Green line for the average line
        cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 4)
    return line_img

# cv2.circle(ignore_mask, (bx, by), int(br * 1.5), 0, -1)
def detect_lines(frame):
    # frame = cv2.imread(IMG_PATH)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {IMG_PATH}")
    frame = rotate_frame(frame, "ccw")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    gray_blur = cv2.GaussianBlur(gray_eq, (5, 5), 0)
    edges = cv2.Canny(gray_blur, 50, 150)

    # edges = cv2.bitwise_and(edges, ignore_mask)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=40,
        maxLineGap=20
    )
    line_img = draw_lines(frame, lines)
    h, w, _ = frame.shape
    all_candidates = []
    if lines is not None:
        all_candidates = [make_line_candidate(line, w, h, border_margin=5) for line in lines]

    border_candidates = [candidate for candidate in all_candidates if candidate["touches_border"]]
    centerline, seed, partner = choose_pole_centerline(all_candidates, frame.shape)
    print(
        f"raw lines: {0 if lines is None else len(lines)}, "
        f"border seeds: {len(border_candidates)}, "
        f"paired: {partner is not None}"
    )

    if seed is not None:
        x1, y1, x2, y2 = seed["points"]
        cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 4)

    if partner is not None:
        x1, y1, x2, y2 = partner["points"]
        cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 255), 4)

    if centerline is not None:
        line_img = draw_full_line(line_img, centerline, (0, 255, 255), 4)

    return line_img

# run on image folder
# for path in IMG_PATHS:
#     frame = cv2.imread(path)
#     detect_lines(frame)

# run on vid
cap = cv2.VideoCapture(str(VID_PATH))
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VID_PATH}")

idx = 1
while True:
    print(idx)
    ret, img = cap.read()
    if not ret:
        break

    idx += 1
    # img = rotate_frame(img, ROTATION)
    line_img = detect_lines(img)
    cv2.imshow("pole lines", line_img)
    key = cv2.waitKey(0) & 0xFF
    if key in (ord("q"), 27):
        break
    elif key == ord(" "):
        continue

cap.release()
cv2.destroyAllWindows()


















# other steps from chatgpt
#     # 5. Choose dominant angle by weighted voting
#     # Group lines with similar angle
#     best_group = []
#     best_score = -1

#     for c in candidates:
#         group = [
#             other for other in candidates
#             if angle_diff(c["angle"], other["angle"]) < 12
#         ]
#         score = sum(g["length"] for g in group)

#         if score > best_score:
#             best_score = score
#             best_group = group

#     if len(best_group) == 0:
#         return None, edges, candidates

#     # 6. Fit one centerline to all endpoints in the dominant group
#     pts = []
#     for g in best_group:
#         x1, y1, x2, y2 = g["line"]
#         pts.append([x1, y1])
#         pts.append([x2, y2])

#     pts = np.array(pts, dtype=np.float32)

#     vx, vy, x0, y0 = cv2.fitLine(
#         pts,
#         cv2.DIST_L2,
#         0,
#         0.01,
#         0.01
#     )

#     pole_line = (
#         float(vx),
#         float(vy),
#         float(x0),
#         float(y0),
#     )

#     return pole_line, edges, candidates


# def draw_fit_line(frame, line, color=(0, 255, 0)):
#     out = frame.copy()

#     if line is None:
#         return out

#     vx, vy, x0, y0 = line
#     t = 1000

#     x1 = int(x0 - vx * t)
#     y1 = int(y0 - vy * t)
#     x2 = int(x0 + vx * t)
#     y2 = int(y0 + vy * t)

#     cv2.line(out, (x1, y1), (x2, y2), color, 2)
#     return out
