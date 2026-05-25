import cv2
import numpy as np
import matplotlib.pyplot as plt
from capstone_robot.utils import rotate_frame
from pathlib import Path

# Change this to your saved frame path

IMG_PATH = "../data/extracted_frames/may25/may25_align_trim/frame_000100.jpg"
IMG_FOLDER = "../data/extracted_frames/may25/may25_align_trim"
VID_PATH = "../data/videos/may26/recording.mp4"
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

# def line_length(x1, y1, x2, y2):
#     return math.hypot(x2 - x1, y2 - y1)

# def line_angle(x1, y1, x2, y2):
#     # angle in degrees, normalized to [0, 180)
#     ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
#     if ang < 0:
#         ang += 180
#     if ang >= 180:
#         ang -= 180
#     return ang

# def angle_diff(a, b):
#     d = abs(a - b)
#     return min(d, 180 - d)

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
    frame = rotate_frame(frame, "180")

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

    h, w, _ = frame.shape
    candidates = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        touches_border = near_border(x1, y1, w, h) or near_border(x2, y2, w, h)
        if not touches_border:
            continue
        # check length
        # get angle
        candidates.append(line)
    print(f"number of lines: {len(candidates)}")
    
    # show(draw_lines(frame, candidates))
    line_img = draw_lines(frame, candidates)
    # cv2.imshow("line image", line_img)
    # cv2.waitKey(0)

    avg_line = average_lines(candidates, frame.shape)
    if avg_line is not None:
        line_img = draw_average_line(line_img, avg_line)

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