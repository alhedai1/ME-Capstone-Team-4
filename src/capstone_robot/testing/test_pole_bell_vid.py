import cv2
import numpy as np

from capstone_robot.utils import find_repo_root, rotate_frame
from capstone_robot.vision.pole_bell2 import (
    PoleBellTracker,
    angle_diff_deg,
    choose_pole_line,
    detect_bell,
    get_line_candidates,
    gray_from_frame,
    near_border,
)


REPO_ROOT = find_repo_root(__file__)
VID_PATH = REPO_ROOT / "src/capstone_robot/data/videos/may25/may25_align_trim.mp4"

COLOR_FORMAT = "bgr"
ROTATION = "180"
WAIT_MS = 0

BORDER_MARGIN = 8
MIN_LINE_LENGTH = 40
MAX_LINE_GAP = 20
HOUGH_THRESHOLD = 40
MAX_ANGLE_FROM_VERTICAL_DEG = 35
ANGLE_GROUP_DEG = 12


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


def put_label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return out


def to_bgr(img):
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def stack_tiles(tiles, tile_width=320, columns=3):
    resized = []
    for img in tiles:
        scale = tile_width / img.shape[1]
        tile_height = int(img.shape[0] * scale)
        resized.append(cv2.resize(img, (tile_width, tile_height)))

    rows = []
    for i in range(0, len(resized), columns):
        row = resized[i : i + columns]
        while len(row) < columns:
            row.append(np.zeros_like(row[0]))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def line_detector_debug(frame):
    gray = gray_from_frame(frame, COLOR_FORMAT)
    height, width = gray.shape

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray_eq, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    raw_lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=MIN_LINE_LENGTH,
        maxLineGap=MAX_LINE_GAP,
    )

    raw_vis = frame.copy()
    border_vis = frame.copy()
    candidates_vis = frame.copy()
    group_vis = frame.copy()

    raw_count = 0
    border_count = 0
    if raw_lines is not None:
        raw_count = len(raw_lines)
        for x1, y1, x2, y2 in raw_lines[:, 0]:
            cv2.line(raw_vis, (x1, y1), (x2, y2), (0, 255, 255), 1)
            if near_border(x1, y1, width, height, BORDER_MARGIN) or near_border(
                x2,
                y2,
                width,
                height,
                BORDER_MARGIN,
            ):
                border_count += 1
                cv2.line(border_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    candidates = get_line_candidates(
        frame,
        color_format=COLOR_FORMAT,
        border_margin=BORDER_MARGIN,
        min_line_length=MIN_LINE_LENGTH,
        max_line_gap=MAX_LINE_GAP,
        hough_threshold=HOUGH_THRESHOLD,
        max_angle_from_vertical_deg=MAX_ANGLE_FROM_VERTICAL_DEG,
    )

    for rank, candidate in enumerate(candidates[:10], start=1):
        x1, y1, x2, y2 = candidate.points
        color = (0, 255, 0) if rank <= 3 else (0, 165, 255)
        cv2.line(candidates_vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            candidates_vis,
            f"{rank}:{int(candidate.score)}",
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    dominant_group = []
    if candidates:
        seed = candidates[0]
        dominant_group = [
            candidate
            for candidate in candidates
            if angle_diff_deg(seed.angle_deg, candidate.angle_deg) <= ANGLE_GROUP_DEG
        ]
        for candidate in dominant_group[:10]:
            x1, y1, x2, y2 = candidate.points
            cv2.line(group_vis, (x1, y1), (x2, y2), (255, 0, 255), 2)

    pole_line = choose_pole_line(candidates, angle_group_deg=ANGLE_GROUP_DEG)
    if pole_line is not None:
        group_vis = draw_line(group_vis, pole_line, color=(0, 255, 0), thickness=4)

    return {
        "gray_eq": gray_eq,
        "edges": edges,
        "raw_vis": raw_vis,
        "border_vis": border_vis,
        "candidates_vis": candidates_vis,
        "group_vis": group_vis,
        "raw_count": raw_count,
        "border_count": border_count,
        "candidate_count": len(candidates),
        "group_count": len(dominant_group),
        "pole_line": pole_line,
    }


def draw_final(frame, alignment):
    vis = frame.copy()
    if alignment is None:
        cv2.putText(vis, "NO FINAL DETECTION", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return vis

    vis = draw_line(vis, alignment.pole_line, color=(0, 255, 0), thickness=3)
    bx, by, br = alignment.bell
    cv2.circle(vis, (bx, by), br, (255, 0, 0), 2)
    cv2.circle(vis, (bx, by), 3, (0, 0, 255), -1)
    cv2.putText(
        vis,
        f"error={alignment.error_px:.1f}px side={alignment.side}",
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2,
    )
    return vis


tracker = PoleBellTracker(color_format=COLOR_FORMAT)

cap = cv2.VideoCapture(str(VID_PATH))
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VID_PATH}")

idx = 0
while True:
    ret, img = cap.read()
    if not ret:
        break

    idx += 1
    img = rotate_frame(img, ROTATION)

    bell = detect_bell(img, color_format=COLOR_FORMAT)
    debug = line_detector_debug(img)
    alignment = tracker.detect(img)
    final_vis = draw_final(img, alignment)

    bell_vis = img.copy()
    if bell is not None:
        bx, by, br = bell
        cv2.circle(bell_vis, (bx, by), br, (255, 0, 0), 2)
        cv2.circle(bell_vis, (bx, by), 3, (0, 0, 255), -1)

    print(
        f"frame={idx} bell={bell} raw={debug['raw_count']} border={debug['border_count']} "
        f"candidates={debug['candidate_count']} group={debug['group_count']} "
        f"alignment={alignment}"
    )

    tiles = [
        put_label(bell_vis, "1 original + bell_circle"),
        put_label(to_bgr(debug["gray_eq"]), "2 CLAHE gray"),
        put_label(to_bgr(debug["edges"]), "3 Canny 50/150"),
        put_label(debug["raw_vis"], f"4 raw Hough: {debug['raw_count']}"),
        put_label(debug["border_vis"], f"5 border-touching: {debug['border_count']}"),
        put_label(debug["candidates_vis"], f"6 accepted/ranked: {debug['candidate_count']}"),
        put_label(debug["group_vis"], f"7 dominant group: {debug['group_count']}"),
        put_label(final_vis, "8 final polebelltracker2 output"),
        put_label(img, "9 source frame"),
    ]

    cv2.imshow("pole bell 2 debug", stack_tiles(tiles))
    key = cv2.waitKey(WAIT_MS) & 0xFF
    if key in (ord("q"), 27):
        break
    if key == ord("r"):
        tracker.reset()
        print("tracker reset")

cap.release()
cv2.destroyAllWindows()
