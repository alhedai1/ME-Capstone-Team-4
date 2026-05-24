from dataclasses import dataclass

import cv2
import numpy as np


LOWER_BRIGHT = np.array([0, 0, 200])
UPPER_BRIGHT = np.array([180, 100, 255])
LOWER_DARK = np.array([00, 0, 0])
UPPER_DARK = np.array([180, 100, 50])


@dataclass
class PoleBellAlignment:
    error_px: float
    side: str
    pole_line: tuple
    bell: tuple


@dataclass
class PoleCandidate:
    mask: np.ndarray
    line: tuple
    area: int
    aspect: float
    center: tuple


def get_clean_pole_mask(hsv):
    pole_mask_dark = cv2.inRange(hsv, LOWER_DARK, UPPER_DARK)
    pole_mask_bright = cv2.inRange(hsv, LOWER_BRIGHT, UPPER_BRIGHT)

    near_dark = cv2.dilate(
        pole_mask_dark,
        np.ones((50, 50), np.uint8),
        iterations=1,
    )
    bright_near_dark = cv2.bitwise_and(pole_mask_bright, near_dark)
    pole_mask = cv2.bitwise_or(pole_mask_dark, bright_near_dark)

    clean_mask = cv2.morphologyEx(pole_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, np.ones((20, 20), np.uint8))
    return clean_mask


def get_pole_candidates(mask, min_area=300, min_aspect=1.5):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    candidates = []

    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area < min_area:
            continue

        aspect = max(w, h) / max(1, min(w, h))
        if aspect < min_aspect:
            continue

        component_mask = np.uint8(labels == label) * 255
        line = fit_line_from_mask(component_mask)
        if line is None:
            continue

        candidates.append(
            PoleCandidate(
                mask=component_mask,
                line=line,
                area=int(area),
                aspect=float(aspect),
                center=(x + w / 2.0, y + h / 2.0),
            )
        )
    return candidates


def keep_pole_like_component(mask, min_area=300, min_aspect=1.5):
    candidates = get_pole_candidates(mask, min_area=min_area, min_aspect=min_aspect)
    if not candidates:
        return np.zeros_like(mask)

    best = max(candidates, key=lambda candidate: candidate.area * candidate.aspect)
    return best.mask


def line_angle_diff_deg(line_a, line_b):
    va = np.array(line_a[:2], dtype=float)
    vb = np.array(line_b[:2], dtype=float)
    dot = abs(float(np.dot(va, vb)))
    dot = max(-1.0, min(1.0, dot))
    return float(np.degrees(np.arccos(dot)))


def line_point_distance(px, py, line):
    vx, vy, x0, y0 = line
    return abs(vx * (py - y0) - vy * (px - x0))


def line_center_distance(line_a, line_b):
    return float(np.hypot(line_a[2] - line_b[2], line_a[3] - line_b[3]))


def align_line_direction(line, reference_line):
    vx, vy, x0, y0 = line
    rvx, rvy, _, _ = reference_line
    if vx * rvx + vy * rvy < 0:
        return -vx, -vy, x0, y0
    return line


def smooth_line(previous_line, new_line, alpha):
    new_line = align_line_direction(new_line, previous_line)
    vx = alpha * new_line[0] + (1.0 - alpha) * previous_line[0]
    vy = alpha * new_line[1] + (1.0 - alpha) * previous_line[1]
    norm = max(1e-6, float(np.hypot(vx, vy)))
    vx /= norm
    vy /= norm
    x0 = alpha * new_line[2] + (1.0 - alpha) * previous_line[2]
    y0 = alpha * new_line[3] + (1.0 - alpha) * previous_line[3]
    return vx, vy, x0, y0


class PoleBellTracker:
    def __init__(
        self,
        color_format="rgb",
        max_angle_jump_deg=35,
        max_line_distance_px=120,
        max_center_jump_px=180,
        smooth_alpha=0.45,
        reset_after_misses=8,
    ):
        self.previous_line = None
        self.missed_frames = 0
        self.color_format = color_format
        self.max_angle_jump_deg = max_angle_jump_deg
        self.max_line_distance_px = max_line_distance_px
        self.max_center_jump_px = max_center_jump_px
        self.smooth_alpha = smooth_alpha
        self.reset_after_misses = reset_after_misses

    def reset(self):
        self.previous_line = None
        self.missed_frames = 0

    def choose_candidate(self, candidates):
        if not candidates:
            return None

        if self.previous_line is None:
            return max(candidates, key=lambda candidate: candidate.area * candidate.aspect)

        scored = []
        for candidate in candidates:
            angle = line_angle_diff_deg(candidate.line, self.previous_line)
            distance = line_point_distance(candidate.line[2], candidate.line[3], self.previous_line)
            center_distance = line_center_distance(candidate.line, self.previous_line)

            if (
                angle > self.max_angle_jump_deg
                or distance > self.max_line_distance_px
                or center_distance > self.max_center_jump_px
            ):
                continue

            score = angle * 3.0 + distance + center_distance * 0.35
            scored.append((score, candidate))

        if not scored:
            return None

        return min(scored, key=lambda item: item[0])[1]

    def update_line(self, line):
        if self.previous_line is None:
            self.previous_line = line
        else:
            self.previous_line = smooth_line(self.previous_line, line, self.smooth_alpha)

        self.missed_frames = 0
        return self.previous_line

    def mark_missed(self):
        self.missed_frames += 1
        if self.missed_frames >= self.reset_after_misses:
            self.reset()

    def detect(self, frame):
        return detect_pole_bell_alignment(frame, tracker=self, color_format=self.color_format)


def fit_line_from_mask(mask, min_points=100):
    ys, xs = np.where(mask > 0)
    if len(xs) < min_points:
        return None

    points = np.column_stack((xs, ys)).astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
    return float(vx[0]), float(vy[0]), float(x0[0]), float(y0[0])


def gray_from_frame(frame, color_format):
    if color_format == "rgb":
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    if color_format == "bgr":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Unsupported color format: {color_format}")


def hsv_from_frame(frame, color_format):
    if color_format == "rgb":
        return cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    if color_format == "bgr":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    raise ValueError(f"Unsupported color format: {color_format}")


def detect_bell(frame, color_format="rgb"):
    gray = gray_from_frame(frame, color_format)
    blur = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=60,
        param1=300,
        param2=20,
        minRadius=10,
        maxRadius=30,
    )

    if circles is None:
        return None

    circles = np.round(circles[0]).astype(int)
    return tuple(max(circles, key=lambda c: c[2]))


def orient_line_toward_bell(line, mask, bell_center):
    vx, vy, x0, y0 = line
    bx, by = bell_center

    ys, xs = np.where(mask > 0)
    pts = np.column_stack((xs, ys)).astype(np.float32)
    proj = (pts[:, 0] - x0) * vx + (pts[:, 1] - y0) * vy

    t_min = np.min(proj)
    t_max = np.max(proj)
    p_min = np.array([x0 + t_min * vx, y0 + t_min * vy])
    p_max = np.array([x0 + t_max * vx, y0 + t_max * vy])
    bell = np.array([bx, by])

    if np.linalg.norm(p_max - bell) < np.linalg.norm(p_min - bell):
        return vx, vy, x0, y0
    return -vx, -vy, x0, y0


def signed_distance_to_line(px, py, line):
    vx, vy, x0, y0 = line
    return vx * (py - y0) - vy * (px - x0)


def detect_pole_bell_alignment(frame, tracker=None, color_format="rgb"):
    hsv = hsv_from_frame(frame, color_format)
    clean_mask = get_clean_pole_mask(hsv)
    candidates = get_pole_candidates(clean_mask)
    candidate = (
        tracker.choose_candidate(candidates)
        if tracker is not None
        else max(candidates, key=lambda item: item.area * item.aspect, default=None)
    )
    if candidate is None:
        if tracker is not None:
            tracker.mark_missed()
        return None

    line = candidate.line
    if tracker is not None:
        line = tracker.update_line(line)

    bell = detect_bell(frame, color_format=color_format)
    if bell is None:
        return None

    bx, by, _ = bell
    line = orient_line_toward_bell(line, candidate.mask, bell_center=(bx, by))
    error = signed_distance_to_line(bx, by, line)
    side = "right" if error > 0 else "left"
    return PoleBellAlignment(error_px=error, side=side, pole_line=line, bell=bell)
