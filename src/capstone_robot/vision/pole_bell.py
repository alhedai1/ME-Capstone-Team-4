from dataclasses import dataclass

import cv2
import numpy as np


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
    source: str = "edge_pair"


def make_line_from_points(x1, y1, x2, y2):
    dx = float(x2 - x1)
    dy = float(y2 - y1)
    length = max(1e-6, float(np.hypot(dx, dy)))
    return dx / length, dy / length, (x1 + x2) / 2.0, (y1 + y2) / 2.0


def line_x_at_y(line, y):
    vx, vy, x0, y0 = line
    if abs(vy) < 1e-6:
        return None
    return x0 + vx * ((y - y0) / vy)


def line_from_x_at_y(x_a, y_a, x_b, y_b):
    return make_line_from_points(float(x_a), float(y_a), float(x_b), float(y_b))


def get_hough_pole_candidates(
    frame,
    bell,
    color_format="rgb",
    min_line_length=70,
    max_line_gap=35,
    max_bell_distance_px=80,
    min_vertical_fraction=0.92,
    bottom_touch_margin=12,
    min_pair_width_px=10,
    max_pair_width_px=120,
    single_edge_center_offset_px=28,
):
    gray = gray_from_frame(frame, color_format)
    height, width = gray.shape

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normalized = clahe.apply(gray)
    blur = cv2.GaussianBlur(normalized, (5, 5), 0)

    median = float(np.median(blur))
    lower = max(15, int(0.55 * median))
    upper = min(255, int(1.45 * median))
    canny = cv2.Canny(blur, lower, upper)

    edges = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=30,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    if lines is None:
        return []

    bx, by, br = bell
    paired_edge_candidates = []
    valid_lines = []
    bottom_y = height - 1
    min_bottom_y = height - 1 - bottom_touch_margin

    for x1, y1, x2, y2 in lines[:, 0]:
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = float(np.hypot(dx, dy))
        if length < min_line_length:
            continue

        vertical_fraction = abs(dy) / length
        if vertical_fraction < min_vertical_fraction:
            continue

        if max(y1, y2) < min_bottom_y:
            continue

        line = make_line_from_points(x1, y1, x2, y2)
        x_bottom = line_x_at_y(line, bottom_y)
        if x_bottom is None or x_bottom < 0 or x_bottom >= width:
            continue

        distance_to_bell = line_point_distance(bx, by, line)
        if distance_to_bell > max(max_bell_distance_px * 2.0, br * 4.0):
            continue

        center_distance = abs(x_bottom - width / 2.0)
        score = length * 3.0 - center_distance * 0.7 - distance_to_bell * 0.3
        valid_lines.append(
            {
                "points": (int(x1), int(y1), int(x2), int(y2)),
                "line": line,
                "length": length,
                "distance_to_bell": distance_to_bell,
                "x_bottom": x_bottom,
                "score": score,
            }
        )

    valid_lines.sort(key=lambda item: item["score"], reverse=True)

    for idx, line_a in enumerate(valid_lines[:12]):
        avx, avy, ax0, ay0 = line_a["line"]
        for line_b in valid_lines[idx + 1 : 12]:
            bvx, bvy, bx0, by0 = line_b["line"]
            if avx * bvx + avy * bvy < 0:
                bvx, bvy = -bvx, -bvy

            angle_dot = max(-1.0, min(1.0, abs(avx * bvx + avy * bvy)))
            angle_diff = float(np.degrees(np.arccos(angle_dot)))
            if angle_diff > 8:
                continue

            bottom_separation = abs(line_a["x_bottom"] - line_b["x_bottom"])
            if bottom_separation < min_pair_width_px or bottom_separation > max_pair_width_px:
                continue

            vx = avx + bvx
            vy = avy + bvy
            norm = max(1e-6, float(np.hypot(vx, vy)))
            vx /= norm
            vy /= norm
            x0 = (ax0 + bx0) / 2.0
            y0 = (ay0 + by0) / 2.0

            distance_to_bell = line_point_distance(bell[0], bell[1], (vx, vy, x0, y0))
            if distance_to_bell > max(max_bell_distance_px, br * 2.5):
                continue

            x_top_a = line_x_at_y(line_a["line"], 0)
            x_top_b = line_x_at_y(line_b["line"], 0)
            if x_top_a is None or x_top_b is None:
                continue

            x_center_bottom = (line_a["x_bottom"] + line_b["x_bottom"]) / 2.0
            x_center_top = (x_top_a + x_top_b) / 2.0
            center_line = line_from_x_at_y(x_center_bottom, bottom_y, x_center_top, 0)

            pair_mask = np.zeros(gray.shape, dtype=np.uint8)
            ax1, ay1, ax2, ay2 = line_a["points"]
            bx1, by1, bx2, by2 = line_b["points"]
            cv2.line(pair_mask, (ax1, ay1), (ax2, ay2), 255, 5)
            cv2.line(pair_mask, (bx1, by1), (bx2, by2), 255, 5)

            length = (line_a["length"] + line_b["length"]) / 2.0
            center_distance = abs(x_center_bottom - width / 2.0)
            score = line_a["score"] + line_b["score"] + length * 2.0 - center_distance
            paired_edge_candidates.append(
                PoleCandidate(
                    mask=pair_mask,
                    line=center_line,
                    area=int(score),
                    aspect=max(1.0, length / 7.0),
                    center=(center_line[2], center_line[3]),
                    source="edge_pair",
                )
            )

    if paired_edge_candidates:
        return sorted(paired_edge_candidates, key=lambda item: item.area, reverse=True)[:6]

    single_edge_candidates = []
    for line_info in valid_lines[:6]:
        vx, vy, x0, y0 = line_info["line"]
        length = line_info["length"]
        line_mask = np.zeros(gray.shape, dtype=np.uint8)
        x1, y1, x2, y2 = line_info["points"]
        cv2.line(line_mask, (x1, y1), (x2, y2), 255, 7)

        x_top = line_x_at_y(line_info["line"], 0)
        if x_top is None:
            continue

        toward_center = 1.0 if line_info["x_bottom"] < width / 2.0 else -1.0
        x_center_bottom = line_info["x_bottom"] + toward_center * single_edge_center_offset_px
        x_center_top = x_top + toward_center * single_edge_center_offset_px
        center_line = line_from_x_at_y(x_center_bottom, bottom_y, x_center_top, 0)

        score = line_info["score"] - 180.0
        single_edge_candidates.append(
            PoleCandidate(
                mask=line_mask,
                line=center_line,
                area=int(score),
                aspect=max(1.0, length / 12.0),
                center=(center_line[2], center_line[3]),
                source="edge_single",
            )
        )
    return single_edge_candidates


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


def pole_candidate_quality(candidate):
    return candidate.area


def pole_candidate_tracking_bonus(candidate):
    return {
        "edge_pair": 120.0,
        "edge_single": 10.0,
    }.get(candidate.source, 0.0)


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
        smooth_alpha=1.0,
        reset_after_misses=8,
        reacquire_on_jump=True,
    ):
        self.previous_line = None
        self.missed_frames = 0
        self.color_format = color_format
        self.max_angle_jump_deg = max_angle_jump_deg
        self.max_line_distance_px = max_line_distance_px
        self.max_center_jump_px = max_center_jump_px
        self.smooth_alpha = smooth_alpha
        self.reset_after_misses = reset_after_misses
        self.reacquire_on_jump = reacquire_on_jump

    def reset(self):
        self.previous_line = None
        self.missed_frames = 0

    def choose_candidate(self, candidates):
        if not candidates:
            return None

        if self.previous_line is None:
            return max(candidates, key=pole_candidate_quality)

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

            score = angle * 3.0 + distance + center_distance * 0.35 - pole_candidate_tracking_bonus(candidate)
            scored.append((score, candidate))

        if not scored:
            if self.reacquire_on_jump:
                self.previous_line = None
                return max(candidates, key=pole_candidate_quality)
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


def gray_from_frame(frame, color_format):
    if color_format == "rgb":
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    if color_format == "bgr":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
    bell = detect_bell(frame, color_format=color_format)
    if bell is None:
        return None

    candidates = get_hough_pole_candidates(frame, bell, color_format=color_format)
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

    bx, by, _ = bell
    line = orient_line_toward_bell(line, candidate.mask, bell_center=(bx, by))
    error = signed_distance_to_line(bx, by, line)
    side = "right" if error > 0 else "left"
    return PoleBellAlignment(error_px=error, side=side, pole_line=line, bell=bell)
