from dataclasses import dataclass

import cv2
import numpy as np

from capstone_robot.vision.bell_circle import BellCircle


@dataclass
class PoleBellAlignment:
    error_px: float
    side: str
    pole_line: tuple
    bell: tuple


@dataclass
class LineCandidate:
    points: tuple
    line: tuple
    length: float
    angle_deg: float
    score: float


def near_border(x, y, width, height, margin=8):
    return x < margin or x > width - margin or y < margin or y > height - margin


def gray_from_frame(frame, color_format):
    if color_format == "rgb":
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    if color_format == "bgr":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Unsupported color format: {color_format}")


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


def line_angle_deg(x1, y1, x2, y2):
    angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
    if angle < 0:
        angle += 180.0
    return angle


def angle_diff_deg(angle_a, angle_b):
    diff = abs(angle_a - angle_b)
    return min(diff, 180.0 - diff)


def line_point_signed_distance(px, py, line):
    vx, vy, x0, y0 = line
    return vx * (py - y0) - vy * (px - x0)


def line_point_distance(px, py, line):
    return abs(line_point_signed_distance(px, py, line))


def horizontal_error_to_line(px, py, line):
    line_x = line_x_at_y(line, py)
    if line_x is None:
        return line_point_signed_distance(px, py, line)
    return px - line_x


def fit_line_from_segments(segments):
    points = []
    for candidate in segments:
        x1, y1, x2, y2 = candidate.points
        points.append((x1, y1))
        points.append((x2, y2))

    if len(points) < 2:
        return None

    points = np.array(points, dtype=np.float32)
    vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
    return float(vx[0]), float(vy[0]), float(x0[0]), float(y0[0])


def orient_line_toward_bell(line, bell_center):
    vx, vy, x0, y0 = line
    bx, by = bell_center

    p_forward = np.array([x0 + vx * 100.0, y0 + vy * 100.0])
    p_backward = np.array([x0 - vx * 100.0, y0 - vy * 100.0])
    bell = np.array([bx, by])

    if np.linalg.norm(p_forward - bell) <= np.linalg.norm(p_backward - bell):
        return vx, vy, x0, y0
    return -vx, -vy, x0, y0


def get_line_candidates(
    frame,
    color_format="rgb",
    border_margin=8,
    min_line_length=40,
    max_line_gap=20,
    hough_threshold=40,
    max_angle_from_vertical_deg=35,
    require_border=True,
):
    gray = gray_from_frame(frame, color_format)
    height, width = gray.shape

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    gray_blur = cv2.GaussianBlur(gray_eq, (5, 5), 0)
    edges = cv2.Canny(gray_blur, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    if lines is None:
        return []

    candidates = []
    for x1, y1, x2, y2 in lines[:, 0]:
        touches_border = (
            near_border(x1, y1, width, height, border_margin)
            or near_border(x2, y2, width, height, border_margin)
        )
        if require_border and not touches_border:
            continue

        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < min_line_length:
            continue

        angle = line_angle_deg(x1, y1, x2, y2)
        vertical_error = angle_diff_deg(angle, 90.0)
        if vertical_error > max_angle_from_vertical_deg:
            continue

        line = make_line_from_points(x1, y1, x2, y2)
        center_x = (x1 + x2) / 2.0
        center_score = max(0.0, width / 2.0 - abs(center_x - width / 2.0))
        score = length * 2.0 + center_score * 0.35 - vertical_error * 3.0
        candidates.append(
            LineCandidate(
                points=(int(x1), int(y1), int(x2), int(y2)),
                line=line,
                length=length,
                angle_deg=angle,
                score=score,
            )
        )

    return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)


def choose_pole_line(candidates, angle_group_deg=12, max_group_lines=8):
    if not candidates:
        return None

    best_group = []
    best_score = -float("inf")

    for candidate in candidates[:16]:
        group = [
            other
            for other in candidates[:24]
            if angle_diff_deg(candidate.angle_deg, other.angle_deg) <= angle_group_deg
        ]
        score = sum(item.score for item in group)
        if score > best_score:
            best_score = score
            best_group = group

    if not best_group:
        return None

    best_group = sorted(best_group, key=lambda item: item.score, reverse=True)[:max_group_lines]
    return fit_line_from_segments(best_group)


def choose_pole_line_from_seed_and_partner(
    border_candidates,
    all_candidates,
    frame_shape,
    min_top_width_px=6,
    min_bottom_width_px=10,
    max_bottom_width_px=110,
    pair_angle_deg=18,
    min_taper_px=2,
    single_edge_center_offset_px=28,
):
    if not border_candidates:
        return None

    height, width = frame_shape[:2]
    best = None
    best_score = -float("inf")

    for seed in border_candidates:
        seed_bottom = line_x_at_y(seed.line, height - 1)
        seed_top = line_x_at_y(seed.line, 0)
        if seed_bottom is None or seed_top is None:
            continue

        for partner in all_candidates:
            if partner is seed:
                continue
            if angle_diff_deg(seed.angle_deg, partner.angle_deg) > pair_angle_deg:
                continue

            partner_bottom = line_x_at_y(partner.line, height - 1)
            partner_top = line_x_at_y(partner.line, 0)
            if partner_bottom is None or partner_top is None:
                continue

            bottom_width = abs(seed_bottom - partner_bottom)
            top_width = abs(seed_top - partner_top)
            if bottom_width < min_bottom_width_px or bottom_width > max_bottom_width_px:
                continue

            if top_width < min_top_width_px:
                continue

            if bottom_width < top_width + min_taper_px:
                continue

            center_bottom = (seed_bottom + partner_bottom) / 2.0
            center_top = (seed_top + partner_top) / 2.0
            center_score = max(0.0, width / 2.0 - abs(center_bottom - width / 2.0))
            taper_score = min(30.0, bottom_width - top_width)
            score = seed.score + partner.score + center_score + taper_score * 2.0
            if score > best_score:
                best_score = score
                best = line_from_x_at_y(center_bottom, height - 1, center_top, 0)

    if best is not None:
        return best

    seed = max(border_candidates, key=lambda item: item.score)
    seed_bottom = line_x_at_y(seed.line, height - 1)
    seed_top = line_x_at_y(seed.line, 0)
    if seed_bottom is None or seed_top is None:
        return seed.line

    inward = 1.0 if seed_bottom < width / 2.0 else -1.0
    return line_from_x_at_y(
        seed_bottom + inward * single_edge_center_offset_px,
        height - 1,
        seed_top + inward * single_edge_center_offset_px,
        0,
    )


class PoleBellTracker2:
    def __init__(
        self,
        color_format="rgb",
        border_margin=8,
        max_angle_from_vertical_deg=35,
        angle_group_deg=12,
        smooth_alpha=0.65,
    ):
        self.color_format = color_format
        self.border_margin = border_margin
        self.max_angle_from_vertical_deg = max_angle_from_vertical_deg
        self.angle_group_deg = angle_group_deg
        self.smooth_alpha = smooth_alpha
        self.previous_line = None
        self.bell_detector = BellCircle(color_format=color_format)

    def reset(self):
        self.previous_line = None

    def detect(self, frame):
        return detect_pole_bell_alignment(frame, tracker=self, color_format=self.color_format)

    def detect_pole_line(self, frame):
        border_candidates = get_line_candidates(
            frame,
            color_format=self.color_format,
            border_margin=self.border_margin,
            max_angle_from_vertical_deg=self.max_angle_from_vertical_deg,
            require_border=True,
        )
        all_candidates = get_line_candidates(
            frame,
            color_format=self.color_format,
            border_margin=self.border_margin,
            max_angle_from_vertical_deg=self.max_angle_from_vertical_deg,
            require_border=False,
        )
        line = choose_pole_line_from_seed_and_partner(
            border_candidates,
            all_candidates,
            frame.shape,
            pair_angle_deg=self.angle_group_deg,
        )
        if line is None:
            return None

        if self.previous_line is None or self.smooth_alpha >= 1.0:
            self.previous_line = line
        else:
            self.previous_line = smooth_line(self.previous_line, line, self.smooth_alpha)
        return self.previous_line

    def detect_bell(self, frame):
        detection = self.bell_detector.detect(frame)
        return None if detection is None else detection.circle


def smooth_line(previous_line, new_line, alpha):
    pvx, pvy, px0, py0 = previous_line
    nvx, nvy, nx0, ny0 = new_line
    if pvx * nvx + pvy * nvy < 0:
        nvx, nvy = -nvx, -nvy

    vx = alpha * nvx + (1.0 - alpha) * pvx
    vy = alpha * nvy + (1.0 - alpha) * pvy
    norm = max(1e-6, float(np.hypot(vx, vy)))
    vx /= norm
    vy /= norm
    x0 = alpha * nx0 + (1.0 - alpha) * px0
    y0 = alpha * ny0 + (1.0 - alpha) * py0
    return vx, vy, x0, y0


def detect_bell(frame, color_format="rgb"):
    detection = BellCircle(color_format=color_format).detect(frame)
    return None if detection is None else detection.circle


def detect_pole_bell_alignment(frame, tracker=None, color_format="rgb"):
    if tracker is None:
        tracker = PoleBellTracker2(color_format=color_format)

    bell = tracker.detect_bell(frame)
    if bell is None:
        return None

    pole_line = tracker.detect_pole_line(frame)
    if pole_line is None:
        return None

    bx, by, _ = bell
    error = horizontal_error_to_line(bx, by, pole_line)
    side = "right" if error > 0 else "left"
    return PoleBellAlignment(error_px=error, side=side, pole_line=pole_line, bell=bell)


# Drop-in name: `from capstone_robot.vision.pole_bell2 import PoleBellTracker`
PoleBellTracker = PoleBellTracker2
