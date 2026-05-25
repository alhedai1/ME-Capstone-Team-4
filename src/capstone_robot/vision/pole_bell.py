from dataclasses import dataclass

import cv2
import numpy as np


LOWER_BRIGHT = np.array([0, 0, 200])
UPPER_BRIGHT = np.array([180, 100, 255])
LOWER_DARK = np.array([0, 0, 0])
UPPER_DARK = np.array([180, 100, 80])


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
    source: str = "mask"


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


def box_touches_frame_edge(x, y, w, h, frame_shape, edge_margin=35):
    height, width = frame_shape[:2]
    return (
        x <= edge_margin
        or y <= edge_margin
        or x + w >= width - edge_margin
        or y + h >= height - edge_margin
    )


def line_segment_touches_frame_edge(x1, y1, x2, y2, frame_shape, edge_margin=60):
    height, width = frame_shape[:2]
    points = ((x1, y1), (x2, y2))
    return any(
        x <= edge_margin
        or y <= edge_margin
        or x >= width - edge_margin
        or y >= height - edge_margin
        for x, y in points
    )


def get_pole_candidates(
    mask,
    min_area=300,
    min_aspect=1.5,
    require_edge_connection=True,
    edge_margin=35,
):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    candidates = []

    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area < min_area:
            continue

        aspect = max(w, h) / max(1, min(w, h))
        if aspect < min_aspect:
            continue

        if require_edge_connection and not box_touches_frame_edge(x, y, w, h, mask.shape, edge_margin):
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
                source="mask",
            )
        )
    return candidates


def get_hough_pole_candidates(
    frame,
    bell,
    color_format="rgb",
    min_line_length=60,
    max_line_gap=25,
    max_bell_distance_px=80,
    min_vertical_fraction=0.85,
    min_below_bell_px=50,
    require_edge_connection=True,
    edge_margin=60,
):
    gray = gray_from_frame(frame, color_format)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    median = float(np.median(blur))
    lower = max(20, int(0.66 * median))
    upper = min(255, int(1.33 * median))
    edges = cv2.Canny(blur, lower, upper)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=35,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    if lines is None:
        return []

    bx, by, br = bell
    single_edge_candidates = []
    paired_edge_candidates = []
    valid_lines = []

    for x1, y1, x2, y2 in lines[:, 0]:
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = float(np.hypot(dx, dy))
        if length < min_line_length:
            continue

        vertical_fraction = abs(dy) / length
        if vertical_fraction < min_vertical_fraction:
            continue

        distance_to_bell = abs(dx * (y1 - by) - dy * (x1 - bx)) / max(1.0, length)
        if distance_to_bell > max(max_bell_distance_px, br * 2.5):
            continue

        below_bell = max(y1, y2) - by
        if below_bell < max(min_below_bell_px, br * 2):
            continue

        if require_edge_connection and not line_segment_touches_frame_edge(
            x1,
            y1,
            x2,
            y2,
            gray.shape,
            edge_margin=edge_margin,
        ):
            continue

        vx = dx / length
        vy = dy / length
        x0 = (x1 + x2) / 2.0
        y0 = (y1 + y2) / 2.0
        valid_lines.append(
            {
                "points": (int(x1), int(y1), int(x2), int(y2)),
                "line": (vx, vy, x0, y0),
                "length": length,
                "distance_to_bell": distance_to_bell,
                "below_bell": below_bell,
            }
        )

        refined_line, refined = refine_edge_line_to_center((vx, vy, x0, y0), edges)
        rvx, rvy, rx0, ry0 = refined_line

        line_mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.line(
            line_mask,
            (int(rx0 - rvx * length / 2.0), int(ry0 - rvy * length / 2.0)),
            (int(rx0 + rvx * length / 2.0), int(ry0 + rvy * length / 2.0)),
            255,
            9,
        )

        score = length + max(0.0, max_bell_distance_px - distance_to_bell) * 2.0 + below_bell * 0.2
        single_edge_candidates.append(
            PoleCandidate(
                mask=line_mask,
                line=refined_line,
                area=int(score * 0.55),
                aspect=max(1.0, length / 10.0),
                center=(rx0, ry0),
                source="hough_refined" if refined else "hough",
            )
        )

    for idx, line_a in enumerate(valid_lines):
        avx, avy, ax0, ay0 = line_a["line"]
        for line_b in valid_lines[idx + 1 :]:
            bvx, bvy, bx0, by0 = line_b["line"]
            if avx * bvx + avy * bvy < 0:
                bvx, bvy = -bvx, -bvy

            angle_dot = max(-1.0, min(1.0, abs(avx * bvx + avy * bvy)))
            angle_diff = float(np.degrees(np.arccos(angle_dot)))
            if angle_diff > 12:
                continue

            separation = abs(avx * (by0 - ay0) - avy * (bx0 - ax0))
            if separation < 6 or separation > 80:
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

            pair_mask = np.zeros(gray.shape, dtype=np.uint8)
            ax1, ay1, ax2, ay2 = line_a["points"]
            bx1, by1, bx2, by2 = line_b["points"]
            cv2.line(pair_mask, (ax1, ay1), (ax2, ay2), 255, 5)
            cv2.line(pair_mask, (bx1, by1), (bx2, by2), 255, 5)

            length = (line_a["length"] + line_b["length"]) / 2.0
            below_bell = max(line_a["below_bell"], line_b["below_bell"])
            score = length * 1.8 + max(0.0, max_bell_distance_px - distance_to_bell) * 3.0 + below_bell * 0.25
            paired_edge_candidates.append(
                PoleCandidate(
                    mask=pair_mask,
                    line=(vx, vy, x0, y0),
                    area=int(score),
                    aspect=max(1.0, length / 7.0),
                    center=(x0, y0),
                    source="hough_pair",
                )
            )

    return paired_edge_candidates if paired_edge_candidates else single_edge_candidates


def refine_edge_line_to_center(line, edges, max_width_px=80):
    vx, vy, x0, y0 = line
    normal = np.array([-vy, vx], dtype=float)
    direction = np.array([vx, vy], dtype=float)
    height, width = edges.shape

    side_distances = []
    for side in (-1.0, 1.0):
        distances = []
        for t in np.linspace(-45, 45, 7):
            base = np.array([x0, y0], dtype=float) + direction * t
            for distance in range(8, max_width_px + 1):
                pt = base + normal * side * distance
                px = int(round(pt[0]))
                py = int(round(pt[1]))
                if px < 1 or px >= width - 1 or py < 1 or py >= height - 1:
                    continue

                if np.any(edges[py - 1 : py + 2, px - 1 : px + 2] > 0):
                    distances.append(distance)
                    break

        if len(distances) >= 3:
            side_distances.append((len(distances), float(np.median(distances)), side))

    if not side_distances:
        return line, False

    _, separation, side = max(side_distances, key=lambda item: (item[0], -item[1]))
    if separation < 8:
        return line, False

    x0 += normal[0] * side * separation / 2.0
    y0 += normal[1] * side * separation / 2.0
    return (vx, vy, x0, y0), True


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


def pole_candidate_quality(candidate):
    source_bonus = {
        "hough_pair": 4.0,
        "mask": 2.0,
        "hough_refined": 1.6,
        "hough": 1.0,
    }.get(candidate.source, 1.0)
    return candidate.area * candidate.aspect * source_bonus


def pole_candidate_tracking_bonus(candidate):
    return {
        "hough_pair": 120.0,
        "mask": 35.0,
        "hough_refined": 15.0,
        "hough": 0.0,
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
        smooth_alpha=0.45,
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
    bell = detect_bell(frame, color_format=color_format)
    if bell is None:
        return None

    hsv = hsv_from_frame(frame, color_format)
    clean_mask = get_clean_pole_mask(hsv)
    cv2.imshow("clean_mask", clean_mask)
    candidates = get_pole_candidates(clean_mask)
    candidates.extend(get_hough_pole_candidates(frame, bell, color_format=color_format))
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
