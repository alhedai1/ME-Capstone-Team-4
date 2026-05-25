from dataclasses import dataclass

import cv2
import numpy as np


BRASS_HSV_LOW = np.array([8, 45, 50])
BRASS_HSV_HIGH = np.array([34, 255, 255])
BRASS_LAB_LOW = np.array([35, 102, 132])
BRASS_LAB_HIGH = np.array([255, 155, 195])

NEUTRAL_HIGHLIGHT_LOW = np.array([0, 0, 85])
NEUTRAL_HIGHLIGHT_HIGH = np.array([179, 105, 255])
WARM_SHADOW_LAB_LOW = np.array([20, 100, 125])
WARM_SHADOW_LAB_HIGH = np.array([235, 160, 190])


@dataclass
class BellDetection:
    box: tuple
    area: int
    area_fraction: float
    width_fraction: float
    height_fraction: float
    fill: float
    aspect: float
    brass_ratio: float
    warm_brass_ratio: float
    center: tuple
    score: float


@dataclass
class BellCandidate:
    mask: np.ndarray
    box: tuple
    area: int
    area_fraction: float
    width_fraction: float
    height_fraction: float
    fill: float
    aspect: float
    brass_ratio: float
    warm_brass_ratio: float
    center: tuple
    score: float


def hsv_from_frame(frame, color_format):
    if color_format == "rgb":
        return cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    if color_format == "bgr":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    raise ValueError(f"Unsupported color format: {color_format}")


def lab_from_frame(frame, color_format):
    if color_format == "rgb":
        return cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
    if color_format == "bgr":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    raise ValueError(f"Unsupported color format: {color_format}")


def bgr_from_frame(frame, color_format):
    if color_format == "bgr":
        return frame
    if color_format == "rgb":
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    raise ValueError(f"Unsupported color format: {color_format}")


def build_bell_mask(frame, color_format="rgb"):
    hsv = hsv_from_frame(frame, color_format)
    lab = lab_from_frame(frame, color_format)
    bgr = bgr_from_frame(frame, color_format).astype(np.float32)
    blue, green, red = cv2.split(bgr)

    brass_hsv = cv2.inRange(hsv, BRASS_HSV_LOW, BRASS_HSV_HIGH)
    brass_lab = cv2.inRange(lab, BRASS_LAB_LOW, BRASS_LAB_HIGH)
    brass_seed = cv2.bitwise_and(brass_hsv, brass_lab)

    red_dominant = np.uint8((red > green) & (red > blue * 1.15)) * 255
    brass_seed = cv2.bitwise_and(brass_seed, red_dominant)
    warm_brass_seed = cv2.bitwise_and(brass_seed, cv2.inRange(hsv[:, :, 0], 8, 24))

    brass_seed = cv2.morphologyEx(
        brass_seed,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )

    near_brass = cv2.dilate(
        brass_seed,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61)),
        iterations=1,
    )

    neutral_highlights = cv2.inRange(hsv, NEUTRAL_HIGHLIGHT_LOW, NEUTRAL_HIGHLIGHT_HIGH)
    warm_shadows = cv2.inRange(lab, WARM_SHADOW_LAB_LOW, WARM_SHADOW_LAB_HIGH)
    metal_fill = cv2.bitwise_and(cv2.bitwise_or(neutral_highlights, warm_shadows), near_brass)

    mask = cv2.bitwise_or(brass_seed, metal_fill)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35)),
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    return mask, brass_seed, warm_brass_seed


def get_bell_candidates(
    frame,
    color_format="rgb",
    min_area=2500,
    min_area_fraction=0.10,
    min_width_fraction=0.30,
    min_height_fraction=0.25,
    min_fill=0.12,
    min_brass_ratio=0.015,
    min_warm_brass_ratio=0.04,
    max_orange_frame_fraction=None,
):
    del max_orange_frame_fraction

    mask, brass_seed, warm_brass_seed = build_bell_mask(frame, color_format=color_format)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    img_h, img_w = mask.shape
    candidates = []

    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area < min_area or w < 45 or h < 45:
            continue

        area_fraction = area / float(img_w * img_h)
        width_fraction = w / float(img_w)
        height_fraction = h / float(img_h)
        if (
            area_fraction < min_area_fraction
            or width_fraction < min_width_fraction
            or height_fraction < min_height_fraction
        ):
            continue

        fill = area / float(w * h)
        aspect = w / float(h)
        if fill < min_fill or not (0.65 <= aspect <= 4.5):
            continue

        component_mask = np.uint8(labels == label) * 255
        brass_pixels = cv2.countNonZero(cv2.bitwise_and(brass_seed, component_mask))
        brass_ratio = brass_pixels / float(area)
        warm_brass_pixels = cv2.countNonZero(cv2.bitwise_and(warm_brass_seed, component_mask))
        warm_brass_ratio = warm_brass_pixels / float(area)
        if brass_ratio < min_brass_ratio or warm_brass_ratio < min_warm_brass_ratio:
            continue

        cx, cy = centroids[label]
        center_score = 1.0 - min(1.0, abs(cx - img_w * 0.52) / (img_w * 0.65))
        upper_score = 1.0 - min(1.0, max(0.0, cy - img_h * 0.72) / (img_h * 0.28))
        score = area * (0.7 + fill) * (0.65 + center_score) * (0.8 + 0.2 * upper_score)

        candidates.append(
            BellCandidate(
                mask=component_mask,
                box=(int(x), int(y), int(w), int(h)),
                area=int(area),
                area_fraction=float(area_fraction),
                width_fraction=float(width_fraction),
                height_fraction=float(height_fraction),
                fill=float(fill),
                aspect=float(aspect),
                brass_ratio=float(brass_ratio),
                warm_brass_ratio=float(warm_brass_ratio),
                center=(float(cx), float(cy)),
                score=float(score),
            )
        )

    return candidates


def candidate_to_detection(candidate):
    return BellDetection(
        box=candidate.box,
        area=candidate.area,
        area_fraction=candidate.area_fraction,
        width_fraction=candidate.width_fraction,
        height_fraction=candidate.height_fraction,
        fill=candidate.fill,
        aspect=candidate.aspect,
        brass_ratio=candidate.brass_ratio,
        warm_brass_ratio=candidate.warm_brass_ratio,
        center=candidate.center,
        score=candidate.score,
    )


def smooth_box(previous_box, new_box, alpha):
    return tuple(
        int(round(alpha * new_value + (1.0 - alpha) * previous_value))
        for previous_value, new_value in zip(previous_box, new_box)
    )


def smooth_center(previous_center, new_center, alpha):
    return (
        alpha * new_center[0] + (1.0 - alpha) * previous_center[0],
        alpha * new_center[1] + (1.0 - alpha) * previous_center[1],
    )


class BellTracker:
    def __init__(
        self,
        color_format="rgb",
        max_center_jump_px=180,
        max_area_ratio_jump=3.0,
        smooth_alpha=0.55,
        reset_after_misses=8,
        reacquire_on_jump=True,
        min_area_fraction=0.10,
        min_width_fraction=0.30,
        min_height_fraction=0.25,
        max_orange_frame_fraction=None,
    ):
        self.previous_detection = None
        self.missed_frames = 0
        self.color_format = color_format
        self.max_center_jump_px = max_center_jump_px
        self.max_area_ratio_jump = max_area_ratio_jump
        self.smooth_alpha = smooth_alpha
        self.reset_after_misses = reset_after_misses
        self.reacquire_on_jump = reacquire_on_jump
        self.min_area_fraction = min_area_fraction
        self.min_width_fraction = min_width_fraction
        self.min_height_fraction = min_height_fraction
        self.max_orange_frame_fraction = max_orange_frame_fraction

    def reset(self):
        self.previous_detection = None
        self.missed_frames = 0

    def choose_candidate(self, candidates):
        if not candidates:
            return None

        if self.previous_detection is None:
            return max(candidates, key=lambda candidate: candidate.score)

        scored = []
        prev_cx, prev_cy = self.previous_detection.center
        prev_area = max(1.0, float(self.previous_detection.area))

        for candidate in candidates:
            cx, cy = candidate.center
            center_distance = float(np.hypot(cx - prev_cx, cy - prev_cy))
            area_ratio = max(candidate.area / prev_area, prev_area / candidate.area)

            if center_distance > self.max_center_jump_px or area_ratio > self.max_area_ratio_jump:
                continue

            score = center_distance + abs(1.0 - area_ratio) * 40.0 - candidate.score / 50000.0
            scored.append((score, candidate))

        if not scored:
            if self.reacquire_on_jump:
                self.previous_detection = None
                return max(candidates, key=lambda candidate: candidate.score)
            return None

        return min(scored, key=lambda item: item[0])[1]

    def update_detection(self, candidate):
        detection = candidate_to_detection(candidate)
        if self.previous_detection is None:
            self.previous_detection = detection
        else:
            alpha = self.smooth_alpha
            self.previous_detection = BellDetection(
                box=smooth_box(self.previous_detection.box, detection.box, alpha),
                area=int(round(alpha * detection.area + (1.0 - alpha) * self.previous_detection.area)),
                area_fraction=(
                    alpha * detection.area_fraction
                    + (1.0 - alpha) * self.previous_detection.area_fraction
                ),
                width_fraction=(
                    alpha * detection.width_fraction
                    + (1.0 - alpha) * self.previous_detection.width_fraction
                ),
                height_fraction=(
                    alpha * detection.height_fraction
                    + (1.0 - alpha) * self.previous_detection.height_fraction
                ),
                fill=alpha * detection.fill + (1.0 - alpha) * self.previous_detection.fill,
                aspect=alpha * detection.aspect + (1.0 - alpha) * self.previous_detection.aspect,
                brass_ratio=alpha * detection.brass_ratio + (1.0 - alpha) * self.previous_detection.brass_ratio,
                warm_brass_ratio=(
                    alpha * detection.warm_brass_ratio
                    + (1.0 - alpha) * self.previous_detection.warm_brass_ratio
                ),
                center=smooth_center(self.previous_detection.center, detection.center, alpha),
                score=detection.score,
            )

        self.missed_frames = 0
        return self.previous_detection

    def mark_missed(self):
        self.missed_frames += 1
        if self.missed_frames >= self.reset_after_misses:
            self.reset()

    def detect(self, frame):
        candidates = get_bell_candidates(
            frame,
            color_format=self.color_format,
            min_area_fraction=self.min_area_fraction,
            min_width_fraction=self.min_width_fraction,
            min_height_fraction=self.min_height_fraction,
            max_orange_frame_fraction=self.max_orange_frame_fraction,
        )
        candidate = self.choose_candidate(candidates)
        if candidate is None:
            self.mark_missed()
            return None

        return self.update_detection(candidate)


def detect_bell(
    frame,
    color_format="rgb",
    min_area_fraction=0.10,
    min_width_fraction=0.30,
    min_height_fraction=0.25,
    max_orange_frame_fraction=None,
):
    candidates = get_bell_candidates(
        frame,
        color_format=color_format,
        min_area_fraction=min_area_fraction,
        min_width_fraction=min_width_fraction,
        min_height_fraction=min_height_fraction,
        max_orange_frame_fraction=max_orange_frame_fraction,
    )
    candidate = max(candidates, key=lambda item: item.score, default=None)
    if candidate is None:
        return None
    return candidate_to_detection(candidate)
