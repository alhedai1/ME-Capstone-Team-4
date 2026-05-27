from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

FEATURE_NAMES = [
    "warm_ratio",
    "bright_ratio",
    "edge_ratio",
    "largest_blob_ratio",
    "mean_saturation",
    "mean_value",
    "std_value",
    "p90_value",
    "p10_value",
    "dark_ratio",
    "high_saturation_ratio",
]


@dataclass
class RoiConfig:
    x: float = 0.0
    y: float = 0.0
    w: float = 1.0
    h: float = 1.0


@dataclass
class BellFeatureConfig:
    warm_h_min: int = 5
    warm_h_max: int = 40
    warm_s_min: int = 35
    warm_v_min: int = 40
    bright_v_min: int = 170
    bright_s_max: int = 130
    dark_v_max: int = 55
    high_s_min: int = 120
    canny_low: int = 60
    canny_high: int = 140
    blur_kernel: int = 5
    morph_kernel: int = 9


def config_to_dict(config):
    return asdict(config)


def config_from_dict(values):
    return BellFeatureConfig(**{key: values[key] for key in BellFeatureConfig.__dataclass_fields__ if key in values})


def roi_from_dict(values):
    return RoiConfig(**{key: values[key] for key in RoiConfig.__dataclass_fields__ if key in values})


def image_paths(folder):
    folder = Path(folder)
    return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS)


def clamp_roi(frame_shape, roi_config):
    height, width = frame_shape[:2]
    x1 = int(round(np.clip(roi_config.x, 0.0, 1.0) * width))
    y1 = int(round(np.clip(roi_config.y, 0.0, 1.0) * height))
    x2 = int(round(np.clip(roi_config.x + roi_config.w, 0.0, 1.0) * width))
    y2 = int(round(np.clip(roi_config.y + roi_config.h, 0.0, 1.0) * height))

    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)

    return x1, y1, x2, y2


def extract_roi(frame, roi_config):
    x1, y1, x2, y2 = clamp_roi(frame.shape, roi_config)
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def odd_kernel_size(value):
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def build_feature_masks(frame_bgr, roi_config=None, config=None):
    if roi_config is None:
        roi_config = RoiConfig()
    if config is None:
        config = BellFeatureConfig()

    roi, roi_box = extract_roi(frame_bgr, roi_config)
    blur_size = odd_kernel_size(config.blur_kernel)
    blur = cv2.GaussianBlur(roi, (blur_size, blur_size), 0) if blur_size > 1 else roi
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    warm_mask = cv2.inRange(
        hsv,
        np.array([config.warm_h_min, config.warm_s_min, config.warm_v_min]),
        np.array([config.warm_h_max, 255, 255]),
    )
    bright_mask = cv2.inRange(
        hsv,
        np.array([0, 0, config.bright_v_min]),
        np.array([179, config.bright_s_max, 255]),
    )

    kernel_size = odd_kernel_size(config.morph_kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    warm_clean = cv2.morphologyEx(warm_mask, cv2.MORPH_OPEN, kernel)
    warm_clean = cv2.morphologyEx(warm_clean, cv2.MORPH_CLOSE, kernel)

    warm_nearby = cv2.dilate(warm_clean, kernel, iterations=2)
    bright_near_warm = cv2.bitwise_and(bright_mask, warm_nearby)
    candidate_mask = cv2.bitwise_or(warm_clean, bright_near_warm)
    candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, kernel)

    edges = cv2.Canny(gray, config.canny_low, config.canny_high)
    edges_near_candidate = cv2.bitwise_and(edges, cv2.dilate(candidate_mask, kernel, iterations=1))

    return {
        "roi": roi,
        "roi_box": roi_box,
        "hsv": hsv,
        "gray": gray,
        "warm_mask": warm_clean,
        "bright_mask": bright_near_warm,
        "edge_mask": edges_near_candidate,
        "candidate_mask": candidate_mask,
    }


def largest_blob_ratio(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    roi_area = float(mask.shape[0] * mask.shape[1])
    return cv2.contourArea(max(contours, key=cv2.contourArea)) / roi_area


def extract_bell_features(frame_bgr, roi_config=None, config=None):
    masks = build_feature_masks(frame_bgr, roi_config=roi_config, config=config)
    hsv = masks["hsv"]
    value = hsv[:, :, 2]
    saturation = hsv[:, :, 1]
    roi_area = float(value.shape[0] * value.shape[1])

    features = {
        "warm_ratio": cv2.countNonZero(masks["warm_mask"]) / roi_area,
        "bright_ratio": cv2.countNonZero(masks["bright_mask"]) / roi_area,
        "edge_ratio": cv2.countNonZero(masks["edge_mask"]) / roi_area,
        "largest_blob_ratio": largest_blob_ratio(masks["candidate_mask"]),
        "mean_saturation": float(np.mean(saturation)),
        "mean_value": float(np.mean(value)),
        "std_value": float(np.std(value)),
        "p90_value": float(np.percentile(value, 90)),
        "p10_value": float(np.percentile(value, 10)),
        "dark_ratio": float(np.mean(value <= (config or BellFeatureConfig()).dark_v_max)),
        "high_saturation_ratio": float(np.mean(saturation >= (config or BellFeatureConfig()).high_s_min)),
    }
    return {name: float(features[name]) for name in FEATURE_NAMES}, masks


def feature_vector(features):
    return [features[name] for name in FEATURE_NAMES]


def draw_roi(frame, roi_box, color=(255, 255, 0), thickness=2):
    x1, y1, x2, y2 = roi_box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)


def put_lines(frame, lines, origin=(12, 28), line_height=24, color=(255, 255, 255)):
    x, y = origin
    for index, line in enumerate(lines):
        yy = y + index * line_height
        cv2.putText(frame, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)

