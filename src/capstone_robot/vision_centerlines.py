import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CenterlineConfig:
    conf: float = 0.35
    imgsz: int = 640
    roi_pad: float = 0.25
    full_frame_fallback: bool = False
    resize_width: int = 0
    blur: int = 5
    dark_threshold: int = 95
    dark_max_saturation: int = 120
    very_dark_threshold: int = 55
    pole_min_dark_support: float = 0.10
    rod_min_dark_support: float = 0.08
    canny_low: int = 45
    canny_high: int = 130
    hough_threshold: int = 20
    min_line_length: int = 25
    max_line_gap: int = 20
    pole_min_length_ratio: float = 0.12
    pole_angle_tolerance: float = 35.0
    pole_min_width: float = 10.0
    pole_max_width_ratio: float = 0.28
    rod_min_length: int = 20
    rod_search_radius_ratio: float = 0.28
    rod_min_pole_angle: float = 25.0


def process_frame(frame, model, config):
    frame = resize_frame(frame, config.resize_width)
    detections = run_yolo(model, frame, config)
    pole_detection = best_detection(detections, "pole")
    rod_detection = best_detection(detections, "rod")

    pole_box = pole_detection["box"] if pole_detection is not None else None
    rod_box = rod_detection["box"] if rod_detection is not None else None
    pole_roi = expand_box(pole_box, config.roi_pad, frame.shape) if pole_box is not None else None
    rod_roi = expand_box(rod_box, config.roi_pad, frame.shape) if rod_box is not None else None
    yolo_enabled = model is not None

    edges = build_edges(frame, config)
    dark_mask = build_dark_mask(frame, config)
    segments = hough_segments(edges, config)

    pole_result = None
    if pole_roi is not None or not yolo_enabled or config.full_frame_fallback:
        pole_result = detect_pole_centerline(
            segments,
            frame.shape,
            dark_mask,
            config,
            pole_roi=pole_roi,
            pole_box=pole_box,
        )

    rod_result = None
    # if rod_roi is not None or not yolo_enabled or config.full_frame_fallback:
    #     rod_result = detect_rod_centerline(
    #         segments,
    #         pole_result,
    #         frame.shape,
    #         dark_mask,
    #         config,
    #         rod_roi=rod_roi,
    #         rod_box=rod_box,
    #         pole_roi=pole_roi,
    #     )

    annotated = draw_detection(frame, pole_result, rod_result, edges, dark_mask, detections)
    return annotated, edges, dark_mask


def resize_frame(frame, target_width):
    if target_width <= 0 or frame.shape[1] == target_width:
        return frame
    scale = target_width / frame.shape[1]
    target_height = max(1, int(round(frame.shape[0] * scale)))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def build_edges(frame, config):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    if config.blur > 0:
        kernel = odd_kernel(config.blur)
        gray = cv2.GaussianBlur(gray, (kernel, kernel), 0)

    edges = cv2.Canny(gray, config.canny_low, config.canny_high)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kernel, iterations=1)


def build_dark_mask(frame, config):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    if config.blur > 0:
        kernel = odd_kernel(config.blur)
        saturation = cv2.GaussianBlur(saturation, (kernel, kernel), 0)
        value = cv2.GaussianBlur(value, (kernel, kernel), 0)

    low_value = cv2.inRange(value, 0, config.dark_threshold)
    low_saturation = cv2.inRange(saturation, 0, config.dark_max_saturation)
    very_dark = cv2.inRange(value, 0, config.very_dark_threshold)
    dark_mask = cv2.bitwise_or(cv2.bitwise_and(low_value, low_saturation), very_dark)

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    return cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)


def odd_kernel(value):
    return value if value % 2 == 1 else value + 1


def run_yolo(model, frame, config):
    if model is None:
        return []

    result = model(frame, imgsz=config.imgsz, conf=config.conf, verbose=False)[0]
    if result.boxes is None:
        return []

    detections = []
    for box in result.boxes:
        cls_id = int(box.cls.item())
        name = str(result.names.get(cls_id, cls_id)).lower()
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append(
            {
                "class_id": cls_id,
                "name": name,
                "conf": float(box.conf.item()),
                "box": clamp_box((x1, y1, x2, y2), frame.shape),
            }
        )
    return detections


def best_detection(detections, class_name):
    matches = [det for det in detections if det["name"] == class_name]
    return max(matches, key=lambda det: det["conf"]) if matches else None


def hough_segments(edges, config):
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=config.hough_threshold,
        minLineLength=config.min_line_length,
        maxLineGap=config.max_line_gap,
    )
    if lines is None:
        return []

    segments = []
    for raw in lines[:, 0, :]:
        segment = segment_from_hough(raw)
        if segment is not None:
            segments.append(segment)
    return segments


def segment_from_hough(raw):
    x1, y1, x2, y2 = [float(v) for v in raw]
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length <= 1e-6:
        return None

    return {
        "p1": np.array([x1, y1], dtype=np.float32),
        "p2": np.array([x2, y2], dtype=np.float32),
        "center": np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32),
        "length": length,
        "angle": math.degrees(math.atan2(dy, dx)) % 180.0,
    }


def detect_pole_centerline(segments, frame_shape, dark_mask, config, pole_roi=None, pole_box=None):
    search_segments = filter_segments_by_box(segments, pole_roi)
    best = None
    for i, seg_a in enumerate(search_segments):
        for seg_b in search_segments[i + 1 :]:
            candidate = score_pole_pair(seg_a, seg_b, frame_shape, dark_mask, config)
            if candidate is not None and (best is None or candidate["score"] > best["score"]):
                candidate["source"] = "opencv_in_yolo_roi" if pole_roi is not None else "opencv_full_frame"
                best = candidate
    return best or fallback_pole_from_box(pole_box)


def score_pole_pair(seg_a, seg_b, frame_shape, dark_mask, config):
    height, width = frame_shape[:2]
    diagonal = math.hypot(width, height)

    if min(seg_a["length"], seg_b["length"]) < config.pole_min_length_ratio * diagonal:
        return None

    angle_gap = angle_diff_deg(seg_a["angle"], seg_b["angle"])
    if angle_gap > config.pole_angle_tolerance:
        return None

    direction = averaged_direction(seg_a, seg_b)
    if direction is None:
        return None

    normal = np.array([-direction[1], direction[0]], dtype=np.float32)
    side_distance = abs(float(np.dot(seg_a["center"] - seg_b["center"], normal)))
    if side_distance < config.pole_min_width or side_distance > config.pole_max_width_ratio * diagonal:
        return None

    endpoints = paired_edge_endpoints(seg_a, seg_b, direction, config.pole_min_length_ratio * diagonal * 0.45)
    if endpoints is None:
        return None
    start_a, start_b, end_a, end_b = endpoints

    start_width = float(np.linalg.norm(start_a - start_b))
    end_width = float(np.linalg.norm(end_a - end_b))
    if start_width < config.pole_min_width or end_width < config.pole_min_width:
        return None

    side_thickness = max(5.0, min(start_width, end_width) * 0.16)
    side_dark = (
        line_mask_ratio(dark_mask, start_a, end_a, side_thickness)
        + line_mask_ratio(dark_mask, start_b, end_b, side_thickness)
    ) / 2.0
    body_dark = polygon_mask_ratio(dark_mask, [start_a, start_b, end_b, end_a])
    if side_dark < config.pole_min_dark_support and body_dark < config.pole_min_dark_support * 0.55:
        return None

    if start_width <= end_width:
        top, bottom = (start_a + start_b) / 2.0, (end_a + end_b) / 2.0
        top_width, bottom_width = start_width, end_width
    else:
        top, bottom = (end_a + end_b) / 2.0, (start_a + start_b) / 2.0
        top_width, bottom_width = end_width, start_width

    center_length = float(np.linalg.norm(top - bottom))
    if center_length < config.pole_min_length_ratio * diagonal * 0.45:
        return None

    taper_bonus = max(0.0, bottom_width - top_width)
    score = (
        center_length
        + 0.35 * (seg_a["length"] + seg_b["length"])
        + 1.5 * taper_bonus
        + 300.0 * side_dark
        + 160.0 * body_dark
        - 3.0 * angle_gap
    )
    return {
        "score": score,
        "line": (tuple_int(bottom), tuple_int(top)),
        "top": top,
        "bottom": bottom,
        "top_width": top_width,
        "bottom_width": bottom_width,
        "side_lines": ((tuple_int(seg_a["p1"]), tuple_int(seg_a["p2"])), (tuple_int(seg_b["p1"]), tuple_int(seg_b["p2"]))),
        "angle": math.degrees(math.atan2(float(top[1] - bottom[1]), float(top[0] - bottom[0]))) % 180.0,
        "length": center_length,
        "side_dark_support": side_dark,
        "body_dark_support": body_dark,
    }


def averaged_direction(seg_a, seg_b):
    direction = segment_unit(seg_a)
    if direction is None:
        return None

    other_direction = segment_unit(seg_b, direction)
    direction = direction + other_direction
    norm = np.linalg.norm(direction)
    return direction / norm if norm > 1e-6 else None


def paired_edge_endpoints(seg_a, seg_b, direction, min_overlap):
    a_proj = sorted([float(np.dot(seg_a["p1"], direction)), float(np.dot(seg_a["p2"], direction))])
    b_proj = sorted([float(np.dot(seg_b["p1"], direction)), float(np.dot(seg_b["p2"], direction))])
    overlap_start = max(a_proj[0], b_proj[0])
    overlap_end = min(a_proj[1], b_proj[1])
    if overlap_end - overlap_start < min_overlap:
        return None

    points = (
        point_at_projection(seg_a, direction, overlap_start),
        point_at_projection(seg_b, direction, overlap_start),
        point_at_projection(seg_a, direction, overlap_end),
        point_at_projection(seg_b, direction, overlap_end),
    )
    return None if any(point is None for point in points) else points


def fallback_pole_from_box(pole_box):
    if pole_box is None:
        return None

    x1, y1, x2, y2 = pole_box
    top = np.array([(x1 + x2) / 2.0, y1], dtype=np.float32)
    bottom = np.array([(x1 + x2) / 2.0, y2], dtype=np.float32)
    width = float(x2 - x1)
    return {
        "score": 0.0,
        "line": (tuple_int(bottom), tuple_int(top)),
        "top": top,
        "bottom": bottom,
        "top_width": width,
        "bottom_width": width,
        "side_lines": (((x1, y1), (x1, y2)), ((x2, y1), (x2, y2))),
        "angle": math.degrees(math.atan2(float(top[1] - bottom[1]), float(top[0] - bottom[0]))) % 180.0,
        "length": float(np.linalg.norm(top - bottom)),
        "side_dark_support": 0.0,
        "body_dark_support": 0.0,
        "source": "yolo_box",
    }


def detect_rod_centerline(segments, pole_result, frame_shape, dark_mask, config, rod_roi=None, rod_box=None, pole_roi=None):
    if pole_result is None:
        return fallback_rod_from_box(rod_box)

    diagonal = math.hypot(frame_shape[1], frame_shape[0])
    search_radius = config.rod_search_radius_ratio * diagonal
    pole_top = pole_result["top"]
    pole_angle = pole_result["angle"]

    best = None
    for segment in filter_segments_by_box(segments, rod_roi):
        if segment["length"] < config.rod_min_length:
            continue

        angle_gap = angle_diff_deg(segment["angle"], pole_angle)
        if angle_gap < config.rod_min_pole_angle:
            continue

        endpoint_distance = min(
            float(np.linalg.norm(segment["p1"] - pole_top)),
            float(np.linalg.norm(segment["p2"] - pole_top)),
            float(np.linalg.norm(segment["center"] - pole_top)),
        )
        if rod_roi is None and endpoint_distance > search_radius:
            continue

        dark_support = line_mask_ratio(dark_mask, segment["p1"], segment["p2"], thickness=7)
        if dark_support < config.rod_min_dark_support:
            continue

        score = rod_score(segment, pole_result, pole_roi, angle_gap, endpoint_distance, dark_support)
        if best is None or score > best["score"]:
            best = {
                "score": score,
                "line": (tuple_int(segment["p1"]), tuple_int(segment["p2"])),
                "length": segment["length"],
                "angle": segment["angle"],
                "distance_to_pole_top": endpoint_distance,
                "dark_support": dark_support,
                "source": "opencv_in_yolo_roi" if rod_roi is not None else "opencv_full_frame",
            }
    return best or fallback_rod_from_box(rod_box)


def rod_score(segment, pole_result, pole_roi, angle_gap, endpoint_distance, dark_support):
    length_penalty = max(0.0, segment["length"] - pole_result["length"] * 0.75)
    pole_overlap_penalty = 60.0 if pole_roi is not None and point_in_box(segment["center"], pole_roi) else 0.0
    return (
        segment["length"]
        + 1.5 * angle_gap
        + 220.0 * dark_support
        - 1.2 * endpoint_distance
        - length_penalty
        - pole_overlap_penalty
    )


def fallback_rod_from_box(rod_box):
    if rod_box is None:
        return None

    x1, y1, x2, y2 = rod_box
    width = x2 - x1
    height = y2 - y1
    if width >= height:
        p1 = (x1, int(round((y1 + y2) / 2.0)))
        p2 = (x2, int(round((y1 + y2) / 2.0)))
    else:
        p1 = (int(round((x1 + x2) / 2.0)), y1)
        p2 = (int(round((x1 + x2) / 2.0)), y2)

    angle = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0])) % 180.0
    return {
        "score": 0.0,
        "line": (p1, p2),
        "length": math.hypot(p2[0] - p1[0], p2[1] - p1[1]),
        "angle": angle,
        "distance_to_pole_top": 0.0,
        "dark_support": 0.0,
        "source": "yolo_box",
    }


def draw_detection(frame, pole_result, rod_result, edges, dark_mask, detections):
    annotated = frame.copy()
    draw_yolo_detections(annotated, detections)
    draw_pole_result(annotated, pole_result)
    draw_rod_result(annotated, rod_result)
    cv2.putText(
        annotated,
        f"edge px: {cv2.countNonZero(edges)}  dark px: {cv2.countNonZero(dark_mask)}",
        (10, annotated.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    return annotated


def draw_yolo_detections(annotated, detections):
    colors = {"pole": (0, 220, 255), "rod": (255, 100, 0), "bell": (0, 255, 0)}
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        color = colors.get(det["name"], (180, 180, 180))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)
        cv2.putText(
            annotated,
            f"{det['name']} {det['conf']:.2f}",
            (x1, max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )


def draw_pole_result(annotated, pole_result):
    if pole_result is None:
        cv2.putText(annotated, "pole not found", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        return

    for side_a, side_b in pole_result["side_lines"]:
        cv2.line(annotated, side_a, side_b, (0, 140, 255), 2, cv2.LINE_AA)

    bottom, top = pole_result["line"]
    cv2.line(annotated, bottom, top, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(annotated, bottom, 5, (0, 255, 255), -1)
    cv2.circle(annotated, top, 5, (0, 255, 255), -1)
    cv2.putText(
        annotated,
        f"pole widths {pole_result['bottom_width']:.0f}->{pole_result['top_width']:.0f}px  dark {pole_result['side_dark_support']:.2f}  {pole_result['source']}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_rod_result(annotated, rod_result):
    if rod_result is None:
        cv2.putText(annotated, "rod not found", (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        return

    p1, p2 = rod_result["line"]
    cv2.line(annotated, p1, p2, (255, 80, 0), 1, cv2.LINE_AA)
    cv2.circle(annotated, p1, 1, (255, 80, 0), -1)
    cv2.circle(annotated, p2, 1, (255, 80, 0), -1)
    cv2.putText(
        annotated,
        f"rod angle {rod_result['angle']:.0f} deg  dark {rod_result['dark_support']:.2f}  {rod_result['source']}",
        (10, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 80, 0),
        2,
        cv2.LINE_AA,
    )


def clamp_box(box, frame_shape):
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, int(round(x1))))
    y1 = max(0, min(height - 1, int(round(y1))))
    x2 = max(x1 + 1, min(width, int(round(x2))))
    y2 = max(y1 + 1, min(height, int(round(y2))))
    return (x1, y1, x2, y2)


def expand_box(box, pad_fraction, frame_shape):
    x1, y1, x2, y2 = box
    pad_x = (x2 - x1) * pad_fraction
    pad_y = (y2 - y1) * pad_fraction
    return clamp_box((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), frame_shape)


def point_in_box(point, box):
    x, y = float(point[0]), float(point[1])
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def segment_in_box(segment, box):
    return point_in_box(segment["center"], box) or point_in_box(segment["p1"], box) or point_in_box(segment["p2"], box)


def filter_segments_by_box(segments, box):
    if box is None:
        return segments
    return [segment for segment in segments if segment_in_box(segment, box)]


def segment_unit(segment, reference=None):
    unit = segment["p2"] - segment["p1"]
    norm = np.linalg.norm(unit)
    if norm <= 1e-6:
        return None
    unit = unit / norm
    if reference is not None and float(np.dot(unit, reference)) < 0:
        unit = -unit
    return unit


def point_at_projection(segment, direction, t):
    unit = segment_unit(segment, direction)
    if unit is None:
        return None

    denom = float(np.dot(unit, direction))
    if abs(denom) < 1e-4:
        return None

    step = (t - float(np.dot(segment["p1"], direction))) / denom
    return segment["p1"] + unit * step


def line_mask_ratio(mask, p1, p2, thickness):
    support = np.zeros(mask.shape[:2], dtype=np.uint8)
    cv2.line(support, tuple_int(p1), tuple_int(p2), 255, max(1, int(round(thickness))), cv2.LINE_AA)
    support_pixels = cv2.countNonZero(support)
    if support_pixels == 0:
        return 0.0
    return cv2.countNonZero(cv2.bitwise_and(mask, support)) / support_pixels


def polygon_mask_ratio(mask, points):
    support = np.zeros(mask.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(support, np.array([tuple_int(point) for point in points], dtype=np.int32), 255)
    support_pixels = cv2.countNonZero(support)
    if support_pixels == 0:
        return 0.0
    return cv2.countNonZero(cv2.bitwise_and(mask, support)) / support_pixels


def angle_diff_deg(a, b):
    return abs((a - b + 90.0) % 180.0 - 90.0)


def tuple_int(point):
    return tuple(int(round(float(v))) for v in point)
