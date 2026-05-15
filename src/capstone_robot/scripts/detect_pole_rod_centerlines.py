#!/usr/bin/env python3
import argparse
import math
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from capstone_robot.utils import find_repo_root

REPO_ROOT = find_repo_root(__file__)

DEFAULT_VIDEO_PATH = REPO_ROOT / "src/capstone_robot/data/videos/may14/raw/bellcenter.mp4"
DEFAULT_IMAGE_PATH = REPO_ROOT / "src/capstone_robot/data/extracted_frames/bell-bottom/rotated_90_ccw/frame_000170.jpg"
IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
DEFAULT_MODEL = REPO_ROOT / "src/capstone_robot/train/runs/detect/runs/upward_2/yolo11n_upward_2_640/weights/best.pt"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect pole and rod centerlines from edge geometry in an image or video."
    )
    parser.add_argument("--path", type=Path, default=DEFAULT_VIDEO_PATH, help="input image or video path")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="trained YOLO pole/rod model path")
    parser.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--no-yolo", action="store_true", help="use full-frame OpenCV only")
    parser.add_argument("--full-frame-fallback", action="store_true", help="run full-frame OpenCV if YOLO misses a class")
    parser.add_argument("--roi-pad", type=float, default=0.25, help="fractional padding around YOLO boxes for OpenCV refinement")
    parser.add_argument("--output", type=Path, default=None, help="optional annotated output image/video path")
    parser.add_argument("--rotate", choices=["none", "cw", "ccw", "180"], default="none")
    parser.add_argument("--resize-width", type=int, default=0, help="resize frames to this width before processing")
    parser.add_argument("--blur", type=int, default=5, help="odd Gaussian blur kernel size; use 0 to disable")
    parser.add_argument("--dark-threshold", type=int, default=95, help="gray/value threshold for black pole and rod pixels")
    parser.add_argument("--dark-max-saturation", type=int, default=120, help="max HSV saturation for black-material support")
    parser.add_argument("--very-dark-threshold", type=int, default=55, help="value threshold accepted even if saturation is high")
    parser.add_argument("--pole-min-dark-support", type=float, default=0.10, help="minimum dark-pixel ratio along paired pole edges")
    parser.add_argument("--rod-min-dark-support", type=float, default=0.08, help="minimum dark-pixel ratio along rod candidate")
    parser.add_argument("--canny-low", type=int, default=45, help="Canny low threshold")
    parser.add_argument("--canny-high", type=int, default=130, help="Canny high threshold")
    parser.add_argument("--hough-threshold", type=int, default=20, help="Hough line vote threshold")
    parser.add_argument("--min-line-length", type=int, default=25, help="minimum Hough segment length in pixels")
    parser.add_argument("--max-line-gap", type=int, default=20, help="maximum Hough segment gap in pixels")
    parser.add_argument("--pole-min-length-ratio", type=float, default=0.12, help="pole side must be this fraction of frame diagonal")
    parser.add_argument("--pole-angle-tolerance", type=float, default=35.0, help="max angle between paired pole sides")
    parser.add_argument("--pole-min-width", type=float, default=10.0, help="minimum distance between pole side lines")
    parser.add_argument("--pole-max-width-ratio", type=float, default=0.28, help="maximum pole width as fraction of frame diagonal")
    parser.add_argument("--rod-min-length", type=int, default=20, help="minimum rod segment length in pixels")
    parser.add_argument("--rod-search-radius-ratio", type=float, default=0.28, help="rod search radius around pole top")
    parser.add_argument("--rod-min-pole-angle", type=float, default=25.0, help="minimum angle between rod and pole centerline")
    parser.add_argument("--show-edges", action="store_true", help="show the edge image used for line detection")
    parser.add_argument("--show-dark-mask", action="store_true", help="show the black-material support mask")
    parser.add_argument("--no-display", action="store_true", help="process without opening preview windows")
    parser.add_argument("--window-name", default="pole_rod_centerlines")
    return parser.parse_args()


def rotate_frame(frame, rotation):
    if rotation == "cw":
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation == "ccw":
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotation == "180":
        return cv2.rotate(frame, cv2.ROTATE_180)
    return frame


def resize_frame(frame, target_width):
    if target_width <= 0 or frame.shape[1] == target_width:
        return frame
    scale = target_width / frame.shape[1]
    target_height = max(1, int(round(frame.shape[0] * scale)))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def open_writer(output_path, fps, frame_shape):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame_shape[:2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video for writing: {output_path}")
    return writer


def resolve_model_path(model_path):
    if model_path.exists():
        return model_path

    package_root = Path(__file__).resolve().parents[1]
    candidates = sorted(
        package_root.rglob("weights/best.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates and model_path == DEFAULT_MODEL:
        return candidates[0]

    raise SystemExit(f"Model does not exist: {model_path}")


def build_edges(frame, blur_kernel, canny_low, canny_high):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    if blur_kernel > 0:
        blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

    edges = cv2.Canny(gray, canny_low, canny_high)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kernel, iterations=1)


def build_dark_mask(frame, blur_kernel, dark_threshold, dark_max_saturation, very_dark_threshold):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    if blur_kernel > 0:
        blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        saturation = cv2.GaussianBlur(saturation, (blur_kernel, blur_kernel), 0)
        value = cv2.GaussianBlur(value, (blur_kernel, blur_kernel), 0)

    low_value = cv2.inRange(value, 0, dark_threshold)
    low_saturation = cv2.inRange(saturation, 0, dark_max_saturation)
    very_dark = cv2.inRange(value, 0, very_dark_threshold)
    dark_mask = cv2.bitwise_or(cv2.bitwise_and(low_value, low_saturation), very_dark)

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    return cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)


def angle_diff_deg(a, b):
    diff = abs((a - b + 90.0) % 180.0 - 90.0)
    return diff


def segment_from_hough(raw):
    x1, y1, x2, y2 = [float(v) for v in raw]
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length <= 1e-6:
        return None

    angle = math.degrees(math.atan2(dy, dx)) % 180.0
    center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)
    return {
        "p1": np.array([x1, y1], dtype=np.float32),
        "p2": np.array([x2, y2], dtype=np.float32),
        "center": center,
        "length": length,
        "angle": angle,
    }


def hough_segments(edges, threshold, min_line_length, max_line_gap):
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    if lines is None:
        return []

    segments = []
    for raw in lines[:, 0, :]:
        segment = segment_from_hough(raw)
        if segment is not None:
            segments.append(segment)
    return segments


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
    width = x2 - x1
    height = y2 - y1
    pad_x = width * pad_fraction
    pad_y = height * pad_fraction
    return clamp_box((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), frame_shape)


def point_in_box(point, box):
    x, y = float(point[0]), float(point[1])
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def segment_in_box(segment, box):
    return point_in_box(segment["center"], box) or point_in_box(segment["p1"], box) or point_in_box(segment["p2"], box)


def run_yolo(model, frame, args):
    if model is None:
        return []

    result = model(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
    detections = []
    if result.boxes is None:
        return detections

    names = result.names
    for box in result.boxes:
        cls_id = int(box.cls.item())
        name = str(names.get(cls_id, cls_id)).lower()
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
    if not matches:
        return None
    return max(matches, key=lambda det: det["conf"])


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
    supported = cv2.bitwise_and(mask, support)
    return cv2.countNonZero(supported) / support_pixels


def polygon_mask_ratio(mask, points):
    support = np.zeros(mask.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(support, np.array([tuple_int(point) for point in points], dtype=np.int32), 255)
    support_pixels = cv2.countNonZero(support)
    if support_pixels == 0:
        return 0.0
    supported = cv2.bitwise_and(mask, support)
    return cv2.countNonZero(supported) / support_pixels


def score_pole_pair(seg_a, seg_b, frame_shape, dark_mask, args):
    height, width = frame_shape[:2]
    diagonal = math.hypot(width, height)

    if min(seg_a["length"], seg_b["length"]) < args.pole_min_length_ratio * diagonal:
        return None

    angle_gap = angle_diff_deg(seg_a["angle"], seg_b["angle"])
    if angle_gap > args.pole_angle_tolerance:
        return None

    direction = segment_unit(seg_a)
    if direction is None:
        return None

    other_direction = segment_unit(seg_b, direction)
    direction = direction + other_direction
    direction_norm = np.linalg.norm(direction)
    if direction_norm <= 1e-6:
        return None
    direction = direction / direction_norm
    normal = np.array([-direction[1], direction[0]], dtype=np.float32)

    offset_distance = abs(float(np.dot(seg_a["center"] - seg_b["center"], normal)))
    max_width = args.pole_max_width_ratio * diagonal
    if offset_distance < args.pole_min_width or offset_distance > max_width:
        return None

    a_proj = sorted([float(np.dot(seg_a["p1"], direction)), float(np.dot(seg_a["p2"], direction))])
    b_proj = sorted([float(np.dot(seg_b["p1"], direction)), float(np.dot(seg_b["p2"], direction))])
    overlap_start = max(a_proj[0], b_proj[0])
    overlap_end = min(a_proj[1], b_proj[1])
    overlap = overlap_end - overlap_start
    if overlap < args.pole_min_length_ratio * diagonal * 0.45:
        return None

    start_a = point_at_projection(seg_a, direction, overlap_start)
    start_b = point_at_projection(seg_b, direction, overlap_start)
    end_a = point_at_projection(seg_a, direction, overlap_end)
    end_b = point_at_projection(seg_b, direction, overlap_end)
    if start_a is None or start_b is None or end_a is None or end_b is None:
        return None

    start_center = (start_a + start_b) / 2.0
    end_center = (end_a + end_b) / 2.0
    start_width = float(np.linalg.norm(start_a - start_b))
    end_width = float(np.linalg.norm(end_a - end_b))

    if start_width < args.pole_min_width or end_width < args.pole_min_width:
        return None

    side_thickness = max(5.0, min(start_width, end_width) * 0.16)
    side_a_dark = line_mask_ratio(dark_mask, start_a, end_a, side_thickness)
    side_b_dark = line_mask_ratio(dark_mask, start_b, end_b, side_thickness)
    side_dark_support = (side_a_dark + side_b_dark) / 2.0
    body_dark_support = polygon_mask_ratio(dark_mask, [start_a, start_b, end_b, end_a])
    if side_dark_support < args.pole_min_dark_support and body_dark_support < args.pole_min_dark_support * 0.55:
        return None

    if start_width <= end_width:
        top = start_center
        bottom = end_center
        top_width = start_width
        bottom_width = end_width
    else:
        top = end_center
        bottom = start_center
        top_width = end_width
        bottom_width = start_width

    center_length = float(np.linalg.norm(top - bottom))
    if center_length < args.pole_min_length_ratio * diagonal * 0.45:
        return None

    taper_bonus = max(0.0, bottom_width - top_width)
    dark_bonus = 300.0 * side_dark_support + 160.0 * body_dark_support
    score = overlap + 0.35 * (seg_a["length"] + seg_b["length"]) + 1.5 * taper_bonus + dark_bonus - 3.0 * angle_gap
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
        "side_dark_support": side_dark_support,
        "body_dark_support": body_dark_support,
    }


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


def detect_pole_centerline(segments, frame_shape, dark_mask, args, pole_roi=None, pole_box=None):
    search_segments = filter_segments_by_box(segments, pole_roi)
    best = None
    for i, seg_a in enumerate(search_segments):
        for seg_b in search_segments[i + 1 :]:
            candidate = score_pole_pair(seg_a, seg_b, frame_shape, dark_mask, args)
            if candidate is not None and (best is None or candidate["score"] > best["score"]):
                candidate["source"] = "opencv_in_yolo_roi" if pole_roi is not None else "opencv_full_frame"
                best = candidate
    return best or fallback_pole_from_box(pole_box)


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


def detect_rod_centerline(segments, pole_result, frame_shape, dark_mask, args, rod_roi=None, rod_box=None, pole_roi=None):
    if pole_result is None:
        return fallback_rod_from_box(rod_box)

    height, width = frame_shape[:2]
    diagonal = math.hypot(width, height)
    search_radius = args.rod_search_radius_ratio * diagonal
    pole_top = pole_result["top"]
    pole_angle = pole_result["angle"]
    search_segments = filter_segments_by_box(segments, rod_roi)

    best = None
    for segment in search_segments:
        if segment["length"] < args.rod_min_length:
            continue

        angle_gap = angle_diff_deg(segment["angle"], pole_angle)
        if angle_gap < args.rod_min_pole_angle:
            continue

        endpoint_distance = min(
            float(np.linalg.norm(segment["p1"] - pole_top)),
            float(np.linalg.norm(segment["p2"] - pole_top)),
            float(np.linalg.norm(segment["center"] - pole_top)),
        )
        if rod_roi is None and endpoint_distance > search_radius:
            continue

        dark_support = line_mask_ratio(dark_mask, segment["p1"], segment["p2"], thickness=7)
        if dark_support < args.rod_min_dark_support:
            continue

        length_penalty = max(0.0, segment["length"] - pole_result["length"] * 0.75)
        pole_overlap_penalty = 0.0
        if pole_roi is not None and point_in_box(segment["center"], pole_roi):
            pole_overlap_penalty = 60.0

        score = (
            segment["length"]
            + 1.5 * angle_gap
            + 220.0 * dark_support
            - 1.2 * endpoint_distance
            - length_penalty
            - pole_overlap_penalty
        )
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


def tuple_int(point):
    return tuple(int(round(float(v))) for v in point)


def draw_yolo_detections(annotated, detections):
    colors = {
        "pole": (0, 220, 255),
        "rod": (255, 100, 0),
        "bell": (0, 255, 0),
    }
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


def draw_detection(frame, pole_result, rod_result, edges, dark_mask, detections):
    annotated = frame.copy()
    draw_yolo_detections(annotated, detections)

    if pole_result is not None:
        for side_a, side_b in pole_result["side_lines"]:
            cv2.line(annotated, side_a, side_b, (0, 140, 255), 2, cv2.LINE_AA)

        bottom, top = pole_result["line"]
        cv2.line(annotated, bottom, top, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(annotated, bottom, 5, (0, 255, 255), -1)
        cv2.circle(annotated, top, 5, (0, 255, 255), -1)
        cv2.putText(
            annotated,
            f"pole centerline  widths {pole_result['bottom_width']:.0f}->{pole_result['top_width']:.0f}px  dark {pole_result['side_dark_support']:.2f}  {pole_result['source']}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(annotated, "pole not found", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    if rod_result is not None:
        p1, p2 = rod_result["line"]
        cv2.line(annotated, p1, p2, (255, 80, 0), 1, cv2.LINE_AA)
        cv2.circle(annotated, p1, 1, (255, 80, 0), -1)
        cv2.circle(annotated, p2, 1, (255, 80, 0), -1)
        cv2.putText(
            annotated,
            f"rod centerline  angle {rod_result['angle']:.0f} deg  dark {rod_result['dark_support']:.2f}  {rod_result['source']}",
            (10, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 80, 0),
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(annotated, "rod not found", (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

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


def process_frame(frame, args, model):
    frame = rotate_frame(frame, args.rotate)
    frame = resize_frame(frame, args.resize_width)
    detections = run_yolo(model, frame, args)
    pole_detection = best_detection(detections, "pole")
    rod_detection = best_detection(detections, "rod")

    pole_box = pole_detection["box"] if pole_detection is not None else None
    rod_box = rod_detection["box"] if rod_detection is not None else None
    pole_roi = expand_box(pole_box, args.roi_pad, frame.shape) if pole_box is not None else None
    rod_roi = expand_box(rod_box, args.roi_pad, frame.shape) if rod_box is not None else None
    yolo_enabled = model is not None

    edges = build_edges(frame, args.blur, args.canny_low, args.canny_high)
    dark_mask = build_dark_mask(
        frame,
        args.blur,
        args.dark_threshold,
        args.dark_max_saturation,
        args.very_dark_threshold,
    )
    segments = hough_segments(edges, args.hough_threshold, args.min_line_length, args.max_line_gap)
    pole_result = None
    if pole_roi is not None or not yolo_enabled or args.full_frame_fallback:
        pole_result = detect_pole_centerline(
            segments,
            frame.shape,
            dark_mask,
            args,
            pole_roi=pole_roi,
            pole_box=pole_box,
        )

    rod_result = None
    if rod_roi is not None or not yolo_enabled or args.full_frame_fallback:
        rod_result = detect_rod_centerline(
            segments,
            pole_result,
            frame.shape,
            dark_mask,
            args,
            rod_roi=rod_roi,
            rod_box=rod_box,
            pole_roi=pole_roi,
        )
    return draw_detection(frame, pole_result, rod_result, edges, dark_mask, detections), edges, dark_mask


def process_image(path, args, model):
    frame = cv2.imread(str(path))
    if frame is None:
        raise SystemExit(f"Could not read image: {path}")

    annotated, edges, dark_mask = process_frame(frame, args, model)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.output), annotated)
        print(f"Saved annotated image to: {args.output}")

    if not args.no_display:
        cv2.imshow(args.window_name, annotated)
        if args.show_edges:
            cv2.imshow(f"{args.window_name}_edges", edges)
        if args.show_dark_mask:
            cv2.imshow(f"{args.window_name}_dark_mask", dark_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_video(path, args, model):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    writer = None
    frame_count = 0
    start = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            annotated, edges, dark_mask = process_frame(frame, args, model)

            if args.output is not None:
                if writer is None:
                    writer = open_writer(args.output, fps, annotated.shape)
                writer.write(annotated)

            if not args.no_display:
                cv2.imshow(args.window_name, annotated)
                if args.show_edges:
                    cv2.imshow(f"{args.window_name}_edges", edges)
                if args.show_dark_mask:
                    cv2.imshow(f"{args.window_name}_dark_mask", dark_mask)
                key = cv2.waitKey(100) & 0xFF
                if key == ord("q") or key == 27:
                    break

            frame_count += 1

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"Saved annotated video to: {args.output}")
        if not args.no_display:
            cv2.destroyAllWindows()

    elapsed = time.time() - start
    average_fps = frame_count / elapsed if elapsed > 0 else 0.0
    print(f"Frames processed: {frame_count}")
    print(f"Average FPS: {average_fps:.2f}")


def main():
    args = parse_args()
    if not args.path.exists():
        raise SystemExit(f"Input does not exist: {args.path}")

    model = None
    if not args.no_yolo:
        model_path = resolve_model_path(args.model)
        print(f"Using YOLO model: {model_path}")
        model = YOLO(str(model_path))

    if args.path.suffix.lower() in IMAGE_SUFFIXES:
        process_image(args.path, args, model)
    else:
        process_video(args.path, args, model)


if __name__ == "__main__":
    main()
