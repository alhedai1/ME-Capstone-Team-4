#!/usr/bin/env python3
import contextlib
import io
from pathlib import Path

import cv2
import numpy as np

from capstone_robot.utils import find_repo_root
from capstone_robot.vision.bell_circle_climb import BellCircle
from capstone_robot.vision.pole_bell2 import PoleBellTracker, line_x_at_y


REPO_ROOT = find_repo_root(__file__)
PACKAGE_ROOT = REPO_ROOT / "src" / "capstone_robot"
OUT_DIR = PACKAGE_ROOT / "data" / "presentation_visuals"

POLE_MODEL = PACKAGE_ROOT / "train" / "runs" / "detect" / "runs" / "pole_final" / "yolo11n_pole_final_640" / "weights" / "best.pt"
POLE_VIDEO = PACKAGE_ROOT / "data" / "videos" / "may28" / "train_ai.mp4"
ALIGN_VIDEO = PACKAGE_ROOT / "data" / "videos" / "may25" / "may25_align_trim.mp4"
CLIMB_VIDEO = PACKAGE_ROOT / "data" / "videos" / "may27" / "ai_bell_upwards3.mp4"


GREEN = (70, 230, 80)
RED = (60, 80, 255)
BLUE = (255, 120, 40)
YELLOW = (40, 220, 255)
WHITE = (245, 245, 245)
BLACK = (20, 20, 20)
GRAY = (120, 120, 120)


def put_label(frame, text, org, scale=0.7, color=WHITE, bg=BLACK, thickness=2):
    x, y = org
    (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.rectangle(frame, (x - 6, y - h - 8), (x + w + 6, y + baseline + 6), bg, -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def title_bar(frame, title):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 44), BLACK, -1)
    cv2.putText(frame, title, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 2, cv2.LINE_AA)


def resize_height(frame, height=480):
    scale = height / frame.shape[0]
    width = int(round(frame.shape[1] * scale))
    return cv2.resize(frame, (width, height))


def fit_to_box(frame, width, height):
    scale = min(width / frame.shape[1], height / frame.shape[0])
    resized = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def compose_slide(title, panels, subtitles, output_name):
    slide_w, slide_h = 1920, 1080
    margin = 44
    header_h = 96
    gap = 24
    footer_h = 44
    panel_w = (slide_w - margin * 2 - gap * (len(panels) - 1)) // len(panels)
    panel_h = slide_h - header_h - footer_h - margin * 2

    slide = np.full((slide_h, slide_w, 3), 18, dtype=np.uint8)
    cv2.putText(slide, title, (margin, 66), cv2.FONT_HERSHEY_SIMPLEX, 1.45, WHITE, 3, cv2.LINE_AA)

    for i, (panel, subtitle) in enumerate(zip(panels, subtitles)):
        x = margin + i * (panel_w + gap)
        y = header_h
        fitted = fit_to_box(panel, panel_w, panel_h)
        slide[y : y + panel_h, x : x + panel_w] = fitted
        cv2.rectangle(slide, (x, y), (x + panel_w, y + panel_h), (70, 70, 70), 2)
        put_label(slide, subtitle, (x + 18, y + 42), 0.9, WHITE)

    cv2.imwrite(str(OUT_DIR / output_name), slide)


def make_contact_sheets():
    videos = [
        POLE_VIDEO,
        PACKAGE_ROOT / "data" / "videos" / "may28" / "train_ai.mp4",
        PACKAGE_ROOT / "data" / "videos" / "may24" / "pole_training" / "pole_training.mp4",
        ALIGN_VIDEO,
        PACKAGE_ROOT / "data" / "videos" / "may27" / "bell_align_controls.mp4",
        CLIMB_VIDEO,
        PACKAGE_ROOT / "data" / "videos" / "may28" / "train_ai_bell.mp4",
        PACKAGE_ROOT / "data" / "videos" / "may28" / "train_ai_bell2.mp4",
    ]
    for video in videos:
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            continue
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_ids = np.linspace(0, max(0, count - 1), 12, dtype=int) if count else np.arange(12) * 30
        thumbs = []
        for frame_id in frame_ids:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.resize(frame, (320, 240))
            title_bar(frame, f"{video.name}  f={frame_id}  t={frame_id / fps:.1f}s")
            thumbs.append(frame)
        cap.release()
        if not thumbs:
            continue
        while len(thumbs) < 12:
            thumbs.append(np.zeros_like(thumbs[0]))
        sheet = np.vstack([np.hstack(thumbs[i : i + 4]) for i in range(0, 12, 4)])
        cv2.imwrite(str(OUT_DIR / f"contact_{video.stem}.jpg"), sheet)


def best_pole_detection(result):
    if result.boxes is None or len(result.boxes) == 0:
        return None
    boxes = result.boxes
    confs = boxes.conf.cpu().numpy()
    idx = int(np.argmax(confs))
    xyxy = boxes.xyxy[idx].cpu().numpy()
    cls_id = int(boxes.cls[idx].item()) if boxes.cls is not None else 0
    name = result.names.get(cls_id, "pole") if hasattr(result, "names") else "pole"
    return xyxy, float(confs[idx]), str(name)


def draw_pole_control(frame, box, conf, label):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    h, w = frame.shape[:2]
    center_x = int(round((x1 + x2) / 2.0))
    image_center = w // 2
    error = center_x - image_center
    width_fraction = (x2 - x1) / w
    deadband = 20
    stop_threshold = 0.20

    if width_fraction >= stop_threshold:
        command = "STOP: pole reached"
        command_color = RED
    elif error < -deadband:
        command = "TURN LEFT"
        command_color = YELLOW
    elif error > deadband:
        command = "TURN RIGHT"
        command_color = YELLOW
    else:
        command = "DRIVE FORWARD"
        command_color = GREEN

    cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 3)
    cv2.line(frame, (image_center, 45), (image_center, h), GRAY, 2)
    cv2.line(frame, (center_x, y1), (center_x, y2), BLUE, 2)
    cv2.circle(frame, (center_x, int((y1 + y2) / 2)), 7, BLUE, -1)
    cv2.arrowedLine(frame, (image_center, h - 70), (center_x, h - 70), YELLOW, 3, tipLength=0.08)

    put_label(frame, f"{label} {conf:.2f}", (x1, max(64, y1 - 10)), 0.7, GREEN)
    put_label(frame, f"error_x = {error:+.0f} px", (24, h - 92), 0.7, WHITE)
    put_label(frame, f"width/frame = {width_fraction:.2f}", (24, h - 54), 0.7, WHITE)
    put_label(frame, command, (w - 300, h - 54), 0.75, command_color)
    title_bar(frame, "AI Camera YOLO Pole Detection and Approach Control")
    return frame, abs(error), width_fraction


def make_pole_visual():
    from ultralytics import YOLO

    model = YOLO(str(POLE_MODEL))
    cap = cv2.VideoCapture(str(POLE_VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {POLE_VIDEO}")

    candidates = []
    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_id % 8 != 0:
            frame_id += 1
            continue
        result = model.predict(frame, imgsz=640, conf=0.25, verbose=False)[0]
        detection = best_pole_detection(result)
        if detection is not None:
            box, conf, label = detection
            x1, y1, x2, y2 = box
            width_fraction = (x2 - x1) / frame.shape[1]
            error = ((x1 + x2) / 2.0) - (frame.shape[1] / 2.0)
            candidates.append((frame_id, frame.copy(), box, conf, label, error, width_fraction))
        frame_id += 1
    cap.release()

    if not candidates:
        raise RuntimeError("No pole detections found")

    clean = [
        item
        for item in candidates
        if item[2][0] > 20 and item[2][2] < item[1].shape[1] - 20 and item[6] > 0.04
    ] or candidates
    far = min(clean, key=lambda item: abs(item[6] - 0.08) + abs(item[5]) / 500.0)
    steer_pool = [c for c in clean if c[6] < 0.18 and abs(c[5]) > 50] or clean
    steer = max(steer_pool, key=lambda item: abs(item[5]))
    close = max(candidates, key=lambda item: item[6])
    selected = [("Steer", steer), ("Centered/Far", far), ("Distance Estimate", close)]

    panels = []
    subtitles = []
    for subtitle, item in selected:
        _, frame, box, conf, label, _, _ = item
        frame = draw_pole_control(frame, box, conf, label)[0]
        put_label(frame, subtitle, (24, 78), 0.8, WHITE)
        panels.append(frame)
        subtitles.append(subtitle)

    compose_slide(
        "AI Camera YOLO Pole Detection and Approach Control",
        panels,
        subtitles,
        "01_pole_detection_approach.png",
    )
    cv2.imwrite(str(OUT_DIR / "01_pole_detection_single.png"), fit_to_box(panels[0], 1280, 720))


def draw_alignment(frame, alignment):
    h, w = frame.shape[:2]
    title_bar(frame, "OpenCV Pole Centerline and Bell Alignment")
    if alignment is None:
        put_label(frame, "No alignment detected", (24, 82), 0.75, RED)
        return frame

    line = alignment.pole_line
    x_top = line_x_at_y(line, 0)
    x_bottom = line_x_at_y(line, h - 1)
    if x_top is not None and x_bottom is not None:
        cv2.line(frame, (int(x_top), 0), (int(x_bottom), h - 1), GREEN, 3)

    bx, by, br = alignment.bell
    cv2.circle(frame, (int(bx), int(by)), int(br), BLUE, 3)
    cv2.circle(frame, (int(bx), int(by)), 6, BLUE, -1)
    line_x = line_x_at_y(line, by)
    if line_x is not None:
        cv2.line(frame, (int(line_x), int(by)), (int(bx), int(by)), YELLOW, 3)
        cv2.circle(frame, (int(line_x), int(by)), 6, GREEN, -1)
        for offset in (-30, 30):
            cv2.line(frame, (int(line_x + offset), 45), (int(line_x + offset), h), (80, 80, 80), 1)

    command = "ORBIT RIGHT" if alignment.side == "right" else "ORBIT LEFT"
    if abs(alignment.error_px) <= 30:
        command = "ALIGNED"
    put_label(frame, f"bell error = {alignment.error_px:+.1f} px", (24, h - 92), 0.7, WHITE)
    put_label(frame, f"threshold = +/-30 px", (24, h - 54), 0.7, WHITE)
    put_label(frame, command, (w - 220, h - 54), 0.75, GREEN if command == "ALIGNED" else YELLOW)
    return frame


def make_alignment_visual():
    tracker = PoleBellTracker(color_format="bgr")
    cap = cv2.VideoCapture(str(ALIGN_VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {ALIGN_VIDEO}")
    candidates = []
    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_id % 5 == 0:
            with contextlib.redirect_stdout(io.StringIO()):
                alignment = tracker.detect(frame)
            if alignment is not None:
                candidates.append((frame_id, frame.copy(), alignment))
        frame_id += 1
    cap.release()
    if not candidates:
        raise RuntimeError("No pole/bell alignment detections found")

    left = min(candidates, key=lambda item: item[2].error_px)
    center = min(candidates, key=lambda item: abs(item[2].error_px))
    right = max(candidates, key=lambda item: item[2].error_px)
    selected = [("Bell Left", left), ("Aligned", center), ("Bell Right", right)]

    panels = []
    subtitles = []
    for subtitle, item in selected:
        _, frame, alignment = item
        frame = draw_alignment(frame, alignment)
        put_label(frame, subtitle, (24, 78), 0.8, WHITE)
        panels.append(frame)
        subtitles.append(subtitle)
    compose_slide(
        "OpenCV Pole Centerline and Bell Alignment",
        panels,
        subtitles,
        "02_pole_bell_alignment.png",
    )
    cv2.imwrite(str(OUT_DIR / "02_pole_bell_alignment_single.png"), fit_to_box(panels[1], 1280, 720))


def draw_climb(frame, detection, missed_frames, stable_frames, status):
    h, w = frame.shape[:2]
    title_bar(frame, "Climbing Feedback: Bell Circle Detection")
    cv2.line(frame, (w // 2, 45), (w // 2, h), GRAY, 2)
    if detection is not None:
        x, y, radius = detection.circle
        cv2.circle(frame, (x, y), radius, BLUE, 3)
        cv2.circle(frame, (x, y), 6, GREEN, -1)
        put_label(frame, f"x={x} y={y} r={radius}", (24, h - 92), 0.7, WHITE)
        put_label(frame, f"stable={stable_frames} missed={missed_frames}", (24, h - 54), 0.7, WHITE)
    else:
        put_label(frame, "circle not visible", (24, h - 92), 0.7, RED)
        put_label(frame, f"missed={missed_frames}", (24, h - 54), 0.7, WHITE)
    command = "DESCEND" if "DESCEND" in status else "CLIMB"
    put_label(frame, command, (w - 180, h - 54), 0.75, GREEN if command == "CLIMB" else YELLOW)
    return frame


def make_climb_visual():
    detector = BellCircle(
        color_format="bgr",
        dp=1.5,
        min_dist=5,
        param1=50,
        param2=50,
        min_radius=10,
        max_radius=50,
        startup_max_radius=50,
        tracking_max_radius=130,
        lost_after_frames=8,
        startup_confirm_threshold=2,
    )
    cap = cv2.VideoCapture(str(CLIMB_VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {CLIMB_VIDEO}")
    detections = []
    misses = []
    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        with contextlib.redirect_stdout(io.StringIO()):
            detection = detector.detect(frame)
        if frame_id % 4 == 0:
            if detection is None:
                misses.append((frame_id, frame.copy(), None, detector.missed_frames, detector.stable_frames))
            else:
                detections.append((frame_id, frame.copy(), detection, detector.missed_frames, detector.stable_frames))
        frame_id += 1
    cap.release()
    if not detections:
        raise RuntimeError("No climb bell circle detections found")

    small = min(detections, key=lambda item: item[2].radius)
    large = max(detections, key=lambda item: item[2].radius)
    middle = detections[len(detections) // 2]
    selected = [
        ("CLIMB: bell visible", small),
        ("CLIMB: tracking circle", middle),
        ("DESCEND: bell passed", large),
    ]

    panels = []
    subtitles = []
    for status, item in selected:
        _, frame, detection, missed_frames, stable_frames = item
        frame = draw_climb(frame, detection, missed_frames, stable_frames, status)
        panels.append(frame)
        subtitles.append(status)
    compose_slide(
        "Climbing Feedback: Bell Circle Detection",
        panels,
        subtitles,
        "03_climb_bell_circle.png",
    )
    cv2.imwrite(str(OUT_DIR / "03_climb_bell_circle_single.png"), fit_to_box(panels[1], 1280, 720))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    make_contact_sheets()
    make_pole_visual()
    make_alignment_visual()
    make_climb_visual()
    print(f"Saved presentation visuals to: {OUT_DIR}")


if __name__ == "__main__":
    main()
