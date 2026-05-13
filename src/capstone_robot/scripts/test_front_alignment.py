#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_source(value):
    return int(value) if value.isdigit() else value


def parse_args():
    parser = argparse.ArgumentParser(description="Test front-camera pole/target alignment with YOLO")
    parser.add_argument("--source", default="0", help="camera index or video path")
    parser.add_argument("--model", required=True, help="path to YOLO .pt model")
    parser.add_argument("--conf", type=float, default=0.4, help="confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO image size")
    parser.add_argument("--save-output", type=Path, default=None, help="optional annotated output video path")
    parser.add_argument("--show", action="store_true", help="show annotated preview")
    parser.add_argument("--rotate", choices=["none", "cw", "ccw", "180"], default="none")
    parser.add_argument("--aligned-threshold-px", type=float, default=40)
    return parser.parse_args()


def rotate_frame(frame, rotation):
    if rotation == "cw":
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation == "ccw":
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotation == "180":
        return cv2.rotate(frame, cv2.ROTATE_180)
    return frame


def center_x(box):
    x1, _, x2, _ = box
    return (x1 + x2) / 2.0


def pick_best_detection(result, class_name):
    names = result.names
    wanted = class_name.lower()
    best = None

    for box in result.boxes:
        cls_id = int(box.cls[0])
        detected_name = names.get(cls_id, str(cls_id)).lower()
        if detected_name != wanted:
            continue

        conf = float(box.conf[0])
        xyxy = [float(v) for v in box.xyxy[0].tolist()]
        detection = {
            "class_name": detected_name,
            "conf": conf,
            "box": xyxy,
            "center_x": center_x(xyxy),
        }

        if best is None or conf > best["conf"]:
            best = detection

    return best


def compute_alignment(pole_detection, target_detection, aligned_threshold_px):
    if pole_detection is None:
        return {
            "status": "MISSING_POLE",
            "pole_x": None,
            "target_x": None,
            "x_error": None,
        }

    pole_x = pole_detection["center_x"]
    if target_detection is None:
        return {
            "status": "MISSING_TARGET",
            "pole_x": pole_x,
            "target_x": None,
            "x_error": None,
        }

    target_x = target_detection["center_x"]
    x_error = target_x - pole_x

    if abs(x_error) <= aligned_threshold_px:
        status = "ALIGNED"
    elif x_error > 0:
        status = "TARGET_RIGHT_OF_POLE"
    else:
        status = "TARGET_LEFT_OF_POLE"

    return {
        "status": status,
        "pole_x": pole_x,
        "target_x": target_x,
        "x_error": x_error,
    }


def draw_box(frame, detection, color):
    if detection is None:
        return

    x1, y1, x2, y2 = [int(v) for v in detection["box"]]
    label = f"{detection['class_name']} {detection['conf']:.2f}"
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        label,
        (x1, max(y1 - 8, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_vertical_line(frame, x_value, color):
    if x_value is None:
        return

    x = int(round(x_value))
    cv2.line(frame, (x, 0), (x, frame.shape[0]), color, 2)


def draw_overlay(frame, pole_detection, target_detection, alignment):
    annotated = frame.copy()

    draw_box(annotated, pole_detection, (0, 255, 0))
    draw_box(annotated, target_detection, (0, 128, 255))

    draw_vertical_line(annotated, alignment["pole_x"], (0, 255, 0))
    draw_vertical_line(annotated, alignment["target_x"], (0, 128, 255))

    status = alignment["status"]
    x_error = alignment["x_error"]
    x_error_text = "x_error=N/A" if x_error is None else f"x_error={x_error:.1f}px"

    cv2.putText(
        annotated,
        status,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        x_error_text,
        (10, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return annotated


def format_value(value):
    return "None" if value is None else f"{value:.1f}"


def open_writer(output_path, fps, frame_shape):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame_shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video for writing: {output_path}")
    return writer


def main():
    args = parse_args()
    source = parse_source(str(args.source))
    model = YOLO(args.model)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    writer = None
    frame_index = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            frame = rotate_frame(frame, args.rotate)
            result = model(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]

            pole = pick_best_detection(result, "pole")
            bell = pick_best_detection(result, "bell")
            rod = pick_best_detection(result, "rod")
            target = bell if bell is not None else rod

            alignment = compute_alignment(pole, target, args.aligned_threshold_px)
            annotated = draw_overlay(frame, pole, target, alignment)

            print(
                f"frame={frame_index} "
                f"status={alignment['status']} "
                f"pole_x={format_value(alignment['pole_x'])} "
                f"target_x={format_value(alignment['target_x'])} "
                f"x_error={format_value(alignment['x_error'])}"
            )

            if args.save_output is not None:
                if writer is None:
                    writer = open_writer(args.save_output, fps, annotated.shape)
                writer.write(annotated)

            if args.show:
                cv2.imshow("front_alignment", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_index += 1

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"Saved annotated video to: {args.save_output}")
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
