#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

from capstone_robot.utils import find_repo_root, rotate_frame


REPO_ROOT = find_repo_root(__file__)
DEFAULT_MODEL = (
    REPO_ROOT
    / "src/capstone_robot/train/runs/detect/runs/pole_final/yolo11n_pole_final_640/weights/best.pt"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "src/capstone_robot/data/presentation_visuals"


GREEN = (70, 230, 80)
RED = (60, 80, 255)
BLUE = (255, 120, 40)
YELLOW = (40, 220, 255)
WHITE = (245, 245, 245)
BLACK = (20, 20, 20)
GRAY = (120, 120, 120)
DARK_BLUE = (130, 55, 0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run pole YOLO on one image and draw steering/approach annotations."
    )
    parser.add_argument("image", type=Path, help="input image path")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="YOLO .pt model path")
    parser.add_argument("--output", type=Path, default=None, help="annotated output image path")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--deadband", type=float, default=20.0, help="center error deadband in pixels")
    parser.add_argument(
        "--stop-width-fraction",
        type=float,
        default=0.15,
        help="report STOP/CLOSE when box width divided by image width is at least this value",
    )
    parser.add_argument("--rotate", choices=["none", "cw", "ccw", "180"], default="none")
    parser.add_argument("--no-show", action="store_true", help="save/print only; do not open OpenCV window")
    return parser.parse_args()


def put_label(frame, text, org, scale=1.0, color=WHITE, thickness=2):
    x, y = org
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_PLAIN, scale, color, thickness, cv2.LINE_AA)


def best_detection(result):
    if result.boxes is None or len(result.boxes) == 0:
        return None

    best_idx = int(result.boxes.conf.argmax().item())
    box = result.boxes.xyxy[best_idx].cpu().numpy()
    conf = float(result.boxes.conf[best_idx].item())
    cls_id = int(result.boxes.cls[best_idx].item()) if result.boxes.cls is not None else 0
    label = str(result.names.get(cls_id, cls_id))
    return box, conf, label


def command_from_error(error_x, width_fraction, deadband, stop_width_fraction):
    if width_fraction >= stop_width_fraction:
        return "STOP", "close", RED
    if error_x < -deadband:
        return "TURN LEFT", "left", YELLOW
    if error_x > deadband:
        return "TURN RIGHT", "right", YELLOW
    return "DRIVE FORWARD", "centered", GREEN


def annotate(frame, detection, deadband, stop_width_fraction):
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    image_center_x = width / 2.0

    cv2.line(annotated, (int(image_center_x), 0), (int(image_center_x), height), GRAY, 1)

    if detection is None:
        put_label(annotated, "NO POLE DETECTED", (24, 48), 0.85, DARK_BLUE)
        return annotated, None

    box, conf, label = detection
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    pole_center_x = (x1 + x2) / 2.0
    pole_center_y = (y1 + y2) / 2.0
    error_x = pole_center_x - image_center_x
    width_fraction = (x2 - x1) / width
    command, side, command_color = command_from_error(
        error_x,
        width_fraction,
        deadband,
        stop_width_fraction,
    )

    cv2.rectangle(annotated, (x1, y1), (x2, y2), BLUE, 2)
    cv2.line(annotated, (int(pole_center_x), y1), (int(pole_center_x), y2), BLUE, 1)
    cv2.circle(annotated, (int(pole_center_x), int(pole_center_y)), 5, RED, -1)

    put_label(annotated, f"POLE {conf:.2f}", (x1, max(58, y1 - 10)), 1.5, RED)
    put_label(annotated, f"ERROR: {error_x:+.0f} px", (24, height - 72), 1.5, BLUE)
    put_label(annotated, f"WIDTH FRACTION: {width_fraction:.2f}", (24, height - 32), 1.5, BLUE)
    put_label(annotated, command, (width - 260, height - 40), 1.5, BLUE)

    info = {
        "label": label,
        "confidence": conf,
        "box_xyxy": (x1, y1, x2, y2),
        "error_x": error_x,
        "side": side,
        "width_fraction": width_fraction,
        "command": command,
    }
    return annotated, info


def default_output_path(image_path):
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_OUTPUT_DIR / f"{image_path.stem}_pole_annotated.png"


def main():
    args = parse_args()

    if not args.image.exists():
        raise SystemExit(f"Image does not exist: {args.image}")
    if not args.model.exists():
        raise SystemExit(f"Model does not exist: {args.model}")

    frame = cv2.imread(str(args.image))
    if frame is None:
        raise SystemExit(f"Could not read image: {args.image}")

    frame = rotate_frame(frame, args.rotate)
    model = YOLO(str(args.model))
    result = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
    annotated, info = annotate(frame, best_detection(result), args.deadband, args.stop_width_fraction)

    output = args.output if args.output is not None else default_output_path(args.image)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), annotated)

    if info is None:
        print("No pole detected")
    else:
        print(
            f"{info['command']}: side={info['side']}, "
            f"error_x={info['error_x']:+.1f}px, "
            f"width_fraction={info['width_fraction']:.3f}, "
            f"conf={info['confidence']:.2f}, "
            f"box={info['box_xyxy']}"
        )
    print(f"Saved annotated image to: {output}")

    if not args.no_show:
        cv2.imshow("pole YOLO annotation", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
