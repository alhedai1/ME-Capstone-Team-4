import argparse
import cv2
from pathlib import Path
from ultralytics import YOLO

class_names = {
    0: "bell",
    1: "pole",
    2: "rod",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Show images with YOLO detections or ground truth labels")
    parser.add_argument("--images-dir", type=Path, required=True, help="Folder containing images")
    parser.add_argument("--labels-dir", type=Path, help="Folder containing YOLO .txt labels (if not using model)")
    parser.add_argument("--model", type=Path, help="YOLO model path (.pt or exported) to run inference")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold for model inference")
    return parser.parse_args()


def load_yolo_labels(label_path, img_w, img_h):
    boxes = []
    if not label_path.exists():
        return boxes

    with open(label_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue

        class_id = int(float(parts[0]))
        x_center = float(parts[1]) * img_w
        y_center = float(parts[2]) * img_h
        width = float(parts[3]) * img_w
        height = float(parts[4]) * img_h

        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        boxes.append((class_id, x1, y1, x2, y2))

    return boxes


def run_inference(model, img_path, conf):
    results = model.predict(str(img_path), conf=conf, verbose=False)
    boxes = []
    if results and len(results) > 0:
        result = results[0]
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                boxes.append((class_id, int(x1), int(y1), int(x2), int(y2)))
    return boxes


def main():
    args = parse_args()
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    image_paths = sorted(
        [p for p in args.images_dir.iterdir() if p.is_file() and p.suffix.lower() in image_extensions]
    )

    if not image_paths:
        print(f"No images found in {args.images_dir}")
        return

    model = None
    if args.model:
        print(f"Loading model: {args.model}")
        model = YOLO(str(args.model))
        class_names.update(model.names)  # Update with model class names if available
        mode = "inference"
    else:
        if not args.labels_dir:
            print("Either --model or --labels-dir must be provided")
            return
        mode = "labels"

    for image_path in image_paths:
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Could not read image: {image_path}")
            continue

        h, w = img.shape[:2]

        if mode == "inference":
            boxes = run_inference(model, image_path, args.conf)
        else:
            label_path = args.labels_dir / f"{image_path.stem}.txt"
            boxes = load_yolo_labels(label_path, w, h)

        for class_id, x1, y1, x2, y2 in boxes:
            class_name = class_names.get(class_id, str(class_id))
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img,
                class_name,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("YOLO Visualization", img)
        print(f"Showing: {image_path.name} ({mode})")
        print("Press any key for next image, or q to quit.")

        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
