import cv2
from pathlib import Path

# ====== CONFIG ======
images_dir = Path("/home/ahmed/Other/capstone/data/roboflow/dataset/images/train")
labels_dir = Path("/home/ahmed/Other/capstone/data/roboflow/dataset/labels/train")

class_names = {
    0: "bell",
    1: "pole",
}
# ====================


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


def main():
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    image_paths = sorted(
        [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in image_extensions]
    )

    if not image_paths:
        print("No images found.")
        return

    for image_path in image_paths:
        label_path = labels_dir / f"{image_path.stem}.txt"

        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Could not read image: {image_path}")
            continue

        h, w = img.shape[:2]
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

        cv2.imshow("YOLO Label Visualization", img)

        print(f"Showing: {image_path.name}")
        print("Press any key for next image, or q to quit.")

        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()