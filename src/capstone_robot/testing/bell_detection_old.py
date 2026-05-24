import argparse
from pathlib import Path

import cv2
import numpy as np


# LOWER_GOLD = np.array([15, 60, 60])
# UPPER_GOLD = np.array([30, 255, 255])
LOWER_GOLD = np.array([15, 30, 30])
UPPER_GOLD = np.array([30, 255, 255])
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def detect_bell(img, min_area=2000, center_tol=0.15, min_width=0.20, min_height=0.20):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_GOLD, UPPER_GOLD)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < min_area:
        return None, mask

    img_h, img_w = img.shape[:2]
    x, y, w, h = cv2.boundingRect(contour)
    bell_cx = x + w / 2
    bell_cy = y + h / 2
    frame_cx = img_w / 2
    frame_cy = img_h / 2

    if abs(bell_cx - frame_cx) > center_tol * img_w:
        return None, mask
    if abs(bell_cy - frame_cy) > center_tol * img_h:
        return None, mask
    if w < min_width * img_w or h < min_height * img_h:
        return None, mask

    return (x, y, w, h, area), mask


def draw_result(img, result):
    out = img.copy()
    if result is None:
        cv2.putText(out, "No Bell Found", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return out

    x, y, w, h, area = result
    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.putText(out, f"Bell: {int(area)}px", (x, max(25, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return out


def image_paths(folder):
    return sorted(path for path in Path(folder).iterdir() if path.suffix.lower() in IMAGE_EXTS)


def main():
    parser = argparse.ArgumentParser(description="Run simple HSV bell detection on a folder of images.")
    parser.add_argument("folder", help="Folder containing images")
    parser.add_argument("--out", help="Optional folder for annotated results")
    parser.add_argument("--show", action="store_true", help="Show each result; press q or Esc to stop")
    parser.add_argument("--min-area", type=float, default=2000, help="Minimum bell contour area")
    parser.add_argument("--center-tol", type=float, default=0.2, help="How close the bell center must be to frame center")
    parser.add_argument("--min-width", type=float, default=0.20, help="Minimum detection width as fraction of image width")
    parser.add_argument("--min-height", type=float, default=0.20, help="Minimum detection height as fraction of image height")
    args = parser.parse_args()

    paths = image_paths(args.folder)
    if not paths:
        raise FileNotFoundError(f"No images found in {args.folder}")

    out_dir = Path(args.out) if args.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    found = 0
    for path in paths:
        img = cv2.imread(str(path))
        if img is None:
            print(f"SKIP {path.name}: could not read")
            continue

        result, _ = detect_bell(
            img,
            min_area=args.min_area,
            center_tol=args.center_tol,
            min_width=args.min_width,
            min_height=args.min_height,
        )
        found += result is not None
        print(f"{path.name}: {'FOUND' if result else 'not found'}")

        vis = draw_result(img, result)
        if out_dir:
            cv2.imwrite(str(out_dir / path.name), vis)
        if args.show:
            cv2.imshow("bell detection", vis)
            if cv2.waitKey(0) & 0xFF in (ord("q"), 27):
                break

    cv2.destroyAllWindows()
    print(f"Detected bell in {found}/{len(paths)} images")


if __name__ == "__main__":
    main()
