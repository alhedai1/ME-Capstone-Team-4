import argparse
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
DEFAULT_IMAGE_FOLDER = Path(__file__).resolve().parent.parent / "data" / "extracted_frames" / "may15" / "bell1"

BRASS_HSV_LOW = np.array([8, 45, 50])
BRASS_HSV_HIGH = np.array([34, 255, 255])
BRASS_LAB_LOW = np.array([35, 102, 132])
BRASS_LAB_HIGH = np.array([255, 155, 195])

NEUTRAL_HIGHLIGHT_LOW = np.array([0, 0, 85])
NEUTRAL_HIGHLIGHT_HIGH = np.array([179, 105, 255])
WARM_SHADOW_LAB_LOW = np.array([20, 100, 125])
WARM_SHADOW_LAB_HIGH = np.array([235, 160, 190])


def build_bell_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    blue, green, red = cv2.split(img.astype(np.float32))

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


def detect_bell(
    img,
    min_area=2500,
    min_fill=0.12,
    min_brass_ratio=0.015,
    min_warm_brass_ratio=0.04,
):
    mask, brass_seed, warm_brass_seed = build_bell_mask(img)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    img_h, img_w = mask.shape
    best = None
    best_score = -1

    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area < min_area or w < 45 or h < 45:
            continue

        fill = area / float(w * h)
        aspect = w / float(h)
        if fill < min_fill or not (0.65 <= aspect <= 4.5):
            continue

        component = np.uint8(labels == label) * 255
        brass_pixels = cv2.countNonZero(cv2.bitwise_and(brass_seed, component))
        brass_ratio = brass_pixels / float(area)
        warm_brass_pixels = cv2.countNonZero(cv2.bitwise_and(warm_brass_seed, component))
        warm_brass_ratio = warm_brass_pixels / float(area)
        if brass_ratio < min_brass_ratio or warm_brass_ratio < min_warm_brass_ratio:
            continue

        cx, cy = centroids[label]
        center_score = 1.0 - min(1.0, abs(cx - img_w * 0.52) / (img_w * 0.65))
        upper_score = 1.0 - min(1.0, max(0.0, cy - img_h * 0.72) / (img_h * 0.28))
        score = area * (0.7 + fill) * (0.65 + center_score) * (0.8 + 0.2 * upper_score)

        if score > best_score:
            best_score = score
            best = {
                "bbox": (int(x), int(y), int(w), int(h)),
                "area": int(area),
                "fill": float(fill),
                "aspect": float(aspect),
                "brass_ratio": float(brass_ratio),
                "warm_brass_ratio": float(warm_brass_ratio),
                "center": (float(cx), float(cy)),
                "score": float(score),
            }

    return best, mask


def draw_result(img, result):
    out = img.copy()
    if result is None:
        cv2.putText(out, "No Bell Found", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return out

    x, y, w, h = result["bbox"]
    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 3)
    label = f"Bell: {result['area']}px fill={result['fill']:.2f}"
    cv2.putText(out, label, (x, max(25, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out


def image_paths(folder):
    return sorted(path for path in Path(folder).iterdir() if path.suffix.lower() in IMAGE_EXTS)


def main():
    parser = argparse.ArgumentParser(description="Run front-facing brass bell detection on a folder of images.")
    parser.add_argument(
        "folder",
        nargs="?",
        default=DEFAULT_IMAGE_FOLDER,
        help="Folder containing images",
    )
    parser.add_argument("--out", help="Optional folder for annotated results")
    parser.add_argument("--show", action="store_true", help="Show each result; press q or Esc to stop")
    parser.add_argument("--min-area", type=float, default=2500, help="Minimum bell component area")
    parser.add_argument("--min-fill", type=float, default=0.12, help="Minimum component fill ratio")
    parser.add_argument("--min-brass-ratio", type=float, default=0.015, help="Minimum brass seed ratio")
    parser.add_argument("--min-warm-brass-ratio", type=float, default=0.04, help="Minimum warm brass seed ratio")
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
            min_fill=args.min_fill,
            min_brass_ratio=args.min_brass_ratio,
            min_warm_brass_ratio=args.min_warm_brass_ratio,
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
