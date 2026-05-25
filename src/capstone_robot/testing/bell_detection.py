import argparse
from pathlib import Path

import cv2

from capstone_robot.vision.bell import detect_bell


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
DEFAULT_IMAGE_FOLDER = Path(__file__).resolve().parent.parent / "data" / "extracted_frames" / "may24" / "bell_indoors"


def draw_result(img, result):
    out = img.copy()
    if result is None:
        cv2.putText(out, "No Bell Found", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return out

    x, y, w, h = result.box
    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 3)
    label = f"Bell: area={result.area_fraction:.2f} metal={result.brass_ratio:.2f} hi={result.warm_brass_ratio:.2f}"
    cv2.putText(out, label, (x, max(25, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    return out


def image_paths(folder):
    return sorted(path for path in Path(folder).iterdir() if path.suffix.lower() in IMAGE_EXTS)


def main():
    parser = argparse.ArgumentParser(description="Run close reflective-bell detection on a folder of images.")
    parser.add_argument("folder", nargs="?", default=DEFAULT_IMAGE_FOLDER, help="Folder containing images")
    parser.add_argument("--out", help="Optional folder for annotated results")
    parser.add_argument("--show", action="store_true", help="Show each result; press q or Esc to stop")
    parser.add_argument("--min-area-fraction", type=float, default=0.12, help="Minimum component area as frame fraction")
    parser.add_argument("--min-width-fraction", type=float, default=0.35, help="Minimum bbox width as frame fraction")
    parser.add_argument("--min-height-fraction", type=float, default=0.28, help="Minimum bbox height as frame fraction")
    parser.add_argument("--max-orange-frame-fraction", type=float, default=0.22, help="Reject orange ball-dominant frames")
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

        result = detect_bell(
            img,
            color_format="bgr",
            min_area_fraction=args.min_area_fraction,
            min_width_fraction=args.min_width_fraction,
            min_height_fraction=args.min_height_fraction,
            max_orange_frame_fraction=args.max_orange_frame_fraction,
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
