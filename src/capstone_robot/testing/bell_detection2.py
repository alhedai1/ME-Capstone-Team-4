from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse

DEFAULT_IMAGE_FOLDER = Path("../data/extracted_frames/may25/may25_strike_bell")
# IMG_FOLDER2 = Path("../data/extracted_frames/may26/may26_bell_moving_indoors")

class BellTrigger:
    def __init__(self):
        self.hit_count = 0
        self.miss_count = 0
        self.triggered = False

    def detect(self, frame):
        h, w = frame.shape[:2]

        # Use only the area where the bell is expected to appear.
        # Adjust these numbers based on camera placement.
        x1 = int(0.0 * w)
        x2 = int(1.0 * w)
        y1 = int(0.0 * h)
        y2 = int(1.0 * h)

        roi = frame[y1:y2, x1:x2]

        # Slight blur helps reduce noise
        roi_blur = cv2.GaussianBlur(roi, (5, 5), 0)

        hsv = cv2.cvtColor(roi_blur, cv2.COLOR_BGR2HSV)

        # Warm/gold/brown/yellow regions
        gold_mask = cv2.inRange(
            hsv,
            np.array([8, 35, 40]),
            np.array([35, 255, 255])
        )

        # Bright low-saturation reflections
        bright_mask = cv2.inRange(
            hsv,
            np.array([0, 0, 180]),
            np.array([180, 90, 255])
        )

        # Only accept bright reflections near warm/gold areas
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        gold_nearby = cv2.dilate(gold_mask, kernel, iterations=2)

        valid_bright = cv2.bitwise_and(bright_mask, gold_nearby)

        # Combine masks
        mask = cv2.bitwise_or(gold_mask, valid_bright)

        # Clean up mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find largest blob
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return False, frame, mask

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        roi_area = roi.shape[0] * roi.shape[1]
        area_ratio = area / roi_area

        # Tune this.
        # At 5 cm, the bell should occupy a fairly large part of the ROI.
        close_enough = area_ratio > 0.3

        if close_enough:
            self.hit_count += 1
            self.miss_count = 0
        else:
            self.miss_count += 1
            self.hit_count = max(0, self.hit_count - 1)

        # Require persistence over multiple frames
        if self.hit_count >= 3:
            self.triggered = True

        # Optional debug drawing
        debug = frame.copy()
        cv2.rectangle(debug, (x1, y1), (x2, y2), (255, 255, 0), 2)

        if close_enough:
            x, y, bw, bh = cv2.boundingRect(largest)
            cv2.rectangle(
                debug,
                (x1 + x, y1 + y),
                (x1 + x + bw, y1 + y + bh),
                (0, 255, 0),
                2
            )
            cv2.putText(
                debug,
                f"bell trigger candidate area={area_ratio:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        return self.triggered, debug, mask

class AdaptiveBellCloseTrigger:
    def __init__(self, trigger_frames=3):
        self.hit_count = 0
        self.trigger_frames = trigger_frames

    def detect(self, frame):
        h, w = frame.shape[:2]

        # Restrict to the expected strike region.
        # Tune this based on camera mounting.
        x1 = int(0.15 * w)
        x2 = int(0.85 * w)
        y1 = int(0.10 * h)
        y2 = int(0.95 * h)

        roi = frame[y1:y2, x1:x2]
        roi_area = roi.shape[0] * roi.shape[1]

        blur = cv2.GaussianBlur(roi, (5, 5), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

        H = hsv[:, :, 0]
        S = hsv[:, :, 1]
        V = hsv[:, :, 2]

        # Adaptive thresholds from current frame
        v_high = np.percentile(V, 85)
        s_mid = np.percentile(S, 55)

        # Broad warm/gold/brown cue
        warm_mask = (
            (H >= 5) &
            (H <= 40) &
            (S > max(35, s_mid * 0.8)) &
            (V > 40)
        ).astype(np.uint8) * 255

        # Bright shiny cue
        bright_mask = (
            (V > max(160, v_high)) &
            (S < 120)
        ).astype(np.uint8) * 255

        # Edge cue
        edges = cv2.Canny(gray, 60, 140)

        # Combine cues, but do not let bright alone dominate too much
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

        warm_clean = cv2.morphologyEx(warm_mask, cv2.MORPH_OPEN, kernel)
        warm_clean = cv2.morphologyEx(warm_clean, cv2.MORPH_CLOSE, kernel)

        warm_dilated = cv2.dilate(warm_clean, kernel, iterations=2)
        bright_near_warm = cv2.bitwise_and(bright_mask, warm_dilated)

        candidate = cv2.bitwise_or(warm_clean, bright_near_warm)

        # Add edge support only near candidate areas
        candidate_dilated = cv2.dilate(candidate, kernel, iterations=1)
        edges_near_candidate = cv2.bitwise_and(edges, candidate_dilated)

        # Feature ratios
        warm_ratio = np.count_nonzero(warm_clean) / roi_area
        bright_ratio = np.count_nonzero(bright_near_warm) / roi_area
        edge_ratio = np.count_nonzero(edges_near_candidate) / roi_area

        contours, _ = cv2.findContours(
            candidate,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        largest_blob_ratio = 0.0
        bbox = None

        if contours:
            largest = max(contours, key=cv2.contourArea)
            largest_blob_ratio = cv2.contourArea(largest) / roi_area
            bbox = cv2.boundingRect(largest)

        # Scoring instead of one hard rule
        score = 0

        if warm_ratio > 0.04:
            score += 1

        if bright_ratio > 0.01:
            score += 1

        if edge_ratio > 0.01:
            score += 1

        if largest_blob_ratio > 0.04:
            score += 2

        # Trigger candidate
        close_candidate = score >= 3

        if close_candidate:
            self.hit_count += 1
        else:
            self.hit_count = max(0, self.hit_count - 1)

        triggered = self.hit_count >= self.trigger_frames

        # Debug visualization
        debug = frame.copy()
        cv2.rectangle(debug, (x1, y1), (x2, y2), (255, 255, 0), 2)

        if bbox is not None:
            bx, by, bw, bh = bbox
            cv2.rectangle(
                debug,
                (x1 + bx, y1 + by),
                (x1 + bx + bw, y1 + by + bh),
                (0, 255, 0) if close_candidate else (0, 0, 255),
                2
            )

        text = (
            f"score={score} warm={warm_ratio:.3f} "
            f"bright={bright_ratio:.3f} edge={edge_ratio:.3f} "
            f"blob={largest_blob_ratio:.3f} hits={self.hit_count}"
        )

        cv2.putText(
            debug,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0) if triggered else (0, 0, 255),
            2
        )

        return triggered, debug, candidate

# trigger = BellTrigger()
trigger = AdaptiveBellCloseTrigger()

# while True:
#     frame = get_frame_somehow()

#     bell_detected, debug_frame, mask = trigger.detect(frame)

#     if bell_detected:
#         print("BELL CLOSE: STRIKE NOW")
#         # servo_strike()
#         break

#     cv2.imshow("debug", debug_frame)
#     cv2.imshow("mask", mask)

#     if cv2.waitKey(1) == ord("q"):
#         break

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
# DEFAULT_IMAGE_FOLDER = Path(__file__).resolve().parent.parent / "data" / "extracted_frames" / "may25" / "may25_strike_bell"
# DEFAULT_IMAGE_FOLDER = Path(__file__).resolve().parent.parent / "data" / "extracted_frames" / "may26" / "may26_bell_moving_indoors"
DEFAULT_IMAGE_FOLDER = Path(__file__).resolve().parent.parent / "data" / "extracted_frames" / "may15" / "bell1"


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

        bell_detected, debug_frame, mask = trigger.detect(img)

        if bell_detected:
            print(f"{path.name}: Bell Detected")
            found += 1

        cv2.imshow("debug", debug_frame)
        cv2.imshow("mask", mask)
        if cv2.waitKey(0) == ord("q"):
            break

    cv2.destroyAllWindows()
    print(f"Detected bell in {found}/{len(paths)} images")


if __name__ == "__main__":
    main()
