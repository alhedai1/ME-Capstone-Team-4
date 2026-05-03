#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Run an NCNN YOLO model on a video")
    parser.add_argument("--model", type=Path, default=Path("models/best_ncnn_model"))
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(str(args.model))

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    fps_in = cap.get(cv2.CAP_PROP_FPS)
    if fps_in <= 0:
        fps_in = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(args.output), fourcc, fps_in, (width, height))
        if not writer.isOpened():
            raise SystemExit(f"Could not open output video: {args.output}")

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()
        results = model(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)
        annotated = results[0].plot()

        dt = time.time() - t0
        inst_fps = 1.0 / dt if dt > 0 else 0.0
        cv2.putText(
            annotated,
            f"Infer FPS: {inst_fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        if writer is not None:
            writer.write(annotated)

        if args.show:
            cv2.imshow("NCNN YOLO", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

        frame_count += 1

    cap.release()
    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0.0
    print(f"Frames processed: {frame_count}")
    print(f"Average end-to-end FPS: {avg_fps:.2f}")
    if args.output is not None:
        print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
