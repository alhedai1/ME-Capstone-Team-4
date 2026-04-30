import argparse
from threading import Thread

import cv2
from ultralytics import YOLO


class VideoStream:
    """Threaded class to handle camera capture."""

    def __init__(self, src=0, width=320, height=240):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False

        if not self.grabbed:
            raise RuntimeError(f"Could not read from camera source {src!r}")

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            self.grabbed, self.frame = self.stream.read()

    def read(self):
        if not self.grabbed:
            return None
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()


def parse_args():
    parser = argparse.ArgumentParser(description="Run USB camera detection and optionally save the output video.")
    parser.add_argument("command", nargs="?", choices=["save", "mp4"], help='pass "save" to record AVI output or "mp4" to record MP4 output')
    parser.add_argument("--camera", type=int, default=0, help="USB camera index")
    parser.add_argument("--model", default="../models/best_ncnn_model_sz320", help="path to the YOLO model")
    parser.add_argument("--width", type=int, default=320, help="capture width")
    parser.add_argument("--height", type=int, default=240, help="capture height")
    parser.add_argument("--skip-rate", type=int, default=3, help="run inference every N frames")
    parser.add_argument("--output", default=None, help="saved video path; defaults to output.avi for save and output.mp4 for mp4")
    parser.add_argument("--fps", type=float, default=10.0, help="saved video FPS when using save or mp4")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)
    vs = VideoStream(src=args.camera, width=args.width, height=args.height).start()

    first_frame = None
    while first_frame is None:
        first_frame = vs.read()

    frame_height, frame_width = first_frame.shape[:2]

    writer = None
    if args.command in {"save", "mp4"}:
        if args.command == "mp4":
            output_path = args.output or "output.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        else:
            output_path = args.output or "output.avi"
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")

        writer = cv2.VideoWriter(output_path, fourcc, args.fps, (frame_width, frame_height))
        if not writer.isOpened():
            vs.stop()
            raise RuntimeError(f"Could not open video writer for {output_path!r}")

    frame_count = 0
    annotated_frame = None

    try:
        while True:
            frame = first_frame if frame_count == 0 else vs.read()
            if frame is None:
                continue

            if frame_count % args.skip_rate == 0:
                results = model(frame, imgsz=320, stream=True, verbose=False, conf=0.7)
                for result in results:
                    annotated_frame = result.plot()

            frame_count += 1

            display_frame = annotated_frame if annotated_frame is not None else frame
            cv2.imshow("Detection", display_frame)

            if writer is not None:
                writer.write(display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        vs.stop()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
