# run_ncnn_video.py
import cv2
import time
from ultralytics import YOLO

# =========================
# Config
# =========================
### on rpi
# MODEL_PATH = "models/best_ncnn_model"   # folder containing model.ncnn.param / model.ncnn.bin
### on pc
MODEL_PATH = "/home/ahmed/Other/capstone/train/runs/detect/yolo26n_sz320/weights/best_ncnn_model_sz320"

VIDEO_PATH = "/home/ahmed/Other/capstone/data/videos/pole2.mp4"
OUTPUT_PATH = "/home/ahmed/Other/capstone/data/videos/detections/pole2.mp4"              # set to None if you do not want to save
IMGSZ = 320                             # try 320 on RPi 4 for better speed
CONF = 0.4
SHOW = False                            # usually False on headless Pi
USE_TRACKING_FPS_TEXT = True

# =========================
# Load model
# =========================
model = YOLO(MODEL_PATH)

# =========================
# Open video
# =========================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

fps_in = cap.get(cv2.CAP_PROP_FPS)
if fps_in <= 0:
    fps_in = 30.0

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"width is: {width}")

writer = None
if OUTPUT_PATH is not None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps_in, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video: {OUTPUT_PATH}")

# =========================
# Inference loop
# =========================
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    t0 = time.time()

    # Run inference on one frame
    results = model(
        frame,
        imgsz=IMGSZ,
        conf=CONF,
        verbose=False
    )

    # Draw detections
    annotated = results[0].plot()

    # FPS text
    if USE_TRACKING_FPS_TEXT:
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
            cv2.LINE_AA
        )

    if writer is not None:
        writer.write(annotated)

    if SHOW:
        cv2.imshow("NCNN YOLO", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    frame_count += 1

# =========================
# Cleanup
# =========================
cap.release()
if writer is not None:
    writer.release()
if SHOW:
    cv2.destroyAllWindows()

total_time = time.time() - start_time
avg_fps = frame_count / total_time if total_time > 0 else 0.0

print(f"Done.")
print(f"Frames processed: {frame_count}")
print(f"Average end-to-end FPS: {avg_fps:.2f}")
if OUTPUT_PATH is not None:
    print(f"Saved to: {OUTPUT_PATH}")