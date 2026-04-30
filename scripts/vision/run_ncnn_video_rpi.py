# run_ncnn_video.py
import cv2
import time
from ultralytics import YOLO

# =========================
# Config
# =========================
MODEL_PATH = "../models/best_ncnn_model_sz320"
VIDEO_PATH = "/home/team4/capstone/vision/pole3.mp4"
OUTPUT_PATH = "/home/team4/capstone/vision/detections/pole3.avi"  # use avi first
IMGSZ = 320
CONF = 0.4
SHOW = False
USE_TRACKING_FPS_TEXT = True

# set this after checking the frame orientation
ROTATE_MODE = "cw"   # options: "none", "cw", "ccw", "180"

def apply_rotation(frame):
    if ROTATE_MODE == "cw":
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif ROTATE_MODE == "ccw":
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif ROTATE_MODE == "180":
        return cv2.rotate(frame, cv2.ROTATE_180)
    return frame

# =========================
# Load model
# =========================
model = YOLO(MODEL_PATH, task="detect")

# =========================
# Open video
# =========================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

# disable backend auto-rotation if supported
if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)

fps_in = cap.get(cv2.CAP_PROP_FPS)
if fps_in <= 0:
    fps_in = 30.0

# read first frame first
ret, frame = cap.read()
if not ret:
    raise RuntimeError("Could not read first frame")

# apply manual rotation
frame = apply_rotation(frame)

# debug
cv2.imwrite("debug_first_frame.jpg", frame)

# get ACTUAL size from frame, not from CAP_PROP_FRAME_WIDTH/HEIGHT
height, width = frame.shape[:2]
print(f"actual frame size: {width} x {height}")

writer = None
if OUTPUT_PATH is not None:
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps_in, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video: {OUTPUT_PATH}")

# =========================
# Inference loop
# =========================
frame_count = 0
start_time = time.time()

while True:
    t0 = time.time()

    results = model(
        frame,
        imgsz=IMGSZ,
        conf=CONF,
        verbose=False
    )

    annotated = results[0].plot()

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

    ret, frame = cap.read()
    if not ret:
        break

    frame = apply_rotation(frame)

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

print("Done.")
print(f"Frames processed: {frame_count}")
print(f"Average end-to-end FPS: {avg_fps:.2f}")
if OUTPUT_PATH is not None:
    print(f"Saved to: {OUTPUT_PATH}")