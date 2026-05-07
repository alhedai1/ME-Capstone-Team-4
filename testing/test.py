from pathlib import Path
import cv2
import time
from ultralytics import YOLO
import git

repo = git.Repo('.', search_parent_directories=True)
root_path = repo.working_tree_dir

print(root_path)


model_path = root_path + "/models/best_ncnn_model_sz320"
video_path = root_path + "/data/videos/may_4_outdoors/20260504_182911.mp4"
# video_path = root_path + "/data/videos/phone1.mp4"

model = YOLO(model_path, task='detect')
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30.0
frame_delay = 1.0 / fps

print(f"Video FPS: {fps:.2f}")

while True:
    start_time = time.time()
    ret, frame = cap.read()

    # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    
    if not ret or frame is None:
        print("No frame.")
        break
    
    results = model(frame, imgsz=320)
    annotated = results[0].plot()
    cv2.imshow('detection', annotated)

    elapsed = time.time() - start_time
    wait_ms = max(1, int((frame_delay - elapsed) * 1000))

    if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
