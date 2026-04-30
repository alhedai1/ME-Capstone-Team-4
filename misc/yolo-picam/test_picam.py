from ultralytics import YOLO
import cv2
from picamera2 import Picamera2
import time

camera = Picamera2()

camera.configure(camera.create_preview_configuration())

model = YOLO("best_ncnn_model", task="detect")

frame_id = 0

camera.start()

time.sleep(2)

while True:
    image = camera.capture_array()

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    

    frame_id += 1
    if frame_id % 3 != 0:
        continue

    results = model(frame, conf=0.5, verbose=False)

    annotated = results[0].plot()
    cv2.imshow("Detection", annotated)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
