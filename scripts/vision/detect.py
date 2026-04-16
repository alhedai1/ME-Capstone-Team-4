from ultralytics import YOLO
import cv2
import os

model = YOLO("yolov8n.pt")

image_folder = "bell_images"

cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)

for filename in os.listdir(image_folder):
    image_path = os.path.join(image_folder, filename)

    frame = cv2.imread(image_path)
    if frame is None:
        continue

    results = model(frame)

    annotated = results[0].plot()

    cv2.imshow("Detection", annotated)
    print(f"Showing: {filename}")

    key = cv2.waitKey(0)

    if key == ord('q'):
        break

cv2.destroyAllWindows()