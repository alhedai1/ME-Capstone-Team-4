from ultralytics import YOLO
import cv2
import time

model = YOLO("best_ncnn_model")

cap = cv2.VideoCapture(1)


# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi', fourcc, 10, (640, 480))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    annotated = results[0].plot()

    cv2.imshow("Detection", annotated)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()