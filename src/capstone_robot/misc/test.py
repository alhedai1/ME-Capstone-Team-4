from ultralytics import YOLO
import cv2

model = YOLO("best.pt")   # your trained bell model

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            conf = float(box.conf[0])

            if conf > 0.7:
                print("Bell detected!")

    cv2.imshow("frame", results[0].plot())

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()