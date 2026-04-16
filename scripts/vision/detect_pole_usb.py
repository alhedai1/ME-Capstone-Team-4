from ultralytics import YOLO
import cv2
import time

# model = YOLO("best.pt")
model = YOLO("best_ncnn_model")

cap = cv2.VideoCapture()

time.sleep(2)

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi', fourcc, 10, (640, 480))

try:
    while True:
        frame = picam2.capture_array()

        if frame.shape[2] == 4:
            frame = frame[:, :, :3]

        # image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = model(frame)

        annotated = results[0].plot()

        # running headless, so cannot use cv2.imshow
        annotated = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        cv2.imshow("Detection", annotated)

        # out.write(annotated)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopping...")

finally:
    # out.release()
    picam2.stop()
    cv2.destroyAllWindows()