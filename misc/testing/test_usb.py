import cv2
import time

cap = cv2.VideoCapture(1, cv2.CAP_V4L2)

if not cap.isOpened():
    raise RuntimeError("Could not open USB camera")

#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
#cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Try MJPG; if ignored, OpenCV will keep something else
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

print("opened =", cap.isOpened())
print("backend =", cap.getBackendName())
print("width =", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("height =", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("fps =", cap.get(cv2.CAP_PROP_FPS))

fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
fmt = "".join([chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)])
print("fourcc =", fmt)

prev_time = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break

    frame_count += 1
    now = time.time()

    if now - prev_time >= 1.0:
        print("capture fps:", frame_count)
        frame_count = 0
        prev_time = now

    cv2.imshow("USB Camera", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
