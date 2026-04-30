from picamera2 import Picamera2
import cv2
import time

picam2 = Picamera2()

config = picam2.create_video_configuration(
    main={"size": (320, 240), "format": "RGB888"},
    controls={"FrameRate": 60.0},
    buffer_count=1,
)

picam2.configure(config)
picam2.start()

time.sleep(1.0)

print("Pi camera started")

prev_time = time.time()
frame_count = 0

while True:
    frame_rgb = picam2.capture_array()

    # Picamera2 gives RGB; convert for OpenCV display (dont convert actually)
    #frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    frame_count += 1
    now = time.time()

    if now - prev_time >= 1.0:
        print("capture fps:", frame_count)
        frame_count = 0
        prev_time = now

    cv2.imshow("Pi Camera", frame_rgb)

    if cv2.waitKey(1) == ord("q"):
        break

picam2.stop()
cv2.destroyAllWindows()
