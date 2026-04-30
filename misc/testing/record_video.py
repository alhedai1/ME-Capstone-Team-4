from picamera2 import Picamera2
import cv2
import time

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration())
picam2.start()

time.sleep(2)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 10, (640, 480))

try:
    while True:
        frame = picam2.capture_array()

        if frame.shape[2] == 4:
            frame = frame[:, :, :3]

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        out.write(frame)

        # key = cv2.waitKey(1)
        # if key == ord('q'):
        #     break

except KeyboardInterrupt:
    print("Stopping...")

finally:
    out.release()
    picam2.stop()
    cv2.destroyAllWindows()