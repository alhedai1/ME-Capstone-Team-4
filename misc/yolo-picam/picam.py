import cv2
from picamera2 import Picamera2
import time

# Initialize the Picamera2 camera object
camera = Picamera2()

# Configure the camera to capture in the default resolution
camera.configure(camera.create_preview_configuration())

# Start the camera
camera.start()

# Allow the camera to warm up for 2 seconds
time.sleep(2)

# Infinite loop for continuous image capture
while True:
    # Capture a single frame (image)
    image = camera.capture_array()

    # Convert the captured image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Optionally, show the image in a window
    cv2.imshow("Captured Image", image_rgb)

    # Wait for 1 ms to handle any window events (like closing)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit the loop
        break

# Close the window and stop the camera
cv2.destroyAllWindows()
camera.stop()

# Notify the user that the loop has ended
print("Image capture loop stopped.")