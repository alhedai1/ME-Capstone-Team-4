import cv2
from picamera2 import Picamera2
import time
from ultralytics import YOLO

# Load the NCNN model
ncnn_model = YOLO("yolo26n_ncnn_model")

# Get the class names directly from the YOLO model
class_names = ncnn_model.names  # This automatically gets the class names from the model

# Initialize the Picamera2 camera object
camera = Picamera2()

# Configure the camera to capture in the default resolution
camera.configure(camera.create_preview_configuration())

# Start the camera
camera.start()

# Infinite loop for continuous image capture
while True:
    # Capture a single frame (image)
    image = camera.capture_array()

    # Convert the captured image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform inference using the NCNN model
    results = ncnn_model(image_rgb)

    # Optionally, you can show the image with the detection results
    # For example, draw bounding boxes and labels
    for result in results:
        boxes = result.boxes  # Bounding boxes for detected objects
        for box in boxes:
            # Get the coordinates and label of the box
            x1, y1, x2, y2 = box.xyxy[0]
            class_id = int(box.cls[0])  # Get the class ID
            label = class_names[class_id]  # Get the class name using the class ID
            # Draw the bounding box and label on the image
            cv2.rectangle(image_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image_rgb, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Optionally, show the image with the detections
    cv2.imshow("Captured Image with Detection", image_rgb)

    # Wait for 1 ms to handle any window events (like closing)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit the loop
        break

# Close the window and stop the camera
cv2.destroyAllWindows()
camera.stop()

# Notify the user that the loop has ended
print("Image capture loop stopped.")