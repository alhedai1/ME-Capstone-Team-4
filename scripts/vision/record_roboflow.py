
# Install inference library on pi first
# pip install inference

import cv2
import os
from inference import InferencePipeline

# Create a folder for your images
output_dir = "capstone_data"
os.makedirs(output_dir, exist_ok=True)

# This function just saves the frame and doesn't need a model
def save_frame(video_frames):
    for video_frame in video_frames:
        # Save one frame every 30 frames (about once per second)
        if video_frame.frame_id % 30 == 0:
            filename = f"{output_dir}/frame_{video_frame.frame_id}.jpg"
            cv2.imwrite(filename, video_frame.image)
            print(f"Saved: {filename}")

# Use 'init_with_custom_logic' to run without a model ID
pipeline = InferencePipeline.init_with_custom_logic(
    video_reference=0, 
    on_video_frame=save_frame,
)

print("Capturing frames... Press Ctrl+C to stop.")
pipeline.start()
pipeline.join()
