import cv2
import os

video_folder = "/home/ahmed/Other/capstone/data/videos"
output_folder = "/home/ahmed/Other/capstone/data/extracted_frames/phone2"

os.makedirs(output_folder, exist_ok=True)

frame_skip = 15
saved_count = 0

# video_file = "20260403_172554.mp4"
video_file = "20260403_172712.mp4"

# for video_file in os.listdir(video_folder):
video_path = os.path.join(video_folder, video_file)

cap = cv2.VideoCapture(video_path)

frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if frame_count % frame_skip == 0:
        filename = f"{os.path.splitext(video_file)[0]}_{saved_count:05d}.jpg"
        save_path = os.path.join(output_folder, filename)

        cv2.imwrite(save_path, frame)
        saved_count += 1

    frame_count += 1

cap.release()

print(f"Saved {saved_count} images")