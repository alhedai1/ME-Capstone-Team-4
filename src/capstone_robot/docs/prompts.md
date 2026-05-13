
Prompt 1
You are working inside my robotics capstone repo.

First, inspect the current repo structure. Do not rewrite the whole project.

Goal:
Add a small, clean testing area for vision experiments without changing existing working code.

Please:
1. Create a scripts/ directory if it does not exist.
2. Create a data/test_videos/ directory if it does not exist.
3. Create a data/test_outputs/ directory if it does not exist.
4. Create a models/ directory if it does not exist.
5. Add a short README section or docs/vision_testing.md explaining the purpose of these test scripts.

Important:
- Do not create ROS2 nodes yet.
- Do not modify existing robot control code unless absolutely necessary.
- Keep changes minimal.
- After editing, show me the files you created/modified.

Prompt 2
Add a standalone Python script at scripts/record_video.py.

Purpose:
Record video from a camera so I can collect test videos for my robot vision alignment experiments.

Requirements:
- Use OpenCV.
- Support command line arguments:
  --source: camera index or video source, default 0
  --output: output video path, default data/test_videos/recording.mp4
  --width: default 640
  --height: default 480
  --fps: default 30
  --rotate: choices none, cw, ccw, 180, default none
  --show: display preview window
- Save the video to the output path.
- Draw timestamp/frame number on preview if --show is used.
- Press q to stop recording when preview is open.
- If preview is not open, record until Ctrl+C.
- Create parent output directory automatically.
- Print the saved output path when finished.
- Keep the code simple and readable.
- Add basic error handling if the camera cannot be opened.
- Do not use ROS2.

Prompt 3
Add a standalone Python script at scripts/test_front_alignment.py.

Context:
My robot starts on the ground 2–4 m away from a 3 m pole. A rod extends horizontally from the top of the pole, and a bell hangs from the end. The normal Pi camera will be tilted slightly upward. I want to test if it can see the pole and rod/bell, then compute whether the robot is aligned with the correct side of the pole.

Goal:
Run YOLO on a camera or video. Detect pole, bell, and/or rod. Compute x_error = target_center_x - pole_center_x, where target is bell if detected, otherwise rod.

Requirements:
- Use Ultralytics YOLO.
- Support command line arguments:
  --source: camera index or video path, default 0
  --model: path to YOLO .pt model, required
  --conf: confidence threshold, default 0.4
  --imgsz: default 640
  --save-output: optional output video path
  --show: show annotated preview
  --rotate: choices none, cw, ccw, 180, default none
  --aligned-threshold-px: default 40
- The script should work with either a live camera or video file.
- Choose the highest-confidence detection for class "pole".
- Choose target as highest-confidence "bell" if available, otherwise highest-confidence "rod".
- Compute:
  pole_center_x
  target_center_x
  x_error = target_center_x - pole_center_x
- Display status:
  MISSING_POLE
  MISSING_TARGET
  ALIGNED
  TARGET_RIGHT_OF_POLE
  TARGET_LEFT_OF_POLE
- Draw:
  bounding boxes
  class names/confidences
  vertical line at pole center
  vertical line at target center
  x_error text
  status text
- Print one line per frame or every few frames with:
  frame index, status, pole_x, target_x, x_error
- If --save-output is given, save annotated video.
- Keep logic modular:
  center_x(box)
  pick_best_detection(results, class_name)
  compute_alignment(...)
  draw_overlay(...)
- Do not create ROS2 nodes yet.
- Do not modify existing code except adding this script and maybe updating docs.

Prompt 4
Add a standalone Python script at scripts/test_upward_rod_angle.py.

Context:
My robot has a Pi AI camera pointing vertically upward while the robot is on the ground. Near the pole, this camera should see the top rod from below. Before climbing, I want to check if the robot is oriented correctly by detecting the rod angle in the upward camera frame. Later, after the robot climbs and rotates 90 degrees, this same AI camera will point horizontally and detect the bell.

Goal:
Run YOLO on a camera or video. Detect the rod, estimate its image angle, and compare it to a desired reference angle.

Requirements:
- Use Ultralytics YOLO.
- Support command line arguments:
  --source: camera index or video path, default 0
  --model: path to YOLO .pt model, required
  --conf: default 0.4
  --imgsz: default 640
  --desired-angle-deg: default 90
  --angle-threshold-deg: default 10
  --save-output: optional output video path
  --show: show annotated preview
  --rotate: choices none, cw, ccw, 180, default none

Detection/angle logic:
- Prefer class "rod".
- Use the best rod detection.
- Estimate rod angle from the bounding box shape:
  - If the box is taller than wide, angle is approximately 90 degrees.
  - If the box is wider than tall, angle is approximately 0 degrees.
- Also include a placeholder function estimate_rod_angle_from_mask_or_line(frame, detection) for future improvement with segmentation/Hough lines.
- Compute angle_error = normalize_angle_deg(rod_angle - desired_angle_deg), normalized to [-90, 90] or similar.
- Status:
  MISSING_ROD
  ANGLE_ALIGNED
  ROTATE_LEFT_OR_RIGHT_UNKNOWN_SIGN
- Since the sign may depend on camera mounting, clearly print that the correction sign must be calibrated experimentally.
- Draw:
  rod bounding box
  estimated angle
  desired angle
  angle error
  status text
- Save annotated video if --save-output is given.
- Keep this non-ROS and standalone.

Prompt 5
Add a standalone Python script at scripts/batch_test_alignment.py.

Goal:
Run the front alignment test over a folder of saved images/videos and write a CSV summary, so I can compare different camera tilt angles and starting positions.

Requirements:
- Use the reusable logic from scripts/test_front_alignment.py if possible, or refactor shared functions into scripts/vision_utils.py.
- Input arguments:
  --input-dir: default data/test_videos
  --model: path to YOLO .pt model, required
  --conf: default 0.4
  --imgsz: default 640
  --output-csv: default data/test_outputs/front_alignment_summary.csv
  --sample-every-n-frames: default 10
- Process common image/video formats:
  images: .jpg, .jpeg, .png
  videos: .mp4, .avi, .mov, .mkv
- For each image or sampled video frame, compute:
  file name
  frame index
  status
  pole_detected
  target_detected
  pole_x
  target_x
  x_error
- Save results to CSV.
- Print a short summary:
  total frames/images processed
  percent with pole detected
  percent with target detected
  percent aligned
- Keep code simple.
- Do not create ROS2 nodes yet.

Prompt 6
Now convert the working front alignment logic into a ROS2 Python node.

Context:
The standalone script scripts/test_front_alignment.py works. I now want a ROS2 node version.

Goal:
Create a ROS2 node called front_alignment_node inside my existing ROS2 package if one exists. If no package exists, ask me before creating one.

Requirements:
- Subscribe to a camera image topic, default /front/image_raw.
- Use cv_bridge to convert sensor_msgs/Image to OpenCV image.
- Run YOLO using the same model/alignment logic as scripts/test_front_alignment.py.
- Publish:
  /front_alignment/error_px as std_msgs/Float32
  /front_alignment/status as std_msgs/String
  optionally /front_alignment/debug_image as sensor_msgs/Image
- Parameters:
  model_path
  conf
  imgsz
  aligned_threshold_px
  image_topic
  publish_debug_image
- Reuse shared code where possible instead of duplicating all logic.
- Do not implement robot motion yet.
- Add an entry point in setup.py if this is ament_python.
- Add a short launch file if appropriate.
- Keep changes minimal and explain how to run it.

Prompt 7
Now convert the working upward rod-angle logic into a ROS2 Python node.

Context:
The standalone script scripts/test_upward_rod_angle.py works. I want a ROS2 node for near-pole final alignment using the upward-facing Pi AI camera.

Goal:
Create a ROS2 node called upward_alignment_node inside my existing ROS2 package.

Requirements:
- Subscribe to image topic, default /upward/image_raw.
- Use cv_bridge.
- Run YOLO and estimate rod angle using the same logic from scripts/test_upward_rod_angle.py.
- Publish:
  /upward_alignment/angle_error_deg as std_msgs/Float32
  /upward_alignment/status as std_msgs/String
  optionally /upward_alignment/debug_image as sensor_msgs/Image
- Parameters:
  model_path
  conf
  imgsz
  desired_angle_deg
  angle_threshold_deg
  image_topic
  publish_debug_image
- Include a clear comment that the correction direction/sign must be calibrated experimentally because it depends on camera mounting.
- Do not implement robot motion yet.
- Add setup.py entry point and launch file only if the existing package structure supports it.
- Keep changes minimal and explain how to run it.

Prompt 8
Create a simple ROS2 ground alignment controller node.

Context:
I have:
- /front_alignment/error_px as std_msgs/Float32
- /front_alignment/status as std_msgs/String
- later /upward_alignment/angle_error_deg as std_msgs/Float32

Goal:
Create a cautious controller that only prints suggested motion first, with an optional parameter to publish /cmd_vel.

Requirements:
- Node name: ground_alignment_controller
- Subscribe:
  /front_alignment/error_px
  /front_alignment/status
- Publish:
  /cmd_vel as geometry_msgs/Twist only if parameter enable_motion is true
- Parameters:
  enable_motion default false
  aligned_threshold_px default 40
  angular_gain default 0.002
  max_angular_speed default 0.25
  forward_speed default 0.05
- Logic:
  If status is MISSING_POLE or MISSING_TARGET, rotate slowly/search.
  If status is TARGET_LEFT_OF_POLE or TARGET_RIGHT_OF_POLE, rotate proportionally to reduce error.
  If status is ALIGNED, drive forward slowly.
- Include clear safety comments.
- Motion should be slow and easy to stop.
- Do not handle climbing or striking.
- Explain how to run it safely with enable_motion:=false first.