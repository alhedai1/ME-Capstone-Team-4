import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
from time import sleep

from capstone_robot.utils import *
from capstone_robot.vision.pole_bell import PoleBellTracker

REPO_ROOT = find_repo_root(__file__)
IMG_PATH = "../data/extracted_frames/may15/test1_trim/frame_000000.jpg"
VID_PATH = REPO_ROOT / "src/capstone_robot/data/videos/may24/pole_bell_11am/alignright.mp4"
IMG_FOLDER = REPO_ROOT / "src/capstone_robot/data/extracted_frames/may15/test1_trim"

def draw_line(img, line, color=(0, 255, 0), thickness=2):
    # draw pole centerline
    out = img.copy()

    vx, vy, x0, y0 = line
    t = 1000

    x1 = int(x0 - vx * t)
    y1 = int(y0 - vy * t)
    x2 = int(x0 + vx * t)
    y2 = int(y0 + vy * t)

    cv2.line(out, (x1, y1), (x2, y2), color, thickness)
    return out

tracker = PoleBellTracker(color_format="bgr")

# VideoCapture uses BGR
cap = cv2.VideoCapture(str(VID_PATH))
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VID_PATH}")

idx = 0
while True:
    ret, img = cap.read()
    if not ret:
        break
    idx += 1
    print(f"frame {idx}")
    img = rotate_frame(img, "180")
    alignment = tracker.detect(img)
    vis = img.copy()
    if alignment is not None:
        print(alignment.pole_line)
        vis = draw_line(vis, alignment.pole_line)
        vis = cv2.circle(vis, (alignment.bell[0], alignment.bell[1]), alignment.bell[2], (255, 0, 0))
    cv2.imshow("vis", vis)
    # sleep(0.05)
    cv2.waitKey(0)
