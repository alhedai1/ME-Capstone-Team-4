import time

import cv2
import numpy as np


LOWER_GOLD = np.array([15, 30, 30])
UPPER_GOLD = np.array([30, 255, 255])


def detect_bell(frame, min_area=2000, center_tol=0.20, min_width=0.20, min_height=0.20):
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, LOWER_GOLD, UPPER_GOLD)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < min_area:
        return None

    img_h, img_w = frame.shape[:2]
    x, y, w, h = cv2.boundingRect(contour)
    bell_cx = x + w / 2.0
    bell_cy = y + h / 2.0

    if abs(bell_cx - img_w / 2.0) > center_tol * img_w:
        return None
    if abs(bell_cy - img_h / 2.0) > center_tol * img_h:
        return None
    if w < min_width * img_w or h < min_height * img_h:
        return None

    return x, y, w, h, area


def wait_for_bell(robot, required_frames=3):
    seen_frames = 0

    while robot.state == "striking_bell":
        ok, frame = robot.pi_camera.read()
        if not ok or frame is None:
            print("[STRIKE] No camera frame")
            time.sleep(0.05)
            continue

        bell = detect_bell(frame)
        if bell is None:
            seen_frames = 0
            print("[STRIKE] Waiting for bell")
            update_preview(robot, frame, None, "STRIKE: WAITING FOR BELL")
        else:
            seen_frames += 1
            print(f"[STRIKE] Bell detected ({seen_frames}/{required_frames})")
            update_preview(robot, frame, bell, f"STRIKE: BELL {seen_frames}/{required_frames}")
            if seen_frames >= required_frames:
                return True

        time.sleep(0.05)

    return False


def update_preview(robot, frame, bell, status):
    vis = frame.copy()
    if bell is not None:
        x, y, w, h, area = bell
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(vis, f"{int(area)}px", (x, max(25, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.putText(vis, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    robot.update_preview(vis)


def strike_once(robot):
    robot.servo.max()
    time.sleep(1.2)
    robot.servo.min()
    # time.sleep(0.5)


def run(robot):
    print("[STATE] Bell within reach! Striking...")
    robot.motors.stop()

    if wait_for_bell(robot):
        strike_once(robot)

    time.sleep(3.0)

    if wait_for_bell(robot):
        strike_once(robot)

    robot.servo.detach()

    robot.mission_complete()
