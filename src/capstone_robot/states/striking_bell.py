import time

import cv2

striking_controls = {
    "LensPosition": 12.0,           # Instantly force lens to maximum physical close-up limit
    "ExposureValue": 0.0
}

def wait_for_bell(robot, required_frames=3):
    seen_frames = 0

    while robot.state == "striking_bell":
        ok, frame = robot.pi_camera.read()
        if not ok or frame is None:
            print("[STRIKE] No camera frame")
            time.sleep(0.05)
            continue

        bell = robot.bell_tracker.detect(frame)
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
        x, y, w, h = bell.box
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            vis,
            f"{bell.area}px",
            (x, max(25, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

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
    robot.bell_tracker.reset()
    robot.pi_camera.picam2.set_controls(striking_controls)

    if wait_for_bell(robot):
        strike_once(robot)

    time.sleep(3.0)
    robot.bell_tracker.reset()

    if wait_for_bell(robot):
        strike_once(robot)

    robot.servo.detach()

    robot.mission_complete()
