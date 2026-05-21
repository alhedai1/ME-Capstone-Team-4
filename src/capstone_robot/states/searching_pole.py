import time

import cv2


def update_preview(robot, frame, pole, status):
    vis = frame.copy()
    if pole is not None:
        x, y, w, h = pole.box
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(vis, (int(x + w / 2), int(y + h / 2)), 4, (0, 255, 0), -1)

    cv2.putText(vis, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    robot.update_preview(vis)


def run(robot):
    stable_frames = 0

    while robot.state == "searching_pole":
        frame, pole = robot.detect_pole()
        if frame is None:
            print("[WARN] No AI camera frame/metadata received")
            robot.motors.stop()
            time.sleep(0.1)
            continue

        if pole is None:
            stable_frames = 0
            print("[SEARCH] Pole not detected; rotating slowly")
            update_preview(robot, frame, None, "SEARCH: NO POLE")
            robot.motors.right(robot.search_turn_speed)
            time.sleep(0.05)
            continue

        x, y, w, h = pole.box
        pole_center_x = x + w / 2.0
        frame_center_x = frame.shape[1] / 2.0
        error_x = pole_center_x - frame_center_x

        if abs(error_x) <= robot.pole_center_deadband_px:
            stable_frames += 1
            robot.motors.stop()
            print(
                f"[SEARCH] Pole centered ({stable_frames}/{robot.pole_stable_frames_required}), "
                f"error_x={error_x:.1f}px, conf={pole.confidence:.2f}"
            )
            update_preview(robot, frame, pole, f"SEARCH: CENTERED {stable_frames}/{robot.pole_stable_frames_required}")

            if stable_frames >= robot.pole_stable_frames_required:
                robot.pole_found()
                return
        else:
            stable_frames = 0
            if error_x < 0:
                print(f"[SEARCH] Pole left of center, error_x={error_x:.1f}px")
                update_preview(robot, frame, pole, f"SEARCH: LEFT error={error_x:.1f}")
                robot.motors.left(robot.center_turn_speed)
            else:
                print(f"[SEARCH] Pole right of center, error_x={error_x:.1f}px")
                update_preview(robot, frame, pole, f"SEARCH: RIGHT error={error_x:.1f}")
                robot.motors.right(robot.center_turn_speed)

        time.sleep(0.05)
