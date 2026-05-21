import time


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

            if stable_frames >= robot.pole_stable_frames_required:
                robot.pole_found()
                return
        else:
            stable_frames = 0
            if error_x < 0:
                print(f"[SEARCH] Pole left of center, error_x={error_x:.1f}px")
                robot.motors.left(robot.center_turn_speed)
            else:
                print(f"[SEARCH] Pole right of center, error_x={error_x:.1f}px")
                robot.motors.right(robot.center_turn_speed)

        time.sleep(0.05)
