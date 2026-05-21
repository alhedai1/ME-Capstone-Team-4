import time


def run(robot):
    close_frames = 0
    missed_frames = 0

    while robot.state == "approaching_pole":
        frame, pole = robot.detect_pole()
        if frame is None:
            print("[WARN] No AI camera frame/metadata received")
            robot.motors.stop()
            time.sleep(0.1)
            continue

        if pole is None:
            missed_frames += 1
            close_frames = 0
            print(f"[APPROACH] Pole lost ({missed_frames}/{robot.approach_missed_frame_limit})")

            if missed_frames >= robot.approach_missed_frame_limit:
                robot.motors.right(robot.search_turn_speed)
            else:
                robot.motors.stop()

            time.sleep(0.05)
            continue

        missed_frames = 0
        x, y, w, h = pole.box
        frame_width = frame.shape[1]
        pole_center_x = x + w / 2.0
        frame_center_x = frame_width / 2.0
        error_x = pole_center_x - frame_center_x
        width_fraction = w / frame_width

        if width_fraction >= robot.approach_stop_width_fraction:
            close_frames += 1
            robot.motors.stop()
            print(
                f"[APPROACH] Close to pole ({close_frames}/{robot.approach_stop_frames_required}), "
                f"width={width_fraction:.2f}, error_x={error_x:.1f}px"
            )

            if close_frames >= robot.approach_stop_frames_required:
                robot.pole_reached()
                return

            time.sleep(0.05)
            continue

        close_frames = 0
        normalized_error = error_x / frame_center_x
        steering = max(-0.5, min(0.5, normalized_error * robot.approach_steer_gain))
        left_speed = robot.approach_speed + steering
        right_speed = robot.approach_speed - steering

        robot.drive_tank(left_speed, right_speed)
        print(
            f"[APPROACH] width={width_fraction:.2f}, error_x={error_x:.1f}px, "
            f"left={left_speed:.2f}, right={right_speed:.2f}, conf={pole.confidence:.2f}"
        )
        time.sleep(0.05)
