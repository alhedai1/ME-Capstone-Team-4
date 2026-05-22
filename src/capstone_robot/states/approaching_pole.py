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


def smooth_box(old_box, new_box, alpha):
    if old_box is None:
        return new_box

    return tuple(
        int(alpha * new + (1.0 - alpha) * old)
        for old, new in zip(old_box, new_box)
    )


# def drive_robot(robot, left_speed, right_speed):
#     if hasattr(robot, "drive_tank"):
#         robot.drive_tank(left_speed, right_speed)
#     else:
#         robot.drive(left_speed, right_speed)


def run(robot):
    close_frames = 0
    missed_frames = 0
    last_pole = None
    smoothed_box = None
    last_left_speed = 0.0
    last_right_speed = 0.0

    while robot.state == "approaching_pole":
        frame, pole = robot.detect_pole()
        if frame is None:
            print("[WARN] No AI camera frame/metadata received")
            robot.motors.stop()
            time.sleep(0.1)
            continue

        if pole is None:
            missed_frames += 1

            if last_pole is not None and missed_frames <= robot.approach_hold_frame_limit:
                print(
                    f"[APPROACH] Pole briefly lost ({missed_frames}/{robot.approach_hold_frame_limit}); "
                    "holding last drive command"
                )
                update_preview(robot, frame, last_pole, f"APPROACH: HOLD {missed_frames}")
                robot.drive(last_left_speed, last_right_speed)
                time.sleep(0.05)
                continue

            close_frames = 0
            print(f"[APPROACH] Pole lost ({missed_frames}/{robot.approach_missed_frame_limit})")
            update_preview(robot, frame, None, f"APPROACH: LOST {missed_frames}")

            if missed_frames >= robot.approach_missed_frame_limit:
                robot.motors.right(robot.search_turn_speed)
            else:
                robot.motors.stop()

            if missed_frames >= robot.approach_missed_frame_limit:
                last_pole = None
                smoothed_box = None
                last_left_speed = 0.0
                last_right_speed = 0.0

            time.sleep(0.05)
            continue

        missed_frames = 0
        smoothed_box = smooth_box(smoothed_box, pole.box, robot.pole_smooth_alpha)
        pole.box = smoothed_box
        last_pole = pole

        x, y, w, h = pole.box
        frame_width = frame.shape[1]
        pole_center_x = x + w / 2.0
        frame_center_x = frame_width / 2.0
        error_x = pole_center_x - frame_center_x
        width_fraction = w / frame_width

        if width_fraction >= robot.approach_stop_width_fraction:
            close_frames += 1
            last_left_speed = 0.0
            last_right_speed = 0.0
            robot.motors.stop()
            print(
                f"[APPROACH] Close to pole ({close_frames}/{robot.approach_stop_frames_required}), "
                f"width={width_fraction:.2f}, error_x={error_x:.1f}px"
            )
            update_preview(robot, frame, pole, f"APPROACH: CLOSE width={width_fraction:.2f}")

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

        last_left_speed = left_speed
        last_right_speed = right_speed
        robot.drive(left_speed, right_speed)
        print(
            f"[APPROACH] width={width_fraction:.2f}, error_x={error_x:.1f}px, "
            f"left={left_speed:.2f}, right={right_speed:.2f}, conf={pole.confidence:.2f}"
        )
        update_preview(robot, frame, pole, f"APPROACH: width={width_fraction:.2f} error={error_x:.1f}")
        time.sleep(0.05)
