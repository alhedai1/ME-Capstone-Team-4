import time

import cv2

from capstone_robot.utils import FixedRateLoop


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


def opposite_direction(direction):
    return "left" if direction == "right" else "right"


def alternating_search_direction(missed_frames, initial_direction, sweep_frames):
    direction = initial_direction if initial_direction in ("left", "right") else "right"
    sweep_frames = max(1, int(sweep_frames))
    remaining = max(0, missed_frames - 1)
    segment = 0

    while True:
        segment_length = (segment + 1) * sweep_frames
        if remaining < segment_length:
            break
        remaining -= segment_length
        segment += 1

    if segment % 2 == 1:
        direction = opposite_direction(direction)

    return direction, segment + 1


def run(robot):
    loop = FixedRateLoop(period_seconds=getattr(robot, "control_loop_period_seconds", 0.05))
    stable_frames = 0
    missed_frames = 0
    last_pole = None
    smoothed_box = None
    last_motor_action = None
    last_center_error_x = None
    last_center_time = None

    search_started_at = time.time()

    while robot.state == "searching_pole":
        frame, pole = robot.detect_pole()
        if frame is None:
            robot.log("[WARN] No AI camera frame/metadata received")
            robot.motors.stop()
            time.sleep(0.1)
            continue

        if pole is None:
            if time.time() - search_started_at < robot.search_startup_wait_seconds:
                robot.motors.stop()
                robot.log(f"[SEARCH] Waiting for initial pole detection - Time: {time.time() -search_started_at}")
                update_preview(robot, frame, None, "SEARCH: STARTUP WAIT")
                loop.sleep()
                continue

            missed_frames += 1

            if last_pole is not None and missed_frames <= robot.search_missed_frame_limit:
                robot.log(
                    f"[SEARCH] Pole briefly lost ({missed_frames}/{robot.search_missed_frame_limit}); "
                    "holding position"
                )
                update_preview(robot, frame, last_pole, f"SEARCH: HOLD {missed_frames}")
                # robot.motors.stop()
                if last_motor_action == "left":
                    robot.motors.left(robot.center_turn_speed)
                elif last_motor_action == "right":
                    robot.motors.right(robot.center_turn_speed)
                else:
                    robot.motors.stop()
                loop.sleep()
                continue

            stable_frames = 0
            last_pole = None
            smoothed_box = None
            last_center_error_x = None
            last_center_time = None
            search_direction, sweep = alternating_search_direction(
                missed_frames,
                getattr(robot, "pole_search_initial_direction", "right"),
                getattr(robot, "pole_search_sweep_frames", 12),
            )
            robot.log(
                f"[SEARCH] Pole not detected; sweep {sweep}, "
                f"rotating {search_direction} slowly"
            )
            update_preview(robot, frame, None, f"SEARCH: NO POLE {search_direction.upper()}")
            if search_direction == "left":
                robot.motors.left(robot.search_turn_speed)
            else:
                robot.motors.right(robot.search_turn_speed)
            loop.sleep()
            continue

        missed_frames = 0
        smoothed_box = smooth_box(smoothed_box, pole.box, robot.pole_smooth_alpha)
        pole.box = smoothed_box
        last_pole = pole

        x, y, w, h = pole.box
        pole_center_x = x + w / 2.0
        frame_center_x = frame.shape[1] / 2.0
        error_x = pole_center_x - frame_center_x

        if abs(error_x) <= robot.pole_center_deadband_px:
            stable_frames += 1
            last_motor_action = "stop"
            last_center_error_x = None
            last_center_time = None
            robot.motors.stop()
            robot.log(
                f"[SEARCH] Pole centered ({stable_frames}/{robot.pole_stable_frames_required}), "
                f"error_x={error_x:.1f}px, conf={pole.confidence:.2f}"
            )
            update_preview(robot, frame, pole, f"SEARCH: CENTERED {stable_frames}/{robot.pole_stable_frames_required}")

            if stable_frames >= robot.pole_stable_frames_required:
                robot.pole_found()
                return
        else:
            stable_frames = 0
            now = time.monotonic()
            dt = None if last_center_time is None else now - last_center_time
            turn_speed = robot.center_turn_speed_for_error(error_x, frame.shape[1], last_center_error_x, dt)
            last_center_error_x = error_x
            last_center_time = now

            if error_x < 0:
                robot.log(f"[SEARCH] Pole left of center, error_x={error_x:.1f}px, turn={turn_speed:.2f}")
                update_preview(robot, frame, pole, f"SEARCH: LEFT error={error_x:.1f}")
                last_motor_action = "left"
                robot.motors.left(turn_speed)
            else:
                robot.log(f"[SEARCH] Pole right of center, error_x={error_x:.1f}px, turn={turn_speed:.2f}")
                update_preview(robot, frame, pole, f"SEARCH: RIGHT error={error_x:.1f}")
                last_motor_action = "right"
                robot.motors.right(turn_speed)

        loop.sleep()
