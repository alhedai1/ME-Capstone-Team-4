import time

import cv2

from capstone_robot.utils import rotate_frame
from capstone_robot.vision.pole_bell2 import PoleBellTracker

try:
    from libcamera import controls
except ImportError:
    controls = None


if controls is not None:
    aligning_controls = {
        "AfMode": controls.AfModeEnum.Manual,
        "LensPosition": 0.0,
        "AwbMode": controls.AwbModeEnum.Daylight,
        "ExposureValue": -1.5,
    }
    print("GOT ALIGNING CONTROLS")
else:
    aligning_controls = {}


def setting(robot, name, default):
    return getattr(robot, name, default)


def draw_line(img, line, color=(0, 255, 0), thickness=2):
    out = img.copy()
    vx, vy, x0, y0 = line
    t = 1000
    x1 = int(x0 - vx * t)
    y1 = int(y0 - vy * t)
    x2 = int(x0 + vx * t)
    y2 = int(y0 + vy * t)
    cv2.line(out, (x1, y1), (x2, y2), color, thickness)
    return out


def update_alignment_preview(robot, frame, alignment, status):
    if frame is None:
        return

    vis = frame.copy()
    if alignment is not None:
        vis = draw_line(vis, alignment.pole_line, (0, 255, 0), 2)
        bx, by, br = alignment.bell
        cv2.circle(vis, (bx, by), br, (0, 0, 255), 2)
        cv2.circle(vis, (bx, by), 3, (0, 0, 255), -1)

    cv2.putText(vis, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    robot.update_preview(vis)


def update_front_preview(robot, frame, pole, status):
    if frame is None:
        return

    vis = frame.copy()
    if pole is not None:
        x, y, w, h = pole.box
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(vis, (int(x + w / 2), int(y + h / 2)), 4, (0, 255, 0), -1)

    cv2.putText(vis, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    robot.update_preview(vis)


def smooth_box(old_box, new_box, alpha):
    if old_box is None:
        return new_box

    return tuple(int(alpha * new + (1.0 - alpha) * old) for old, new in zip(old_box, new_box))


def get_pole_bell_tracker(robot):
    tracker = getattr(robot, "pole_bell_tracker", None)
    if tracker is None or tracker.__class__.__module__ != "capstone_robot.vision.pole_bell2":
        tracker = PoleBellTracker(color_format="rgb")
        robot.pole_bell_tracker = tracker
    return tracker


def read_upward_alignment(robot, rotation="180"):
    ok, frame = robot.pi_camera.read()
    if not ok or frame is None:
        return None, None

    frame = rotate_frame(frame, rotation)
    return frame, get_pole_bell_tracker(robot).detect(frame)


def center_front_pole(robot, label="CENTER", search_direction="right"):
    stable_frames = 0
    missed_frames = 0
    last_pole = None
    last_motor_action = None
    smoothed_box = None

    while robot.state == "aligning_bell":
        frame, pole = robot.detect_pole()
        if frame is None:
            print("[ALIGN-CIRCLE] No AI camera frame/metadata received")
            robot.motors.stop()
            time.sleep(0.1)
            continue

        if pole is None:
            missed_frames += 1

            if last_pole is not None and missed_frames <= setting(robot, "search_missed_frame_limit", 6):
                print(
                    f"[ALIGN-CIRCLE] Front pole briefly lost "
                    f"({missed_frames}/{setting(robot, 'search_missed_frame_limit', 6)}); "
                    "holding last centering action"
                )
                update_front_preview(robot, frame, last_pole, f"{label}: HOLD {missed_frames}")

                if last_motor_action == "left":
                    robot.motors.left(setting(robot, "center_turn_speed", 0.3))
                elif last_motor_action == "right":
                    robot.motors.right(setting(robot, "center_turn_speed", 0.3))
                else:
                    robot.motors.stop()

                time.sleep(0.05)
                continue

            stable_frames = 0
            last_pole = None
            smoothed_box = None
            print(
                f"[ALIGN-CIRCLE] Front pole not detected; rotating {search_direction} slowly "
                f"({missed_frames})"
            )
            update_front_preview(robot, frame, None, f"{label}: NO POLE {missed_frames}")
            if search_direction == "left":
                robot.motors.left(setting(robot, "search_turn_speed", 0.3))
            else:
                robot.motors.right(setting(robot, "search_turn_speed", 0.3))
            time.sleep(0.05)
            continue

        missed_frames = 0
        smoothed_box = smooth_box(smoothed_box, pole.box, setting(robot, "pole_smooth_alpha", 1.0))
        pole.box = smoothed_box
        last_pole = pole

        x, y, w, h = pole.box
        error_x = (x + w / 2.0) - (frame.shape[1] / 2.0)

        if abs(error_x) <= setting(robot, "pole_center_deadband_px", 20):
            stable_frames += 1
            last_motor_action = "stop"
            robot.motors.stop()
            print(
                f"[ALIGN-CIRCLE] Front pole centered "
                f"({stable_frames}/{setting(robot, 'pole_stable_frames_required', 5)}), "
                f"error_x={error_x:.1f}px"
            )
            update_front_preview(robot, frame, pole, f"{label}: CENTERED {stable_frames}")

            if stable_frames >= setting(robot, "pole_stable_frames_required", 5):
                return True
        else:
            stable_frames = 0
            if error_x < 0:
                print(f"[ALIGN-CIRCLE] Front pole left of center, error_x={error_x:.1f}px")
                update_front_preview(robot, frame, pole, f"{label}: LEFT {error_x:.1f}")
                last_motor_action = "left"
                robot.motors.left(setting(robot, "center_turn_speed", 0.3))
            else:
                print(f"[ALIGN-CIRCLE] Front pole right of center, error_x={error_x:.1f}px")
                update_front_preview(robot, frame, pole, f"{label}: RIGHT {error_x:.1f}")
                last_motor_action = "right"
                robot.motors.right(setting(robot, "center_turn_speed", 0.3))

        time.sleep(0.05)

    return False


def approach_front_pole(robot):
    close_frames = 0
    missed_frames = 0
    smoothed_box = None
    last_left_speed = 0.0
    last_right_speed = 0.0

    while robot.state == "aligning_bell":
        frame, pole = robot.detect_pole()
        if frame is None:
            print("[ALIGN-CIRCLE] No AI camera frame/metadata received")
            robot.motors.stop()
            time.sleep(0.1)
            continue

        if pole is None:
            missed_frames += 1
            close_frames = 0
            smoothed_box = None
            print(f"[ALIGN-CIRCLE] Pole lost while re-approaching ({missed_frames})")
            update_front_preview(robot, frame, None, f"REAPPROACH: LOST {missed_frames}")

            if missed_frames <= setting(robot, "approach_hold_frame_limit", 3):
                robot.drive(last_left_speed, last_right_speed)
            else:
                robot.motors.right(setting(robot, "search_turn_speed", 0.3))

            time.sleep(0.05)
            continue

        missed_frames = 0
        smoothed_box = smooth_box(smoothed_box, pole.box, setting(robot, "pole_smooth_alpha", 1.0))
        pole.box = smoothed_box

        x, y, w, h = pole.box
        frame_width = frame.shape[1]
        error_x = (x + w / 2.0) - (frame_width / 2.0)
        width_fraction = w / frame_width

        if width_fraction >= setting(robot, "approach_stop_width_fraction", 0.16):
            close_frames += 1
            robot.motors.stop()
            print(
                f"[ALIGN-CIRCLE] Reached pole distance "
                f"({close_frames}/{setting(robot, 'approach_stop_frames_required', 3)}), "
                f"width={width_fraction:.2f}, error_x={error_x:.1f}px"
            )
            update_front_preview(robot, frame, pole, f"REAPPROACH: CLOSE {width_fraction:.2f}")

            if close_frames >= setting(robot, "approach_stop_frames_required", 3):
                return True

            time.sleep(0.05)
            continue

        close_frames = 0
        normalized_error = error_x / (frame_width / 2.0)
        steering = max(-0.5, min(0.5, normalized_error * setting(robot, "approach_steer_gain", 0.5)))
        speed = setting(robot, "approach_speed", 0.4)
        left_speed = speed + steering
        right_speed = speed - steering
        last_left_speed = left_speed
        last_right_speed = right_speed

        robot.drive(left_speed, right_speed)
        print(
            f"[ALIGN-CIRCLE] Re-approaching pole, width={width_fraction:.2f}, "
            f"error_x={error_x:.1f}px, left={left_speed:.2f}, right={right_speed:.2f}"
        )
        update_front_preview(robot, frame, pole, f"REAPPROACH: width={width_fraction:.2f}")
        time.sleep(0.05)

    return False


def wait_for_pole_bell_alignment(robot):
    stable_frames = 0
    missed_frames = 0

    while robot.state == "aligning_bell":
        frame, alignment = read_upward_alignment(robot)
        if frame is None:
            robot.motors.stop()
            print("[ALIGN-CIRCLE] No upward camera frame received")
            time.sleep(0.1)
            continue

        if alignment is None:
            missed_frames += 1
            stable_frames = 0
            robot.motors.stop()
            print(
                f"[ALIGN-CIRCLE] Need pole+bell alignment "
                f"({missed_frames}/{setting(robot, 'alignment_missed_frame_limit', 15)})"
            )
            update_alignment_preview(robot, frame, None, f"ALIGN: LOST {missed_frames}")
            time.sleep(0.05)
            continue

        missed_frames = 0
        error = alignment.error_px
        threshold = setting(robot, "pole_bell_error_threshold_px", setting(robot, "alignment_error_threshold_px", 20))

        if abs(error) <= threshold:
            stable_frames += 1
            robot.motors.stop()
            print(
                f"[ALIGN-CIRCLE] Pole and bell aligned "
                f"({stable_frames}/{setting(robot, 'alignment_stable_frames_required', 4)}), "
                f"error={error:.1f}px"
            )
            update_alignment_preview(robot, frame, alignment, f"ALIGN: OK {stable_frames} err={error:.1f}")

            if stable_frames >= setting(robot, "alignment_stable_frames_required", 4):
                return "aligned", 0.0
        else:
            side = alignment.side
            print(f"[ALIGN-CIRCLE] Pole/bell error={error:.1f}px, bell side={side}")
            update_alignment_preview(robot, frame, alignment, f"ALIGN: {side} err={error:.1f}")
            return side, error

        time.sleep(0.05)

    return None, None


def wait_for_bell_side(robot):
    side, _ = wait_for_pole_bell_alignment(robot)
    return side


def orbit_until_bell_aligned(robot, rotation="180"):
    stable_frames = 0
    missed_frames = 0

    while robot.state == "aligning_bell":
        frame, alignment = read_upward_alignment(robot, rotation=rotation)
        if alignment is None:
            missed_frames += 1
            stable_frames = 0
            robot.motors.stop()
            print(
                f"[ALIGN-CIRCLE] Lost pole/bell while orbiting "
                f"({missed_frames}/{setting(robot, 'alignment_missed_frame_limit', 15)})"
            )
            update_alignment_preview(robot, frame, None, f"ORBIT: LOST {missed_frames}")
            time.sleep(0.05)
            continue

        missed_frames = 0
        error = alignment.error_px
        threshold = setting(robot, "pole_bell_error_threshold_px", setting(robot, "alignment_error_threshold_px", 20))

        if abs(error) <= threshold:
            stable_frames += 1
            robot.motors.stop()
            update_alignment_preview(robot, frame, alignment, f"ORBIT: ALIGNED {stable_frames}")
            if stable_frames >= setting(robot, "alignment_stable_frames_required", 4):
                return True
        else:
            stable_frames = 0
            side = alignment.side
            duration = orbit_seconds_from_error(robot, error)
            print(f"[ALIGN-CIRCLE] Orbit correction side={side}, error={error:.1f}px, duration={duration:.2f}s")
            drive_forward_with_bias(robot, robot.opposite_direction(side), duration)
            get_pole_bell_tracker(robot).reset()

        time.sleep(0.05)

    return False


def center_front_pole_for_climb(robot):
    return center_front_pole(robot, label="FINAL CENTER")


def drive_forward_with_bias(robot, bias_side, seconds):
    speed = setting(robot, "pole_bell_orbit_forward_speed", 0.35)
    bias = setting(robot, "pole_bell_orbit_turn_bias", 0.12)

    if bias_side == "right":
        left_speed = speed + bias
        right_speed = speed - bias
    else:
        left_speed = speed - bias
        right_speed = speed + bias

    robot.drive(left_speed, right_speed)
    time.sleep(seconds)
    robot.motors.stop()
    time.sleep(0.15)


def orbit_seconds_from_error(robot, error_px):
    min_seconds = setting(robot, "pole_bell_orbit_min_seconds", 0.20)
    max_seconds = setting(robot, "pole_bell_orbit_max_seconds", 1.20)
    px_per_second = setting(robot, "pole_bell_orbit_px_per_second", 80.0)
    seconds = abs(error_px) / max(1.0, px_per_second)
    return max(min_seconds, min(max_seconds, seconds))


def orbit_step(robot, bell_side, error_px):
    reverse_speed = setting(robot, "pole_bell_reverse_speed", 0.25)
    reverse_seconds = setting(robot, "pole_bell_reverse_seconds", 0.25)
    # turn_seconds = orbit_seconds_from_error(robot, error_px) * setting(robot, "pole_bell_turn_time_scale", 0.75)
    turn_seconds = setting(robot, "pole_bell_turn_seconds", 1.5)
    forward_seconds = orbit_seconds_from_error(robot, error_px)
    settle_seconds = setting(robot, "pole_bell_settle_seconds", 0.15)
    turn_speed = setting(robot, "align_turn_speed", 0.3)

    print(
        f"[ALIGN-CIRCLE] Orbit step side={bell_side}, error={error_px:.1f}px, "
        f"turn={turn_seconds:.2f}s, forward={forward_seconds:.2f}s"
    )
    robot.motors.backward(reverse_speed)
    time.sleep(reverse_seconds)
    robot.motors.stop()
    time.sleep(settle_seconds)

    robot.turn_in_place(bell_side, turn_seconds, speed=turn_speed)

    bias_side = robot.opposite_direction(bell_side)
    drive_forward_with_bias(robot, bias_side, forward_seconds)


def run(robot):
    if aligning_controls:
        robot.pi_camera.picam2.set_controls(aligning_controls)

    max_steps = setting(robot, "pole_bell_max_orbit_steps", 20)
    get_pole_bell_tracker(robot).reset()

    if not center_front_pole(robot, label="PREALIGN"):
        return

    for step in range(1, max_steps + 1):
        side, error = wait_for_pole_bell_alignment(robot)
        if side is None:
            return

        if side == "aligned":
            if center_front_pole(robot, label="FINAL CENTER") and approach_front_pole(robot):
                robot.aligned()
            return

        print(f"[ALIGN-CIRCLE] Orbit iteration {step}/{max_steps}, side={side}, error={error:.1f}px")
        orbit_step(robot, side, error)

        search_direction = robot.opposite_direction(side)
        get_pole_bell_tracker(robot).reset()
        if not center_front_pole(robot, label="RECENTER", search_direction=search_direction):
            return

        if not approach_front_pole(robot):
            return

        get_pole_bell_tracker(robot).reset()

    robot.motors.stop()
    print(f"[ALIGN-CIRCLE] Pole and bell did not align after {max_steps} orbit steps")
