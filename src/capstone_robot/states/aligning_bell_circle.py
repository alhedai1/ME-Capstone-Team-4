import time

import cv2

from capstone_robot.utils import rotate_frame
from capstone_robot.vision.bell_circle import BellCircle

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
else:
    aligning_controls = {}


def setting(robot, name, default):
    return getattr(robot, name, default)


def update_bell_preview(robot, frame, bell, status):
    if frame is None:
        return

    vis = frame.copy()
    height, width = vis.shape[:2]
    cv2.line(vis, (width // 2, 0), (width // 2, height), (255, 0, 0), 1)

    if bell is not None:
        cv2.circle(vis, (bell.x, bell.y), bell.radius, (0, 0, 255), 2)
        cv2.circle(vis, (bell.x, bell.y), 3, (0, 0, 255), -1)

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


def read_upward_bell(robot, rotation="180"):
    ok, frame = robot.pi_camera.read()
    if not ok or frame is None:
        return None, None

    frame = rotate_frame(frame, rotation)
    detector = getattr(robot, "bell_circle", None)
    if detector is None:
        detector = BellCircle(color_format="rgb")
        robot.bell_circle = detector

    return frame, detector.detect(frame)


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


def wait_for_bell_position(robot):
    stable_frames = 0
    missed_frames = 0

    while robot.state == "aligning_bell":
        frame, bell = read_upward_bell(robot)
        if frame is None:
            robot.motors.stop()
            print("[ALIGN-CIRCLE] No upward camera frame received")
            time.sleep(0.1)
            continue

        if bell is None:
            missed_frames += 1
            stable_frames = 0
            robot.motors.stop()
            print(f"[ALIGN-CIRCLE] Need bell circle ({missed_frames}/{setting(robot, 'alignment_missed_frame_limit', 15)})")
            update_bell_preview(robot, frame, None, f"BELL: LOST {missed_frames}")
            time.sleep(0.05)
            continue

        missed_frames = 0
        error_x = bell.x - frame.shape[1] / 2.0
        threshold = setting(robot, "bell_circle_error_threshold_px", setting(robot, "alignment_error_threshold_px", 20))

        if abs(error_x) <= threshold:
            stable_frames += 1
            robot.motors.stop()
            print(
                f"[ALIGN-CIRCLE] Bell centered "
                f"({stable_frames}/{setting(robot, 'alignment_stable_frames_required', 4)}), "
                f"error_x={error_x:.1f}px"
            )
            update_bell_preview(robot, frame, bell, f"BELL: CENTERED {stable_frames}")

            if stable_frames >= setting(robot, "alignment_stable_frames_required", 4):
                return "aligned"
        else:
            side = "left" if error_x < 0 else "right"
            print(f"[ALIGN-CIRCLE] Bell is {side}, error_x={error_x:.1f}px")
            update_bell_preview(robot, frame, bell, f"BELL: {side} {error_x:.1f}")
            return side

        time.sleep(0.05)

    return None


def drive_forward_with_bias(robot, bias_side, seconds):
    speed = setting(robot, "bell_circle_orbit_forward_speed", 0.35)
    bias = setting(robot, "bell_circle_orbit_turn_bias", 0.12)

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


def orbit_step(robot, bell_side):
    reverse_speed = setting(robot, "bell_circle_reverse_speed", 0.25)
    reverse_seconds = setting(robot, "bell_circle_reverse_seconds", 0.25)
    turn_seconds = setting(robot, "bell_circle_turn_seconds", 0.35)
    forward_seconds = setting(robot, "bell_circle_orbit_forward_seconds", 0.45)
    settle_seconds = setting(robot, "bell_circle_settle_seconds", 0.15)
    turn_speed = setting(robot, "align_turn_speed", 0.3)

    print(f"[ALIGN-CIRCLE] Incremental orbit step for bell on {bell_side}")
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

    max_steps = setting(robot, "bell_circle_max_orbit_steps", 20)

    if not center_front_pole(robot, label="PREALIGN"):
        return

    for step in range(1, max_steps + 1):
        side = wait_for_bell_position(robot)
        if side is None:
            return

        if side == "aligned":
            if center_front_pole(robot, label="FINAL CENTER") and approach_front_pole(robot):
                robot.aligned()
            return

        print(f"[ALIGN-CIRCLE] Orbit iteration {step}/{max_steps}, bell_side={side}")
        orbit_step(robot, side)

        search_direction = robot.opposite_direction(side)
        if not center_front_pole(robot, label="RECENTER", search_direction=search_direction):
            return

        if not approach_front_pole(robot):
            return

    robot.motors.stop()
    print(f"[ALIGN-CIRCLE] Bell did not center after {max_steps} orbit steps")
