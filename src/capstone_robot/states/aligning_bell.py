import time

import cv2


def read_upward_alignment(robot):
    ok, frame = robot.pi_camera.read()
    if not ok or frame is None:
        return None

    frame = cv2.rotate(frame, cv2.ROTATE_180)
    return robot.pole_bell_tracker.detect(frame)


def wait_for_bell_side(robot):
    missed_frames = 0
    aligned_frames = 0

    while robot.state == "aligning_bell":
        alignment = read_upward_alignment(robot)
        if alignment is None:
            missed_frames += 1
            aligned_frames = 0
            robot.motors.stop()
            print(f"[ALIGN] Need pole and bell ({missed_frames}/{robot.alignment_missed_frame_limit})")
            time.sleep(0.05)
            continue

        if abs(alignment.error_px) <= robot.alignment_error_threshold_px:
            aligned_frames += 1
            robot.motors.stop()
            print(
                f"[ALIGN] Already aligned ({aligned_frames}/{robot.alignment_stable_frames_required}), "
                f"error={alignment.error_px:.1f}px"
            )

            if aligned_frames >= robot.alignment_stable_frames_required:
                return "aligned"

            time.sleep(0.05)
            continue

        print(f"[ALIGN] Bell is on the {alignment.side}, error={alignment.error_px:.1f}px")
        return alignment.side

    return None


def orbit_until_bell_aligned(robot):
    stable_frames = 0
    missed_frames = 0

    while robot.state == "aligning_bell":
        alignment = read_upward_alignment(robot)
        if alignment is None:
            missed_frames += 1
            stable_frames = 0
            robot.motors.stop()
            print(f"[ALIGN] Lost pole/bell while orbiting ({missed_frames}/{robot.alignment_missed_frame_limit})")
            time.sleep(0.05)
            continue

        missed_frames = 0
        error = alignment.error_px

        if abs(error) <= robot.alignment_error_threshold_px:
            stable_frames += 1
            robot.motors.stop()
            print(
                f"[ALIGN] Bell aligned ({stable_frames}/{robot.alignment_stable_frames_required}), "
                f"error={error:.1f}px"
            )

            if stable_frames >= robot.alignment_stable_frames_required:
                return True
        else:
            stable_frames = 0
            robot.motors.forward(robot.orbit_speed)
            print(f"[ALIGN] Orbiting, side={alignment.side}, error={error:.1f}px")

        time.sleep(0.05)

    return False


def center_front_pole_for_climb(robot):
    stable_frames = 0

    while robot.state == "aligning_bell":
        frame, pole = robot.detect_pole()
        if frame is None:
            print("[WARN] No AI camera frame/metadata received")
            robot.motors.stop()
            time.sleep(0.1)
            continue

        if pole is None:
            stable_frames = 0
            print("[ALIGN] Front pole not detected; rotating right slowly")
            robot.motors.right(robot.search_turn_speed)
            time.sleep(0.05)
            continue

        x, y, w, h = pole.box
        error_x = (x + w / 2.0) - (frame.shape[1] / 2.0)

        if abs(error_x) <= robot.pole_center_deadband_px:
            stable_frames += 1
            robot.motors.stop()
            print(
                f"[ALIGN] Front pole centered ({stable_frames}/{robot.pole_stable_frames_required}), "
                f"error_x={error_x:.1f}px"
            )

            if stable_frames >= robot.pole_stable_frames_required:
                return True
        else:
            stable_frames = 0
            if error_x < 0:
                robot.motors.left(robot.center_turn_speed)
                print(f"[ALIGN] Front pole left of center, error_x={error_x:.1f}px")
            else:
                robot.motors.right(robot.center_turn_speed)
                print(f"[ALIGN] Front pole right of center, error_x={error_x:.1f}px")

        time.sleep(0.05)

    return False


def run(robot):
    robot.pole_bell_tracker.reset()
    side = wait_for_bell_side(robot)
    if side is None:
        return

    if side == "aligned":
        if center_front_pole_for_climb(robot):
            robot.aligned()
        return

    print(f"[ALIGN] Turning {side} about 90 degrees")
    robot.turn_in_place(side, robot.align_quarter_turn_seconds)
    robot.pole_bell_tracker.reset()

    if not orbit_until_bell_aligned(robot):
        return

    face_pole_direction = robot.opposite_direction(side)
    print(f"[ALIGN] Turning {face_pole_direction} back toward pole")
    robot.turn_in_place(face_pole_direction, robot.align_quarter_turn_seconds)
    robot.pole_bell_tracker.reset()

    if center_front_pole_for_climb(robot):
        robot.aligned()
