import time

import cv2

from capstone_robot.utils import rotate_frame


def draw_line(frame, line, color=(0, 255, 0), thickness=2):
    vx, vy, x0, y0 = line
    t = 1000
    x1 = int(x0 - vx * t)
    y1 = int(y0 - vy * t)
    x2 = int(x0 + vx * t)
    y2 = int(y0 + vy * t)
    cv2.line(frame, (x1, y1), (x2, y2), color, thickness)


def update_alignment_preview(robot, frame, alignment, status):
    if frame is None:
        return

    vis = frame.copy()
    if alignment is not None:
        draw_line(vis, alignment.pole_line)
        bx, by, br = alignment.bell
        cv2.circle(vis, (bx, by), br, (0, 0, 255), 2)

    cv2.putText(vis, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    robot.update_preview(vis)


def update_front_preview(robot, frame, pole, status):
    vis = frame.copy()
    if pole is not None:
        x, y, w, h = pole.box
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(vis, (int(x + w / 2), int(y + h / 2)), 4, (0, 255, 0), -1)

    cv2.putText(vis, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    robot.update_preview(vis)


def read_upward_alignment(robot, rotation="180"):
    ok, frame = robot.pi_camera.read()
    if not ok or frame is None:
        return None, None

    frame = rotate_frame(frame, rotation)
    return frame, robot.pole_bell_tracker.detect(frame)


def wait_for_bell_side(robot):
    missed_frames = 0
    aligned_frames = 0

    while robot.state == "aligning_bell":
        frame, alignment = read_upward_alignment(robot)
        if alignment is None:
            missed_frames += 1
            aligned_frames = 0
            robot.motors.stop()
            print(f"[ALIGN] Need pole and bell ({missed_frames}/{robot.alignment_missed_frame_limit})")
            update_alignment_preview(robot, frame, None, f"ALIGN: NEED POLE/BELL {missed_frames}")
            time.sleep(0.05)
            continue

        if abs(alignment.error_px) <= robot.alignment_error_threshold_px:
            aligned_frames += 1
            robot.motors.stop()
            print(
                f"[ALIGN] Already aligned ({aligned_frames}/{robot.alignment_stable_frames_required}), "
                f"error={alignment.error_px:.1f}px"
            )
            update_alignment_preview(robot, frame, alignment, f"ALIGN: OK {aligned_frames}")

            if aligned_frames >= robot.alignment_stable_frames_required:
                return "aligned"

            time.sleep(0.05)
            continue

        print(f"[ALIGN] Bell is on the {alignment.side}, error={alignment.error_px:.1f}px")
        update_alignment_preview(robot, frame, alignment, f"ALIGN: {alignment.side} error={alignment.error_px:.1f}")
        return alignment.side

    return None


def orbit_until_bell_aligned(robot, rotation="180"):
    stable_frames = 0
    missed_frames = 0

    while robot.state == "aligning_bell":
        frame, alignment = read_upward_alignment(robot, rotation=rotation)
        if alignment is None:
            missed_frames += 1
            stable_frames = 0
            robot.motors.stop()
            print(f"[ALIGN] Lost pole/bell while orbiting ({missed_frames}/{robot.alignment_missed_frame_limit})")
            update_alignment_preview(robot, frame, None, f"ORBIT: LOST {missed_frames}")
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
            update_alignment_preview(robot, frame, alignment, f"ORBIT: ALIGNED {stable_frames}")

            if stable_frames >= robot.alignment_stable_frames_required:
                return True
        else:
            stable_frames = 0
            robot.motors.forward(robot.orbit_speed)
            print(f"[ALIGN] Orbiting, side={alignment.side}, error={error:.1f}px")
            update_alignment_preview(robot, frame, alignment, f"ORBIT: error={error:.1f}")

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
            update_front_preview(robot, frame, None, "FRONT: NO POLE")
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
            update_front_preview(robot, frame, pole, f"FRONT: CENTERED {stable_frames}")

            if stable_frames >= robot.pole_stable_frames_required:
                return True
        else:
            stable_frames = 0
            if error_x < 0:
                robot.motors.left(robot.center_turn_speed)
                print(f"[ALIGN] Front pole left of center, error_x={error_x:.1f}px")
                update_front_preview(robot, frame, pole, f"FRONT: LEFT {error_x:.1f}")
            else:
                robot.motors.right(robot.center_turn_speed)
                print(f"[ALIGN] Front pole right of center, error_x={error_x:.1f}px")
                update_front_preview(robot, frame, pole, f"FRONT: RIGHT {error_x:.1f}")

        time.sleep(0.05)

    return False


def orbit_rotation_for_turn(side):
    if side == "left":
        return "ccw"
    if side == "right":
        return "cw"
    return "180"


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

    orbit_rotation = orbit_rotation_for_turn(side)
    print(f"[ALIGN] Using {orbit_rotation} camera rotation while orbiting")
    if not orbit_until_bell_aligned(robot, rotation=orbit_rotation):
        return

    face_pole_direction = robot.opposite_direction(side)
    print(f"[ALIGN] Turning {face_pole_direction} back toward pole")
    robot.turn_in_place(face_pole_direction, robot.align_quarter_turn_seconds)
    robot.pole_bell_tracker.reset()

    if center_front_pole_for_climb(robot):
        robot.aligned()
