import time

import cv2

striking_controls = {
    "LensPosition": 12.0,           # Instantly force lens to maximum physical close-up limit
    "ExposureValue": 0.0
}

def update_front_preview(robot, frame, pole, status):
    vis = frame.copy()
    if pole is not None:
        x, y, w, h = pole.box
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(vis, (int(x + w / 2), int(y + h / 2)), 4, (0, 255, 0), -1)

    cv2.putText(vis, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    robot.update_preview(vis)


def update_bell_preview(robot, frame, bell, status):
    vis = frame.copy()
    if bell is not None:
        x, y, w, h = bell.box
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(vis, f"{bell.area}px", (x, max(25, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.putText(vis, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    robot.update_preview(vis)


def center_front_pole(robot):
    stable_frames = 0
    last_pole = None
    missed_frames = 0
    last_motor_action = "stop"
    started_at = time.time()

    while robot.state == "climbing_pole":
        if time.time() - started_at >= robot.climb_center_timeout_seconds:
            robot.motors.stop()
            print("[CLIMB] Front-pole centering timed out; continuing to attach")
            return False

        frame, pole = robot.detect_pole()
        if frame is None:
            print("[CLIMB] No AI camera frame while centering")
            robot.motors.stop()
            time.sleep(0.1)
            continue

        if pole is None:
            missed_frames += 1

            if last_pole is not None and missed_frames <= robot.search_missed_frame_limit:
                print(
                    f"[CLIMB] Front pole briefly lost ({missed_frames}/{robot.search_missed_frame_limit}); "
                    f"holding last action: {last_motor_action}"
                )
                update_front_preview(robot, frame, last_pole, f"CLIMB: HOLD {missed_frames}")

                if last_motor_action == "left":
                    robot.motors.left(robot.center_turn_speed)
                elif last_motor_action == "right":
                    robot.motors.right(robot.center_turn_speed)
                else:
                    robot.motors.stop()

                time.sleep(0.05)
                continue

            stable_frames = 0
            last_pole = None
            last_motor_action = "stop"
            robot.motors.stop()
            print("[CLIMB] Front pole not detected while centering")
            update_front_preview(robot, frame, None, "CLIMB: NO FRONT POLE")
            time.sleep(0.05)
            continue
        
        x, y, w, h = pole.box
        error_x = (x + w / 2.0) - (frame.shape[1] / 2.0)
        missed_frames = 0
        last_pole = pole

        if abs(error_x) <= robot.pole_center_deadband_px:
            stable_frames += 1
            last_motor_action = "stop"
            robot.motors.stop()
            print(
                f"[CLIMB] Front pole centered ({stable_frames}/{robot.pole_stable_frames_required}), "
                f"error_x={error_x:.1f}px"
            )
            update_front_preview(robot, frame, pole, f"CLIMB: CENTERED {stable_frames}")

            if stable_frames >= robot.pole_stable_frames_required:
                return True
        else:
            stable_frames = 0
            if error_x < 0:
                last_motor_action = "left"
                robot.motors.left(robot.center_turn_speed)
                print(f"[CLIMB] Front pole left of center, error_x={error_x:.1f}px")
                update_front_preview(robot, frame, pole, f"CLIMB: LEFT {error_x:.1f}")
            else:
                last_motor_action = "right"
                robot.motors.right(robot.center_turn_speed)
                print(f"[CLIMB] Front pole right of center, error_x={error_x:.1f}px")
                update_front_preview(robot, frame, pole, f"CLIMB: RIGHT {error_x:.1f}")

        time.sleep(0.05)

    return False


def attach_to_pole(robot):
    print(
        f"[CLIMB] Driving forward to attach magnets: "
        f"speed={robot.climb_attach_speed:.2f}, time={robot.climb_attach_seconds:.2f}s"
    )
    robot.motors.forward(robot.climb_attach_speed)
    time.sleep(robot.climb_attach_seconds)
    robot.motors.stop()
    time.sleep(0.2)


def climb_until_bell(robot):
    seen_frames = 0
    started_at = time.time()

    while robot.state == "climbing_pole":
        if time.time() - started_at >= robot.climb_max_seconds:
            robot.motors.stop()
            print("[CLIMB] Timed out before detecting bell; stopping climb")
            return False

        ok, frame = robot.pi_camera.read()
        if not ok or frame is None:
            seen_frames = 0
            robot.motors.forward(robot.climb_speed)
            print("[CLIMB] No Pi camera frame; continuing climb")
            time.sleep(0.05)
            continue

        bell = robot.bell_tracker.detect(frame)
        if bell is None:
            seen_frames = 0
            robot.motors.forward(robot.climb_speed)
            print("[CLIMB] Climbing; bell not detected")
            update_bell_preview(robot, frame, None, "CLIMB: NO BELL")
            time.sleep(0.05)
            continue

        seen_frames += 1
        robot.motors.stop()
        print(f"[CLIMB] Bell detected ({seen_frames}/{robot.climb_bell_stable_frames_required})")
        update_bell_preview(robot, frame, bell, f"CLIMB: BELL {seen_frames}")

        if seen_frames >= robot.climb_bell_stable_frames_required:
            return True

        time.sleep(0.05)

    return False


def run(robot):
    print("[STATE] Climbing pole...")
    robot.pi_camera.picam2.set_controls(striking_controls)
    try:
        robot.bell_tracker.reset()
        center_front_pole(robot)
        attach_to_pole(robot)

        print(f"[CLIMB] Climbing at speed={robot.climb_speed:.2f}")
        if climb_until_bell(robot):
            # robot.motors.stop() # run motors at ~0.5 to hold position after detecting bell
            robot.motors.forward(0.4)
            robot.bell_detected()
        elif robot.state == "climbing_pole":
            robot.climb_failed()
    finally:
        robot.motors.stop()
