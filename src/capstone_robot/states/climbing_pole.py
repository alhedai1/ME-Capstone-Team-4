import time

import cv2

from capstone_robot.vision.bell_circle_climb import BellCircle

striking_controls = {
    "LensPosition": 12.0,           # Instantly force lens to maximum physical close-up limit
    "ExposureValue": 0.0
}


def setting(robot, name, default):
    return getattr(robot, name, default)


def smooth_box(old_box, new_box, alpha):
    if old_box is None:
        return new_box

    return tuple(int(alpha * new + (1.0 - alpha) * old) for old, new in zip(old_box, new_box))


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


def update_circle_preview(robot, frame, detection, status):
    if frame is None:
        return

    vis = frame.copy()
    height, width = vis.shape[:2]
    cv2.line(vis, (width // 2, 0), (width // 2, height), (80, 80, 80), 1)

    if detection is not None:
        x, y, radius = detection.circle
        cv2.circle(vis, (x, y), radius, (255, 0, 0), 2)
        cv2.circle(vis, (x, y), 3, (0, 255, 0), -1)

    cv2.putText(vis, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    robot.update_preview(vis)


def get_climb_bell_circle(robot):
    detector = getattr(robot, "climb_bell_circle", None)
    if detector is None:
        detector = BellCircle(
            color_format="rgb",
            dp=setting(robot, "climb_circle_dp", 1.5),
            min_dist=setting(robot, "climb_circle_min_dist", 5),
            param1=setting(robot, "climb_circle_param1", 50),
            param2=setting(robot, "climb_circle_param2", 50),
            min_radius=setting(robot, "climb_circle_min_radius", 10),
            max_radius=setting(robot, "climb_circle_max_radius", 50),
            startup_max_radius=setting(robot, "climb_circle_startup_max_radius", 50),
            tracking_max_radius=setting(robot, "climb_circle_tracking_max_radius", 130),
            lost_after_frames=setting(robot, "climb_circle_lost_after_frames", 8),
            startup_confirm_threshold=setting(robot, "climb_circle_startup_confirm_frames", 2),
            show_debug=setting(robot, "climb_circle_show_debug", False),
        )
        robot.climb_bell_circle = detector
    return detector


def reset_climb_bell_circle(robot):
    robot.climb_bell_circle = None
    return get_climb_bell_circle(robot)


def read_ai_frame(robot):
    ok, frame, _metadata = robot.ai_camera.read()
    if not ok or frame is None:
        return None
    return frame


def approach_front_pole(robot):
    close_frames = 0
    missed_frames = 0
    smoothed_box = None
    last_left_speed = 0.0
    last_right_speed = 0.0

    while robot.state == "climbing_pole":
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
    time.sleep(robot.start_climb_settle_seconds)


def climb_until_bell(robot):
    seen_frames = 0
    started_at = time.time()

    while robot.state == "climbing_pole":
        if time.time() - started_at >= robot.climb_max_seconds:
            robot.motors.stop()
            print("[CLIMB] Timed out before detecting bell; stopping climb")
            return False

        ok, frame = robot.pi_camera.read()
        # climb a bit before checking for bell
        if (time.time() - started_at <= 5) or not ok or frame is None:
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
        # robot.motors.stop()
        print(f"[CLIMB] Bell detected ({seen_frames}/{robot.climb_bell_stable_frames_required})")
        update_bell_preview(robot, frame, bell, f"CLIMB: BELL {seen_frames}")

        if seen_frames >= robot.climb_bell_stable_frames_required:
            # continue a bit more, then hold position
            time.sleep(1)
            robot.motors.forward(robot.climb_hold_speed)
            return True

        time.sleep(0.05)

    return False


def drive_climb_tracking_bell(robot, detection, frame_width, speed=None):
    speed = setting(robot, "climb_fast_speed", setting(robot, "climb_speed", 0.6)) if speed is None else speed
    error_x = detection.x - frame_width / 2.0
    normalized_error = error_x / max(1.0, frame_width / 2.0)
    steer = max(
        -setting(robot, "climb_circle_max_steer", 0.25),
        min(
            setting(robot, "climb_circle_max_steer", 0.25),
            normalized_error * setting(robot, "climb_circle_steer_gain", 0.35),
        ),
    )
    robot.drive(speed + steer, speed - steer)
    return error_x, steer


def wait_for_circle_lock(robot, detector):
    started_at = time.time()
    stable_frames = 0

    while robot.state == "climbing_pole":
        if time.time() - started_at >= setting(robot, "climb_circle_lock_timeout_seconds", 5.0):
            print("[CLIMB-PASSIVE] Bell circle lock timed out")
            return False

        frame = read_ai_frame(robot)
        if frame is None:
            robot.motors.stop()
            print("[CLIMB-PASSIVE] No AI camera frame while locking bell circle")
            time.sleep(0.05)
            continue

        detection = detector.detect(frame)
        if detection is None:
            stable_frames = 0
            robot.motors.stop()
            update_circle_preview(robot, frame, None, "CLIMB: FIND CIRCLE")
            time.sleep(0.05)
            continue

        error_x = detection.x - frame.shape[1] / 2.0
        if abs(error_x) <= setting(robot, "climb_circle_center_deadband_px", 45):
            stable_frames += 1
            robot.motors.stop()
        elif error_x < 0:
            stable_frames = 0
            robot.motors.left(setting(robot, "climb_circle_search_turn_speed", 0.25))
        else:
            stable_frames = 0
            robot.motors.right(setting(robot, "climb_circle_search_turn_speed", 0.25))

        print(
            f"[CLIMB-PASSIVE] Bell circle lock "
            f"({stable_frames}/{setting(robot, 'climb_circle_lock_frames_required', 3)}), "
            f"error_x={error_x:.1f}px, radius={detection.radius}"
        )
        update_circle_preview(robot, frame, detection, f"CLIMB: LOCK {stable_frames}")

        if stable_frames >= setting(robot, "climb_circle_lock_frames_required", 3):
            return True

        time.sleep(0.05)

    return False


def climb_to_passive_hit(robot, detector):
    started_at = time.time()
    seen_once = False

    while robot.state == "climbing_pole":
        if time.time() - started_at >= setting(robot, "climb_max_seconds", 20.0):
            robot.motors.stop()
            print("[CLIMB-PASSIVE] Timed out before passive bell hit")
            return False

        frame = read_ai_frame(robot)
        if frame is None:
            robot.motors.forward(setting(robot, "climb_fast_speed", setting(robot, "climb_speed", 0.6)))
            print("[CLIMB-PASSIVE] No AI camera frame; climbing straight")
            time.sleep(0.05)
            continue

        detection = detector.detect(frame)
        if detection is None:
            robot.motors.forward(setting(robot, "climb_fast_speed", setting(robot, "climb_speed", 0.6)))
            update_circle_preview(robot, frame, None, "CLIMB: LOST")

            if seen_once:
                print("[CLIMB-PASSIVE] Bell circle left frame; assuming passive hit")
                return True

            print("[CLIMB-PASSIVE] Climbing; bell circle not locked yet")
            time.sleep(0.05)
            continue

        seen_once = True
        error_x, steer = drive_climb_tracking_bell(robot, detection, frame.shape[1])
        update_circle_preview(robot, frame, detection, f"CLIMB: err={error_x:.0f} r={detection.radius}")
        print(
            f"[CLIMB-PASSIVE] Tracking bell circle while climbing, "
            f"error_x={error_x:.1f}px, radius={detection.radius}, steer={steer:.2f}"
        )

        if detection.radius >= setting(robot, "climb_circle_hit_radius_px", 115):
            print("[CLIMB-PASSIVE] Bell circle is very close; assuming passive hit")
            return True

        time.sleep(0.05)

    return False


def descend_and_reacquire(robot):
    descend_seconds = setting(robot, "climb_passive_descend_seconds", 0.8)
    descend_speed = setting(robot, "climb_passive_descend_speed", 0.25)
    hold_seconds = setting(robot, "climb_passive_hold_seconds", 1.0)

    print(
        f"[CLIMB-PASSIVE] Descending after hit: "
        f"speed={descend_speed:.2f}, time={descend_seconds:.2f}s"
    )
    robot.motors.backward(descend_speed)
    time.sleep(descend_seconds)
    robot.motors.forward(setting(robot, "climb_hold_speed", 0.3))
    time.sleep(hold_seconds)

    detector = reset_climb_bell_circle(robot)
    wait_for_circle_lock(robot, detector)


def passive_climb_strike(robot):
    detector = reset_climb_bell_circle(robot)
    cycles = setting(robot, "climb_passive_hit_cycles", 2)
    min_interval_seconds = setting(robot, "climb_passive_min_hit_interval_seconds", 3.0)
    hit_count = 0
    last_hit_at = None

    if not wait_for_circle_lock(robot, detector):
        return False

    while robot.state == "climbing_pole" and (cycles <= 0 or hit_count < cycles):
        if last_hit_at is not None:
            remaining = min_interval_seconds - (time.time() - last_hit_at)
            if remaining > 0:
                robot.motors.forward(setting(robot, "climb_hold_speed", 0.3))
                print(f"[CLIMB-PASSIVE] Waiting {remaining:.2f}s before next hit attempt")
                time.sleep(remaining)

        if not climb_to_passive_hit(robot, detector):
            return False

        hit_count += 1
        last_hit_at = time.time()
        print(f"[CLIMB-PASSIVE] Passive hit {hit_count}/{cycles if cycles > 0 else 'inf'}")

        if cycles > 0 and hit_count >= cycles:
            robot.motors.forward(setting(robot, "climb_hold_speed", 0.3))
            return True

        descend_and_reacquire(robot)
        detector = get_climb_bell_circle(robot)

    return True


def run(robot):
    print("[STATE] Climbing pole...")
    try:
        center_front_pole(robot)
        approach_front_pole(robot)
        center_front_pole(robot)
        attach_to_pole(robot)

        print(f"[CLIMB-PASSIVE] Climbing with AI bell-circle tracking at speed={setting(robot, 'climb_fast_speed', robot.climb_speed):.2f}")
        if passive_climb_strike(robot):
            robot.motors.forward(robot.climb_hold_speed)
            if hasattr(robot, "passive_climb_complete"):
                robot.passive_climb_complete()
            else:
                robot.state = "done"
        elif robot.state == "climbing_pole":
            robot.climb_failed()
    finally:
        robot.motors.stop()
