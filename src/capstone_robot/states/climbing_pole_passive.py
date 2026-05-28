import time

import cv2

from capstone_robot.states.climbing_pole import center_front_pole
from capstone_robot.utils import FixedRateLoop
from capstone_robot.vision.bell_circle_climb import BellCircle


def setting(robot, name, default):
    return getattr(robot, name, default)


def read_ai_frame(robot):
    ok, frame, _metadata = robot.ai_camera.read()
    if not ok or frame is None:
        return None
    return frame

def read_pi_frame(robot):
    ok, frame, _metadata = robot.pi_camera.read()
    if not ok or frame is None:
        return None
    return frame


def get_bell_circle_detector(robot):
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
            startup_max_radius=setting(robot, "climb_circle_startup_max_radius", 30),
            tracking_max_radius=setting(robot, "climb_circle_tracking_max_radius", 130),
            lost_after_frames=setting(robot, "climb_circle_lost_after_frames", 8),
            startup_confirm_threshold=setting(robot, "climb_circle_startup_confirm_frames", 2),
            show_debug=setting(robot, "climb_circle_show_debug", False),
        )
        robot.climb_bell_circle = detector
    return detector

def make_pi_failure_bell_detector(robot):
    return BellCircle(
        color_format="rgb",
        dp=setting(robot, "climb_failure_circle_dp", 1.5),
        min_dist=setting(robot, "climb_failure_circle_min_dist", 5),
        param1=setting(robot, "climb_failure_circle_param1", 50),
        param2=setting(robot, "climb_failure_circle_param2", 50),
        min_radius=setting(robot, "climb_failure_circle_min_radius", 10),
        max_radius=setting(robot, "climb_failure_circle_max_radius", 50),
        startup_max_radius=setting(robot, "climb_failure_circle_startup_max_radius", 50),
        tracking_max_radius=setting(robot, "climb_failure_circle_tracking_max_radius", 130),
        lost_after_frames=setting(robot, "climb_failure_circle_lost_after_frames", 1),
        startup_confirm_threshold=setting(robot, "climb_failure_bell_confirm_frames", 5),
        show_debug=setting(robot, "climb_failure_circle_show_debug", False),
    )

def update_preview(robot, frame, detection, status):
    if frame is None:
        return

    vis = frame.copy()
    height, width = vis.shape[:2]
    cv2.line(vis, (width // 2, 0), (width // 2, height), (80, 80, 80), 1)

    if detection is not None:
        x, y, radius = detection.circle
        cv2.circle(vis, (x, y), radius, (255, 0, 0), 2)
        cv2.circle(vis, (x, y), 3, (0, 255, 0), -1)
        cv2.putText(
            vis,
            f"x={x} y={y} r={radius}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(vis, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    robot.update_preview(vis)

def ramp_climb_speed(robot, target_speed):
    start_speed = setting(robot, "climb_ramp_start_speed", 0.4)
    seconds = setting(robot, "climb_ramp_seconds", 0.75)
    steps = setting(robot, "climb_ramp_steps", 8)

    for step in range(1, steps + 1):
        speed = start_speed + (target_speed - start_speed) * (step / steps)
        robot.motors.forward(speed)
        time.sleep(seconds / steps)

def back_off_from_pole(robot):
    speed = setting(robot, "climb_backoff_speed", 0.25)
    seconds = setting(robot, "climb_backoff_seconds", 0.4)

    if seconds <= 0 or speed <= 0:
        return

    print(f"[CLIMB-PASSIVE] Backing off before attach: speed={speed:.2f}, time={seconds:.2f}s")
    robot.motors.backward(speed)
    time.sleep(seconds)
    robot.motors.stop()
    time.sleep(setting(robot, "climb_backoff_settle_seconds", 0.2))


def attach_to_pole(robot):
    print(
        f"[CLIMB-PASSIVE] Driving forward to attach magnets: "
        f"speed={robot.climb_attach_speed:.2f}, time={robot.climb_attach_seconds:.2f}s"
    )
    robot.motors.forward(robot.climb_attach_speed)
    time.sleep(robot.climb_attach_seconds)
    robot.motors.stop()
    time.sleep(robot.start_climb_settle_seconds)

def pi_camera_still_sees_bell_after_climb_attempt(robot, climb_speed):
    delay_seconds = setting(robot, "climb_failure_check_delay_seconds", 1.5)
    check_seconds = setting(robot, "climb_failure_check_seconds", 1.5)
    confirm_frames = setting(robot, "climb_failure_bell_confirm_frames", 5)

    detector = make_pi_failure_bell_detector(robot)
    loop = FixedRateLoop(period_seconds=setting(robot, "control_loop_period_seconds", 0.05))

    started_at = time.time()
    seen_frames = 0

    while robot.state == "climbing_pole":
        elapsed = time.time() - started_at
        if elapsed >= delay_seconds + check_seconds:
            return False

        robot.motors.forward(climb_speed)

        frame = read_pi_frame(robot)
        if frame is None:
            seen_frames = 0
            robot.log("[CLIMB-PASSIVE] No Pi camera frame during climb failure check")
            loop.sleep()
            continue

        if elapsed < delay_seconds:
            update_preview(robot, frame, None, "CLIMB CHECK: delay")
            loop.sleep()
            continue

        detection = detector.detect(frame)

        # Count only fresh detections, not held last_circle frames.
        if detection is not None and detector.missed_frames == 0:
            seen_frames += 1
            status = f"CLIMB FAIL CHECK: PI BELL {seen_frames}/{confirm_frames}"
        else:
            seen_frames = 0
            status = "CLIMB FAIL CHECK: no pi bell"

        update_preview(robot, frame, detection, status)

        if seen_frames >= confirm_frames:
            robot.log("[CLIMB-PASSIVE] Pi camera still sees bell after climb attempt; retrying climb.")
            robot.motors.stop()
            return True

        loop.sleep()

    return False

def run(robot):
    print("[STATE] Passive climbing pole loop...")
    detector = get_bell_circle_detector(robot)
    climb_speed = setting(robot, "climb_full_speed", 1.0)
    min_repeat_seconds = setting(robot, "climb_passive_min_hit_interval_seconds", 3.0)
    loop = FixedRateLoop(period_seconds=setting(robot, "control_loop_period_seconds", 0.05))

    phase = "climb"
    hit_count = 0
    bell_reacquired_at = None

    try:
        # back_off_from_pole(robot)
        # if not center_front_pole(robot):
        #     robot.log("[CLIMB-PASSIVE] Centering timed out; attaching anyway")
        # attach_to_pole(robot)
        # ramp_climb_speed(robot, climb_speed)
        # print(
        #     f"[CLIMB-PASSIVE] Looping: full-speed climb while circle is visible, "
        #     f"0-speed slip down while circle is gone. speed={climb_speed:.2f}"
        # )
        max_attempts = setting(robot, "climb_attach_retry_attempts", 2)
        attempt = 0

        while robot.state == "climbing_pole":
            attempt += 1
            robot.log(f"[CLIMB-PASSIVE] Climb attach attempt {attempt}/{max_attempts + 1}")

            back_off_from_pole(robot)
            if not center_front_pole(robot):
                robot.log("[CLIMB-PASSIVE] Centering timed out; attaching anyway")

            if attempt >= 4:
                back_off_from_pole(robot)
                robot.motors.left(0.3)
                time.sleep(0.1)
                robot.motors.forward(0.2)
                time.sleep(0.1)

            attach_to_pole(robot)
            ramp_climb_speed(robot, climb_speed)

            failed_attempt = pi_camera_still_sees_bell_after_climb_attempt(robot, climb_speed)
            if not failed_attempt:
                break

            
            if attempt > max_attempts:
                robot.log("[CLIMB-PASSIVE] Climb retry limit reached; continuing passive climb loop anyway.")
                break

        while robot.state == "climbing_pole":
            frame = read_ai_frame(robot)
            if frame is None:
                if phase == "climb":
                    robot.motors.forward(climb_speed)
                else:
                    robot.motors.stop()
                robot.log("[CLIMB-PASSIVE] No AI camera frame")
                loop.sleep()
                continue

            detection = detector.detect(frame)
            now = time.time()

            if phase == "climb":
                if detection is None:
                    # time.sleep(1)
                    hit_count += 1
                    bell_reacquired_at = None
                    phase = "descend"
                    robot.motors.stop()
                    status = f"DESCEND hit={hit_count}: bell gone"
                    robot.log(f"[CLIMB-PASSIVE] Bell gone; hit={hit_count}. Motors at 0 so robot slips down.")
                else:
                    robot.motors.forward(climb_speed)
                    status = f"CLIMB hit={hit_count} r={detection.radius}"
            else:
                robot.motors.stop()
                if detection is None:
                    status = f"DESCEND hit={hit_count}: searching"
                else:
                    if bell_reacquired_at is None:
                        bell_reacquired_at = now
                        robot.log("[CLIMB-PASSIVE] Bell reacquired; waiting before climbing again.")

                    elapsed_since_reacquired = now - bell_reacquired_at
                    if elapsed_since_reacquired >= min_repeat_seconds:
                        phase = "climb"
                        bell_reacquired_at = None
                        status = f"CLIMB AGAIN hit={hit_count} r={detection.radius}"
                        robot.log("[CLIMB-PASSIVE] Reacquire wait complete; climbing again.")
                    else:
                        remaining = min_repeat_seconds - elapsed_since_reacquired
                        status = f"WAIT {remaining:.1f}s hit={hit_count} r={detection.radius}"

            update_preview(robot, frame, detection, status)
            loop.sleep()

    finally:
        robot.motors.stop()
