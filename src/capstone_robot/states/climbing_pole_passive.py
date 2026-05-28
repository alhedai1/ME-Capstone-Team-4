import time

import cv2

from capstone_robot.vision.bell_circle_climb import BellCircle


def setting(robot, name, default):
    return getattr(robot, name, default)


def read_ai_frame(robot):
    ok, frame, _metadata = robot.ai_camera.read()
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
            startup_max_radius=setting(robot, "climb_circle_startup_max_radius", 50),
            tracking_max_radius=setting(robot, "climb_circle_tracking_max_radius", 130),
            lost_after_frames=setting(robot, "climb_circle_lost_after_frames", 8),
            startup_confirm_threshold=setting(robot, "climb_circle_startup_confirm_frames", 2),
            show_debug=setting(robot, "climb_circle_show_debug", False),
        )
        robot.climb_bell_circle = detector
    return detector


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


def attach_to_pole(robot):
    print(
        f"[CLIMB-PASSIVE] Driving forward to attach magnets: "
        f"speed={robot.climb_attach_speed:.2f}, time={robot.climb_attach_seconds:.2f}s"
    )
    robot.motors.forward(robot.climb_attach_speed)
    time.sleep(robot.climb_attach_seconds)
    robot.motors.stop()
    time.sleep(robot.start_climb_settle_seconds)


def run(robot):
    print("[STATE] Passive climbing pole loop...")
    detector = get_bell_circle_detector(robot)
    climb_speed = setting(robot, "climb_full_speed", 1.0)
    min_repeat_seconds = setting(robot, "climb_passive_min_hit_interval_seconds", 3.0)
    loop_sleep = setting(robot, "climb_loop_sleep_seconds", 0.05)

    phase = "climb"
    hit_count = 0
    last_gone_at = None

    try:
        attach_to_pole(robot)
        print(
            f"[CLIMB-PASSIVE] Looping: full-speed climb while circle is visible, "
            f"0-speed slip down while circle is gone. speed={climb_speed:.2f}"
        )

        while robot.state == "climbing_pole":
            frame = read_ai_frame(robot)
            if frame is None:
                if phase == "climb":
                    robot.motors.forward(climb_speed)
                else:
                    robot.motors.stop()
                print("[CLIMB-PASSIVE] No AI camera frame")
                time.sleep(loop_sleep)
                continue

            detection = detector.detect(frame)
            now = time.time()

            if phase == "climb":
                if detection is None:
                    hit_count += 1
                    last_gone_at = now
                    phase = "descend"
                    robot.motors.stop()
                    status = f"DESCEND hit={hit_count}: bell gone"
                    print(f"[CLIMB-PASSIVE] Bell gone; hit={hit_count}. Motors at 0 so robot slips down.")
                else:
                    robot.motors.forward(climb_speed)
                    status = f"CLIMB hit={hit_count} r={detection.radius}"
            else:
                robot.motors.stop()
                elapsed_since_gone = now - last_gone_at if last_gone_at is not None else 0.0
                if detection is not None and elapsed_since_gone >= min_repeat_seconds:
                    phase = "climb"
                    status = f"CLIMB AGAIN hit={hit_count} r={detection.radius}"
                    print("[CLIMB-PASSIVE] Bell reacquired after slipping down; climbing again.")
                elif detection is not None:
                    remaining = min_repeat_seconds - elapsed_since_gone
                    status = f"WAIT {remaining:.1f}s hit={hit_count} r={detection.radius}"
                else:
                    status = f"DESCEND hit={hit_count}: searching"

            update_preview(robot, frame, detection, status)
            time.sleep(loop_sleep)

    finally:
        robot.motors.stop()
