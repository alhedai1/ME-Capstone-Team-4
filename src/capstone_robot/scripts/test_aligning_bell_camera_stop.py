import argparse
import math
import time

import cv2

from capstone_robot.utils import MjpegPreview, PiCamera, resize_preview, rotate_frame
from capstone_robot.vision.pole_bell import PoleBellTracker


def draw_line(frame, line, color=(0, 255, 0), thickness=2):
    vx, vy, x0, y0 = line
    t = max(frame.shape[:2]) * 2
    x1 = int(x0 - vx * t)
    y1 = int(y0 - vy * t)
    x2 = int(x0 + vx * t)
    y2 = int(y0 + vy * t)
    cv2.line(frame, (x1, y1), (x2, y2), color, thickness)


def line_angle_deg(line):
    vx, vy, _, _ = line
    angle = math.degrees(math.atan2(vy, vx)) % 180.0
    return angle


def angle_error_deg(angle, target):
    if target == "horizontal":
        return min(angle, 180.0 - angle)
    if target == "vertical":
        return abs(angle - 90.0)
    raise ValueError(f"Unsupported target angle: {target}")


def rotate_for_orbit(side):
    if side == "left":
        return "ccw"
    if side == "right":
        return "cw"
    return "180"


def opposite_direction(direction):
    return "left" if direction == "right" else "right"


def command_turn(motors, direction, speed):
    if direction == "left":
        motors.left(speed)
    else:
        motors.right(speed)


class DryRunMotors:
    def left(self, speed):
        print(f"[DRY RUN] motors.left({speed})")

    def right(self, speed):
        print(f"[DRY RUN] motors.right({speed})")

    def forward(self, speed):
        print(f"[DRY RUN] motors.forward({speed})")

    def stop(self):
        print("[DRY RUN] motors.stop()")


def create_motors(args):
    if args.dry_run:
        return DryRunMotors()

    from gpiozero import Robot

    return Robot(
        left=(args.left_lpwm, args.left_rpwm),
        right=(args.right_lpwm, args.right_rpwm),
    )


def read_alignment(camera, tracker, rotation):
    ok, frame = camera.read()
    if not ok or frame is None:
        return None, None

    frame = rotate_frame(frame, rotation)
    return frame, tracker.detect(frame)


def draw_alignment(frame, alignment, status):
    if frame is None:
        return None

    vis = frame.copy()
    if alignment is not None:
        draw_line(vis, alignment.pole_line)
        bx, by, br = alignment.bell
        cv2.circle(vis, (int(bx), int(by)), int(br), (255, 0, 0), 2)
        cv2.circle(vis, (int(bx), int(by)), 4, (255, 0, 0), -1)

        angle = line_angle_deg(alignment.pole_line)
        cv2.putText(
            vis,
            f"error={alignment.error_px:.1f}px side={alignment.side} angle={angle:.1f}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(vis, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return vis


def update_display(args, preview, frame, alignment, status):
    vis = draw_alignment(frame, alignment, status)
    if vis is None:
        return True

    if preview is not None:
        preview.update(resize_preview(vis, args.preview_width))

    if args.show:
        cv2.imshow("Aligning bell camera-stop test", vis)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            return False

    return True


def wait_for_bell_side(args, camera, tracker, preview):
    missed_frames = 0
    aligned_frames = 0

    print("[ALIGN TEST] Detecting initial bell side with 180 camera rotation")
    while True:
        frame, alignment = read_alignment(camera, tracker, "180")
        if alignment is None:
            missed_frames += 1
            aligned_frames = 0
            status = f"INITIAL: NEED POLE/BELL {missed_frames}"
            print(f"[ALIGN TEST] Need pole and bell ({missed_frames}/{args.missed_frame_limit})")
            if not update_display(args, preview, frame, None, status):
                return None
            if missed_frames >= args.missed_frame_limit:
                return None
            time.sleep(0.05)
            continue

        missed_frames = 0
        if abs(alignment.error_px) <= args.alignment_error_threshold_px:
            aligned_frames += 1
            status = f"INITIAL: ALIGNED {aligned_frames}/{args.stable_frames}"
            print(
                f"[ALIGN TEST] Already aligned ({aligned_frames}/{args.stable_frames}), "
                f"error={alignment.error_px:.1f}px"
            )
            if not update_display(args, preview, frame, alignment, status):
                return None
            if aligned_frames >= args.stable_frames:
                return "aligned"
            time.sleep(0.05)
            continue

        print(f"[ALIGN TEST] Bell is on the {alignment.side}, error={alignment.error_px:.1f}px")
        update_display(args, preview, frame, alignment, f"INITIAL: {alignment.side}")
        return alignment.side


def turn_until_pole_angle(args, motors, camera, tracker, preview, direction, target, rotation):
    stable_frames = 0
    missed_frames = 0
    started_at = time.time()

    tracker.reset()
    print(f"[ALIGN TEST] Turning {direction} until pole line is {target}")
    command_turn(motors, direction, args.turn_speed)

    try:
        while time.time() - started_at < args.turn_timeout:
            frame, alignment = read_alignment(camera, tracker, rotation)
            if alignment is None:
                missed_frames += 1
                stable_frames = 0
                status = f"TURN {direction}: LOST {missed_frames}"
                print(f"[ALIGN TEST] No pole/bell while turning ({missed_frames}/{args.missed_frame_limit})")
                if not update_display(args, preview, frame, None, status):
                    return False
                if missed_frames >= args.missed_frame_limit:
                    return False
                time.sleep(0.05)
                continue

            missed_frames = 0
            angle = line_angle_deg(alignment.pole_line)
            angle_error = angle_error_deg(angle, target)

            if angle_error <= args.angle_threshold_deg:
                stable_frames += 1
                status = f"TURN {direction}: {target.upper()} {stable_frames}/{args.turn_stable_frames}"
                print(
                    f"[ALIGN TEST] Pole {target} candidate "
                    f"({stable_frames}/{args.turn_stable_frames}), angle={angle:.1f}"
                )
                if stable_frames >= args.turn_stable_frames:
                    return True
            else:
                stable_frames = 0
                status = f"TURN {direction}: angle={angle:.1f} err={angle_error:.1f}"

            if not update_display(args, preview, frame, alignment, status):
                return False
            time.sleep(0.05)

        print(f"[ALIGN TEST] Turn timed out after {args.turn_timeout:.1f}s")
        return False
    finally:
        motors.stop()
        time.sleep(args.stop_pause)


def orbit_until_bell_aligned(args, motors, camera, tracker, preview, rotation):
    stable_frames = 0
    missed_frames = 0
    started_at = time.time()

    tracker.reset()
    print(f"[ALIGN TEST] Orbiting with {rotation} camera rotation until bell is aligned")

    try:
        while time.time() - started_at < args.orbit_timeout:
            frame, alignment = read_alignment(camera, tracker, rotation)
            if alignment is None:
                missed_frames += 1
                stable_frames = 0
                motors.stop()
                status = f"ORBIT: LOST {missed_frames}"
                print(f"[ALIGN TEST] Lost pole/bell while orbiting ({missed_frames}/{args.missed_frame_limit})")
                if not update_display(args, preview, frame, None, status):
                    return False
                if missed_frames >= args.missed_frame_limit:
                    return False
                time.sleep(0.05)
                continue

            missed_frames = 0
            error = alignment.error_px
            if abs(error) <= args.alignment_error_threshold_px:
                stable_frames += 1
                motors.stop()
                status = f"ORBIT: ALIGNED {stable_frames}/{args.stable_frames}"
                print(
                    f"[ALIGN TEST] Bell aligned ({stable_frames}/{args.stable_frames}), "
                    f"error={error:.1f}px"
                )
                if stable_frames >= args.stable_frames:
                    return True
            else:
                stable_frames = 0
                motors.forward(args.orbit_speed)
                status = f"ORBIT: error={error:.1f}"
                print(f"[ALIGN TEST] Orbiting, side={alignment.side}, error={error:.1f}px")

            if not update_display(args, preview, frame, alignment, status):
                return False
            time.sleep(0.05)

        print(f"[ALIGN TEST] Orbit timed out after {args.orbit_timeout:.1f}s")
        return False
    finally:
        motors.stop()
        time.sleep(args.stop_pause)


def run_alignment_test(args, camera, motors, preview):
    tracker = PoleBellTracker(color_format="rgb")

    side = wait_for_bell_side(args, camera, tracker, preview)
    if side is None:
        return False

    if side == "aligned":
        print("[ALIGN TEST] Bell started aligned; no orbit turn needed")
        return True

    if not turn_until_pole_angle(args, motors, camera, tracker, preview, side, "horizontal", "180"):
        return False

    orbit_rotation = rotate_for_orbit(side)
    if not orbit_until_bell_aligned(args, motors, camera, tracker, preview, orbit_rotation):
        return False

    face_pole_direction = opposite_direction(side)
    if not turn_until_pole_angle(args, motors, camera, tracker, preview, face_pole_direction, "vertical", "180"):
        return False

    print("[ALIGN TEST] Completed aligning-bell camera-stop sequence")
    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone aligning-bell behavior test with camera-based turn stop conditions."
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--turn-speed", type=float, default=0.25)
    parser.add_argument("--orbit-speed", type=float, default=0.2)
    parser.add_argument("--alignment-error-threshold-px", type=float, default=20.0)
    parser.add_argument("--stable-frames", type=int, default=4)
    parser.add_argument("--missed-frame-limit", type=int, default=15)
    parser.add_argument("--angle-threshold-deg", type=float, default=12.0)
    parser.add_argument("--turn-stable-frames", type=int, default=3)
    parser.add_argument("--turn-timeout", type=float, default=5.0)
    parser.add_argument("--orbit-timeout", type=float, default=25.0)
    parser.add_argument("--stop-pause", type=float, default=0.2)
    parser.add_argument("--left-lpwm", default="BOARD35")
    parser.add_argument("--left-rpwm", default="BOARD12")
    parser.add_argument("--right-lpwm", default="BOARD11")
    parser.add_argument("--right-rpwm", default="BOARD7")
    parser.add_argument("--preview-width", type=int, default=640)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=1234)
    parser.add_argument("--jpeg-quality", type=int, default=75)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Run camera/detection only and print motor commands.")
    return parser.parse_args()


def main():
    args = parse_args()
    camera = PiCamera(idx=args.camera_index, width=args.width, height=args.height, fps=args.fps)
    motors = create_motors(args)
    preview = None

    if not args.no_preview:
        preview = MjpegPreview(host=args.host, port=args.port, jpeg_quality=args.jpeg_quality)
        preview.start()
        print(f"Preview stream: http://<RPI_IP_ADDRESS>:{args.port}")

    try:
        success = run_alignment_test(args, camera, motors, preview)
        print(f"[ALIGN TEST] Result: {'SUCCESS' if success else 'FAILED'}")
    except KeyboardInterrupt:
        print("\n[ALIGN TEST] Stopped by user")
    finally:
        motors.stop()
        if preview is not None:
            preview.stop()
        if args.show:
            cv2.destroyAllWindows()
        camera.release()


if __name__ == "__main__":
    main()
