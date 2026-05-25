import argparse
from time import sleep

from gpiozero import Robot


# Motor driver pins.
# Each side uses two PWM pins: one for each direction.
LEFT_LPWM = "BOARD35"
LEFT_RPWM = "BOARD12"
RIGHT_LPWM = "BOARD11"
RIGHT_RPWM = "BOARD7"


def turn_in_place(robot, direction, speed, seconds):
    """Rotate the robot in place for a fixed time."""
    if direction == "left":
        robot.left(speed)
    else:
        robot.right(speed)

    sleep(seconds)
    robot.stop()


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate fixed-time left/right 90 degree turns.")
    parser.add_argument("--direction", choices=["left", "right"], default="right")
    parser.add_argument("--speed", type=float, default=0.25, help="Motor speed from 0.0 to 1.0")
    parser.add_argument("--seconds", type=float, default=1.0, help="Turn duration for one trial")
    parser.add_argument("--pause", type=float, default=2.0, help="Pause between repeated trials")
    parser.add_argument("--trials", type=int, default=1, help="Number of repeated turns to run")
    parser.add_argument("--interactive", action="store_true", help="Prompt for speed/duration after each trial")
    return parser.parse_args()


def clamp_speed(speed):
    return max(0.0, min(1.0, speed))


def run_trial(robot, direction, speed, seconds, trial_number=None):
    speed = clamp_speed(speed)
    label = f" trial {trial_number}" if trial_number is not None else ""
    print(f"[MOTOR TEST] Starting{label}: direction={direction}, speed={speed:.2f}, seconds={seconds:.2f}")

    # Short countdown gives you time to move your hand away from the robot.
    for remaining in range(3, 0, -1):
        print(f"[MOTOR TEST] Turning in {remaining}...")
        sleep(1)

    turn_in_place(robot, direction, speed, seconds)
    print("[MOTOR TEST] Stopped")


def interactive_loop(robot, args):
    direction = args.direction
    speed = args.speed
    seconds = args.seconds

    while True:
        run_trial(robot, direction, speed, seconds)

        print("\nMeasure the turn angle, then enter new values.")
        print("Examples: right 0.25 0.8    left 0.3 1.1    q")
        command = input("direction speed seconds > ").strip().lower()

        if command in {"q", "quit", "exit"}:
            break

        parts = command.split()
        if len(parts) != 3 or parts[0] not in {"left", "right"}:
            print("[MOTOR TEST] Invalid input. Use: left 0.25 1.0")
            continue

        try:
            direction = parts[0]
            speed = float(parts[1])
            seconds = float(parts[2])
        except ValueError:
            print("[MOTOR TEST] Speed and seconds must be numbers")


def main():
    args = parse_args()
    robot = Robot(left=(LEFT_LPWM, LEFT_RPWM), right=(RIGHT_LPWM, RIGHT_RPWM))

    try:
        if args.interactive:
            interactive_loop(robot, args)
        else:
            for trial in range(1, args.trials + 1):
                run_trial(robot, args.direction, args.speed, args.seconds, trial_number=trial)
                if trial < args.trials:
                    sleep(args.pause)
    finally:
        # Always stop motors if the script crashes or Ctrl+C is pressed.
        robot.stop()


if __name__ == "__main__":
    main()
