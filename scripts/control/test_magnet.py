from gpiozero import Robot
from time import sleep

import argparse

# Define your pins. 
# 'left' is one motor/half, 'right' is the other motor/half.
# Format: (Forward_Pin, Backward_Pin)
# robot = Robot(left=(23, 24), right=(27, 17))
robot = Robot(left=(27, 17), right=(23, 24))
# physical pins: left =(16,18), right=(13,11)

def parse_args():
    parser = argparse.ArgumentParser(description="Run a short magnetic wheel drive test")
    parser.add_argument("--armed", action="store_true", help="required to move the motors")
    parser.add_argument("--speed", type=float, default=0.25, help="motor speed from 0.0 to 1.0")
    parser.add_argument("--duration", type=float, default=1.0, help="test duration in seconds")
    return parser.parse_args()


def test_climb(speed, duration):
    if not 0.0 <= speed <= 1.0:
        raise ValueError("--speed must be between 0.0 and 1.0")
    if duration <= 0:
        raise ValueError("--duration must be positive")

    try:
        print("--- Magnet Pole Climber Test Start ---")
        print(f"Running forward at {speed:.0%} power for {duration:.1f}s")
        robot.forward(speed)
        sleep(duration)
        print("Test sequence complete.")

    except KeyboardInterrupt:
        print("\nEmergency Stop triggered.")
    finally:
        robot.stop()

if __name__ == "__main__":
    args = parse_args()
    if not args.armed:
        raise SystemExit("Refusing to move motors. Re-run with --armed when the robot is secured.")
    test_climb(args.speed, args.duration)
