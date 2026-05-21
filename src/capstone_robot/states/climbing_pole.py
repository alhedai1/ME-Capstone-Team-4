import time


def run(robot):
    print("[STATE] Climbing pole...")
    # TODO: Move forward to attach magnetic wheels, then climb.
    # robot.motors.forward(speed=1.0)
    time.sleep(3)
    robot.bell_detected()
