# import argparse
# from time import sleep

# from gpiozero import Robot, Motor

# # speed to hold position on pole: 0.3
# # rotate 90 degrees (rubber wheels): speed: 0.2, time: 2.7

# # Motor driver pins.
# # Each side uses two PWM pins: one for each direction.
# LEFT_LPWM = "BOARD7"
# LEFT_RPWM = "BOARD11"
# RIGHT_LPWM = "BOARD35"
# RIGHT_RPWM = "BOARD12"

# robot = Robot(left=(LEFT_LPWM, LEFT_RPWM), right=(RIGHT_LPWM, RIGHT_RPWM))
# sleep(3)
# # while True:
# #     robot.forward(0.2)
# #     sleep(2)
# #     robot.backward(0.2)
# #     sleep(2)
# robot.forward(0.4)
# # sleep(5)
# # robot.value = (0.5, 0.2)
# sleep(100)

# # sleep(1000)

# # motor = Motor(forward='BOARD35', backward='BOARD32')
# # motor.forward(0.1)
# # sleep(10)


from time import sleep
from gpiozero import Robot

# Motor driver pins
LEFT_LPWM = "BOARD7"
LEFT_RPWM = "BOARD11"
RIGHT_LPWM = "BOARD35"
RIGHT_RPWM = "BOARD12"

robot = Robot(
    left=(LEFT_LPWM, LEFT_RPWM),
    right=(RIGHT_LPWM, RIGHT_RPWM)
)

def clamp(x, low=-1.0, high=1.0):
    return max(low, min(high, x))

print("Robot motor terminal control")
print("--------------------------------")
print("Commands:")
print("  f 0.3        forward at speed 0.3")
print("  b 0.3        backward at speed 0.3")
print("  l 0.2        turn left at speed 0.2")
print("  r 0.2        turn right at speed 0.2")
print("  set 0.4 0.2  set left/right motor speeds directly")
print("  s            stop")
print("  q            quit")
print("--------------------------------")

try:
    while True:
        cmd = input("> ").strip().lower().split()

        if not cmd:
            continue

        if cmd[0] == "q":
            break

        elif cmd[0] == "s":
            robot.stop()
            print("Stopped")

        elif cmd[0] in ["f", "forward"]:
            speed = clamp(float(cmd[1]))
            robot.forward(speed)
            print(f"Forward speed = {speed}")

        elif cmd[0] in ["b", "backward"]:
            speed = clamp(float(cmd[1]))
            robot.backward(speed)
            print(f"Backward speed = {speed}")

        elif cmd[0] in ["l", "left"]:
            speed = clamp(float(cmd[1]))
            robot.left(speed)
            print(f"Left turn speed = {speed}")

        elif cmd[0] in ["r", "right"]:
            speed = clamp(float(cmd[1]))
            robot.right(speed)
            print(f"Right turn speed = {speed}")

        elif cmd[0] == "set":
            left_speed = clamp(float(cmd[1]))
            right_speed = clamp(float(cmd[2]))
            robot.value = (left_speed, right_speed)
            print(f"Motor speeds: left={left_speed}, right={right_speed}")

        else:
            print("Unknown command")

except KeyboardInterrupt:
    print("\nKeyboard interrupt")

finally:
    robot.stop()
    print("Motors stopped")