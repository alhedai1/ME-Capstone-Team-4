import argparse
from time import sleep

from gpiozero import Robot, Motor

# speed to hold position on pole: 0.3
# rotate 90 degrees (rubber wheels): speed: 0.2, time: 2.7

# Motor driver pins.
# Each side uses two PWM pins: one for each direction.
LEFT_LPWM = "BOARD7"
LEFT_RPWM = "BOARD11"
RIGHT_LPWM = "BOARD35"
RIGHT_RPWM = "BOARD12"

robot = Robot(left=(LEFT_LPWM, LEFT_RPWM), right=(RIGHT_LPWM, RIGHT_RPWM))
sleep(3)
# while True:
#     robot.forward(0.2)
#     sleep(2)
#     robot.backward(0.2)
#     sleep(2)
robot.forward(0.4)
# sleep(5)
# robot.value = (0.5, 0.2)
sleep(100)

# sleep(1000)

# motor = Motor(forward='BOARD35', backward='BOARD32')
# motor.forward(0.1)
# sleep(10)