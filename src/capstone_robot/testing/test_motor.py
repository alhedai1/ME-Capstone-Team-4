import argparse
from time import sleep

from gpiozero import Robot, Motor

# speed to hold position on pole: 0.3

# Motor driver pins.
# Each side uses two PWM pins: one for each direction.
LEFT_LPWM = "BOARD7"
LEFT_RPWM = "BOARD11"
RIGHT_LPWM = "BOARD35"
RIGHT_RPWM = "BOARD12"

robot = Robot(left=(LEFT_LPWM, LEFT_RPWM), right=(RIGHT_LPWM, RIGHT_RPWM))
sleep(3)
robot.forward(0.3)
sleep(1000)

# motor = Motor(forward='BOARD35', backward='BOARD32')
# motor.forward(0.1)
# sleep(10)