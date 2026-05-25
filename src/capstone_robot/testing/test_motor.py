import argparse
from time import sleep

from gpiozero import Robot, Motor


# Motor driver pins.
# Each side uses two PWM pins: one for each direction.
LEFT_LPWM = "BOARD7"
LEFT_RPWM = "BOARD11"
RIGHT_LPWM = "BOARD35"
RIGHT_RPWM = "BOARD12"

# robot = Robot(left=(LEFT_LPWM, LEFT_RPWM), right=(RIGHT_LPWM, RIGHT_RPWM))
# sleep(3)
# robot.forward(0.5)
# sleep(10)

motor = Motor(forward='BOARD7', backward='BOARD11')
motor.forward(0.1)
sleep(10)