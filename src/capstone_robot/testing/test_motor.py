from gpiozero import Robot
from time import sleep

robot = Robot(left=('BOARD32', 'BOARD33'), right=('BOARD12', 'BOARD35'))
robot.value = (0.5, 0)
sleep(5)
robot.stop()