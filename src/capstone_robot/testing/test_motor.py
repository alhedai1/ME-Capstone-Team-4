from gpiozero import Robot
from time import sleep

robot = Robot(left=('BOARD32', 'BOARD33'), right=('BOARD35', 'BOARD12'))
robot.value = (0.1, 0.1)
sleep(2)
robot.stop()