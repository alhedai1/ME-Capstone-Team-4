from gpiozero import Robot
from time import sleep

robot = Robot(left=('BOARD32', 'BOARD33'), right=('BOARD12', 'BOARD35'))
sleep(2)
robot.value = (0.3, 0.3)
sleep(10)
robot.stop()
