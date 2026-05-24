from gpiozero import Robot
from time import sleep

robot = Robot(left=('BOARD35', 'BOARD12'), right=('BOARD11', 'BOARD7'))
sleep(2)
robot.value = (0.3, 0.3)
sleep(10)
robot.stop()
