from gpiozero import Robot
from time import sleep

robot = Robot(left=('BOARD35', 'BOARD12'), right=('BOARD11', 'BOARD7'))
sleep(5)
robot.value = (-0.1, -0.2)
while True:
    continue
# sleep(3)
robot.stop()
