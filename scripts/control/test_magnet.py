from gpiozero import Robot
from time import sleep

import argparse

# Define your pins. 
# 'left' is one motor/half, 'right' is the other motor/half.
# Format: (Forward_Pin, Backward_Pin)
# robot = Robot(left=(23, 24), right=(27, 17))
robot = Robot(left=(27, 17), right=(23, 24))
# physical pins: left =(16,18), right=(13,11)

def test_climb():
    try:
        print("--- Magnet Pole Climber Test Start ---")
        
        # 1. Low Power Grip Test
        # Seeing if the magnets hold while the motor provides just a little lift
        print("Testing grip: 10% power...")
        robot.forward(1)
        sleep(2)
        
        # 2. Stop and Hold
        # Does the robot slide down when power is cut? 
        # (Depends on gear ratio and magnet strength)
        # print("Testing hold: Stopping motors...")
        # robot.stop()
        # sleep(3)
        
        # 3. High Power Climb
        # print("Testing climb: 70% power...")
        # robot.forward(0.7)
        # sleep(4)
        
        # 4. Descent
        # Be careful here; gravity + motor speed can make it come down fast!
        # print("Testing descent: 10% power...")
        # robot.backward(0.1)
        # sleep(3)
        
        robot.stop()
        print("Test sequence complete.")

    except KeyboardInterrupt:
        robot.stop()
        print("\nEmergency Stop triggered.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # p.add_argument('--')
    test_climb()