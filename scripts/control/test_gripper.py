from gpiozero import Robot, Motor
from time import sleep

import argparse

# Define your pins. 
# 'left' is one motor/half, 'right' is the other motor/half.
# Format: (Forward_Pin, Backward_Pin)
# robot = Robot(left=(23, 24), right=(27, 17))
# physical pins: left =(16,18), right=(13,11)

motor = Motor(23, 24, pwm=True)

def test_climb():
    try:
        print("--- Gripper Pole Climber Test Start ---")
        
        # 1. Low Power Grip Test
        print("Testing grip: 30% power...")
        motor.forward(1)
        sleep(10)
        
        # 2. Stop and Hold
        # print("Testing hold: Stopping motors...")
        # motor.stop()
        # sleep(3)
        
        # # 3. High Power Climb
        # print("Testing climb: 70% power...")
        # motor.forward(0.7)
        # sleep(4)
        
        # # 4. Descent
        # print("Testing descent: 10% power...")
        # motor.backward(0.1)
        # sleep(3)
        
        motor.stop()
        print("Test sequence complete.")

    except KeyboardInterrupt:
        motor.stop()
        print("\nEmergency Stop triggered.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # p.add_argument('--')
    test_climb()