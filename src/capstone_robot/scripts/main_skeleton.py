import time
from transitions import Machine
from gpiozero import Robot, Servo
# import cv2 # For your Camera Module 3
# from picamera2 import Picamera2 # For your AI Camera
from capstone_robot.utils import *

REPO_ROOT = find_repo_root(__file__)

MODEL_PATH = REPO_ROOT / ""

class CapstoneRobot(object):
    # Define your state names
    states = ['searching_pole', 'approaching_pole', 'climbing_pole', 'striking_bell', 'done']

    def __init__(self):
        # Hardware Setup (Adjust GPIO pins based on your hardware)
        self.motors = Robot(left=(7, 8), right=(9, 10))
        self.servo = Servo(11)
        self.pi_camera = PiCamera(width=640, height=480, fps=30)
        self.ai_camera = AiCamera(
            model_path=MODEL_PATH,
            width=640,
            height=480,
            fps=30,
            bbox_normalization=True,
            bbox_order="xy",
        )
        
        # Initialize Finite State Machine
        self.machine = Machine(model=self, states=CapstoneRobot.states, initial='searching_pole')

        # Define State Transitions: trigger, source state, destination state
        self.machine.add_transition(trigger='pole_found', source='searching_pole', dest='approaching_pole')
        self.machine.add_transition(trigger='pole_reached', source='approaching_pole', dest='aligning_bell')
        self.machine.add_transition(trigger='aligned', source='aligning_bell', dest='climbing_pole')
        self.machine.add_transition(trigger='bell_detected', source='climbing_pole', dest='striking_bell')
        self.machine.add_transition(trigger='mission_complete', source='striking_bell', dest='done')

    def run_robot(self):
        while self.state != 'done':
            
            # assume pole is already centered in frame
            if self.state == 'searching_pole':
                print("[STATE] Searching for the pole...")
                # self.motors.forward(speed=0.3) # Slow cruise
                # TODO: Insert AI Camera code here. If bounding box center matches:
                approach_pole()
                self.pole_found() 
                time.sleep(2) # Simulation block
                self.pole_found()

            elif self.state == 'approaching_pole':
                print("[STATE] Aligning and approaching pole...")
                # TODO: Use vision data to correct steering (Differential Drive)
                # If physical contact switches close or motor current spikes (torque change):
                self.pole_reached()

            elif self.state == 'climbing_pole':
                print("[STATE] Magnets engaged. Climbing...")
                # Engage full power to overcome gravity
                self.motors.forward(speed=1.0) 
                # TODO: Camera Module 3 checks for the bell overhead via HSV contour tracking
                time.sleep(3) # Simulation block
                self.bell_detected()

            elif self.state == 'striking_bell':
                print("[STATE] Bell within reach! Striking...")
                self.motors.stop()
                
                # Actuate the servo arm to hit the bell
                self.servo.min()
                time.sleep(0.5)
                self.servo.max() # Quick swing
                time.sleep(0.5)
                self.servo.detach() # Turn off servo to save power
                
                self.mission_complete()

        print("[INFO] Robot execution successfully completed.")

if __name__ == "__main__":
    robot = CapstoneRobot()
    robot.run_robot()
