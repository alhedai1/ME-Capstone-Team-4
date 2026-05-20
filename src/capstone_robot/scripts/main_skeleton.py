'''
This code is for operating a robot to accomplish a task for a capstone design course.
The task is to strike a bell hanging from the top of a pole twice with an interval of at leaast 3 seconds between the 2 strikes.
The bell is hanging from the end of a horizontal rod extending from the top of the pole. The bell is also oscillating vertically using a motor with some unknown frequency between 0.1Hz~0.5Hz, and a range of 30cm.
The pole is 3m high, the rod is 25cm long.
The robot starts at some random location 2-4m away from the base of the pole. The direction the robot faces can be chosen freely, so we will just place it facing the pole, so on startup the pole should be in the frame of the fronn-facing camera.
Our robot will use the front camera and YOLO to detect the pole and move towards the pole, stopping just a bit before it. Our robot has magnetic wheels at the front so if it contacts the pole it attaches immediately.
After it stops, the upwards facing camera detects the pole (appears tapered here, extending from the bottom of the frame towards the top) and the bell, computes the pole centerline and bell center, error between them and direction the bell is on relative to pole.
The robot then rotates right or left, 90 degrees. The camera also rotates obviously, so the pole appears extending from the right or left side of the frame now (this could also possibly be used to keep robot aligned 90 degrees with pole while it rotates around it)
The robot then drives around the pole, continuously detecting the pole centerline and bell center until they are aligned. Once aligned, the robot stops, and rotates back towards the pole, using the front-facing camera to center the pole in the frame.
The robot then drives forward to allow the magentic wheels to contact the pole. Then the motors should probably be driven at max power to oppose gravity and climb up (same motors are used for ground driving and climbing)
while climbing, the previously upwards facing camera is now facing horizontally, and once it detects the bell (opencv can be used since the bell is gold and metallic, and will be very close once in frame), the robot can stop. 
The robot can then strike the bell when it is in frame using an arm controlled by a stepper motor. Wait for at least 3 seconds, then strike again when the bell is detected.
'''

import time
from transitions import Machine
from gpiozero import Robot, Servo
# import cv2 # For your Camera Module 3
# from picamera2 import Picamera2 # For your AI Camera
from capstone_robot.utils import *

REPO_ROOT = find_repo_root(__file__)

MODEL_PATH = REPO_ROOT / "src/capstone_robot/models/pole_imx/network.rpk"
LABELS_PATH = REPO_ROOT / "src/capstone_robot/models/pole_imx/labels.txt"


def load_labels(path):
    if path is None or not path.exists():
        return None
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def choose_pole(detections, target_label="pole"):
    if target_label:
        poles = [det for det in detections if det.label.lower() == target_label.lower()]
        if poles:
            return max(poles, key=lambda det: det.confidence)

    return max(detections, key=lambda det: det.confidence) if detections else None

class CapstoneRobot(object):
    # Define your state names
    states = ['searching_pole', 'approaching_pole', 'aligning_bell', 'climbing_pole', 'striking_bell', 'done']

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
        self.ai_labels = load_labels(LABELS_PATH)

        self.pole_conf_threshold = 0.5
        self.pole_center_deadband_px = 35
        self.pole_stable_frames_required = 5
        self.search_turn_speed = 0.25
        self.center_turn_speed = 0.18
        
        # Initialize Finite State Machine
        self.machine = Machine(model=self, states=CapstoneRobot.states, initial='searching_pole')

        # Define State Transitions: trigger, source state, destination state
        self.machine.add_transition(trigger='pole_found', source='searching_pole', dest='approaching_pole')
        self.machine.add_transition(trigger='pole_reached', source='approaching_pole', dest='aligning_bell')
        self.machine.add_transition(trigger='aligned', source='aligning_bell', dest='climbing_pole')
        self.machine.add_transition(trigger='bell_detected', source='climbing_pole', dest='striking_bell')
        self.machine.add_transition(trigger='mission_complete', source='striking_bell', dest='done')

    def search_for_pole(self):
        stable_frames = 0

        while self.state == 'searching_pole':
            ok, frame, metadata = self.ai_camera.read()
            if not ok:
                print("[WARN] No AI camera frame/metadata received")
                self.motors.stop()
                time.sleep(0.1)
                continue

            detections = self.ai_camera.get_detections(
                metadata=metadata,
                labels=self.ai_labels,
                threshold=self.pole_conf_threshold,
            )
            pole = choose_pole(detections)

            if pole is None:
                stable_frames = 0
                print("[SEARCH] Pole not detected; rotating slowly")
                self.motors.right(self.search_turn_speed)
                time.sleep(0.05)
                continue

            x, y, w, h = pole.box
            pole_center_x = x + w / 2.0
            frame_center_x = frame.shape[1] / 2.0
            error_x = pole_center_x - frame_center_x

            if abs(error_x) <= self.pole_center_deadband_px:
                stable_frames += 1
                self.motors.stop()
                print(
                    f"[SEARCH] Pole centered ({stable_frames}/{self.pole_stable_frames_required}), "
                    f"error_x={error_x:.1f}px, conf={pole.confidence:.2f}"
                )

                if stable_frames >= self.pole_stable_frames_required:
                    self.pole_found()
                    return
            else:
                stable_frames = 0
                if error_x < 0:
                    print(f"[SEARCH] Pole left of center, error_x={error_x:.1f}px")
                    self.motors.left(self.center_turn_speed)
                else:
                    print(f"[SEARCH] Pole right of center, error_x={error_x:.1f}px")
                    self.motors.right(self.center_turn_speed)

            time.sleep(0.05)

    def run_robot(self):
        while self.state != 'done':
            
            if self.state == 'searching_pole':
                print("[STATE] Searching for the pole...")
                self.search_for_pole()

            elif self.state == 'approaching_pole':
                print("[STATE] Approaching pole...")
                # TODO: Use vision data to correct steering (Differential Drive)
                # stop when close to pole, based on size of pole in frame
                self.pole_reached()

            elif self.state == 'aligning_bell':
                print("[STATE] Aligning to bell...")
                # TODO: Use Pi Camera and opencv to detect pole and bell, and detect which side bell is on, and error between pole centerline and bell center
                # rotate left or right 90 degrees.
                # move around pole, continuously detecting pole and bell until pole centerline aligns with bell center
                # stop and rotate towards pole, use front camera to center pole in frame
                self.aligned()

            elif self.state == 'climbing_pole':
                print("[STATE] Climbing pole...")
                # move forward a bit so front wheels 
                # Engage full power to overcome gravity
                # self.motors.forward(speed=1.0) 
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
