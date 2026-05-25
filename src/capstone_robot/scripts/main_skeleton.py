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
from time import sleep
import cv2
from transitions import Machine
from gpiozero import Robot, Servo, AngularServo
from gpiozero import Device
from gpiozero.pins.pigpio import PiGPIOFactory
# import cv2 # For your Camera Module 3
# from picamera2 import Picamera2 # For your AI Camera
from capstone_robot.states import approaching_pole, aligning_bell_circle as aligning_bell, climbing_pole, searching_pole, striking_bell
from capstone_robot.utils import *
from capstone_robot.vision.bell import BellTracker
from capstone_robot.vision.pole_bell import PoleBellTracker

REPO_ROOT = find_repo_root(__file__)

MODEL_PATH = REPO_ROOT / "src/capstone_robot/models/pole_imx/network.rpk"
LABELS_PATH = REPO_ROOT / "src/capstone_robot/models/pole_imx/labels.txt"

### CHANGE PICAM MODULE 3 CONTROLS (STRIKING - ALIGNING)

# Device.pin_factory = PiGPIOFactory()

def load_labels(path):
    if path is None or not path.exists():
        return None
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def choose_pole(detections, frame, target_label="pole"):
    if target_label:
        # poles = [det for det in detections if det.label.lower() == target_label.lower()]
        poles = []
        for det in detections:
            x, y, w, h = det.box
            pole_center_x = x + w / 2.0
            frame_width = frame.shape[1]
            if det.label.lower() == target_label.lower():
                if pole_center_x >= 0.2 * frame_width:
                    if pole_center_x <= 0.8 * frame_width:
                        poles.append(det)
        if poles:
            return max(poles, key=lambda det: det.confidence)

    return max(detections, key=lambda det: det.confidence) if detections else None

class CapstoneRobot(object):
    # Define your state names
    states = ['searching_pole', 'approaching_pole', 'aligning_bell', 'climbing_pole', 'striking_bell', 'done']

    def __init__(self):
        # Hardware Setup (Adjust GPIO pins based on your hardware)
        self.left_rpwm = 'BOARD11'
        self.left_lpwm = 'BOARD7'
        self.right_rpwm = 'BOARD12'
        self.right_lpwm = 'BOARD35'
        self.motors = Robot(left=(self.left_lpwm, self.left_rpwm), right=(self.right_lpwm, self.right_rpwm))
        # self.servo = AngularServo(
        #     16,
        #     min_angle=-60,
        #     max_angle=60,
        #     min_pulse_width=0.001,
        #     max_pulse_width=0.002,
        #     frame_width=0.02
        # )
        self.servo = Servo(16) #BOARD36
        self.servo.value = None
        #     min_pulse_width=0.001,
        #     max_pulse_width=0.002,
        #     frame_width=0.02)
        self.pi_camera = PiCamera(idx=0, width=640, height=480, fps=30)
        self.ai_camera = AiCamera(
            model_path=MODEL_PATH,
            width=640,
            height=480,
            fps=15,
            bbox_normalization=True,
            bbox_order="xy",
        )
        self.ai_labels = load_labels(LABELS_PATH)
        self.pole_bell_tracker = PoleBellTracker()
        self.bell_tracker = BellTracker()
        self.preview_width = 320
        self.preview_server = MjpegPreview(host="0.0.0.0", port=1234, jpeg_quality=75)
        self.preview_server.start()
        print("Preview stream: http://<RPI_IP_ADDRESS>:1234")

        self.search_startup_wait_seconds = 1.0
        self.pole_conf_threshold = 0.5
        self.pole_center_deadband_px = 20
        self.pole_stable_frames_required = 5
        self.search_missed_frame_limit = 6
        self.pole_smooth_alpha = 1
        self.search_turn_speed = 0.3
        self.center_turn_speed = 0.3

        self.approach_hold_frame_limit = 3
        # self.pole_smooth_alpha = 0.75
        self.approach_speed = 0.4
        self.approach_steer_gain = 0.5
        self.approach_stop_width_fraction = 0.16
        self.approach_stop_frames_required = 3
        self.approach_missed_frame_limit = 10

        self.align_turn_speed = 0.3
        self.align_quarter_turn_seconds = 1
        self.orbit_speed = 0.5
        self.alignment_error_threshold_px = 20
        self.alignment_stable_frames_required = 4
        self.alignment_missed_frame_limit = 15

        self.climb_center_timeout_seconds = 2.0
        self.climb_attach_speed = 0.2
        self.climb_attach_seconds = 2
        self.climb_speed = 1.0
        self.climb_bell_stable_frames_required = 3
        self.climb_max_seconds = 20.0
        
        # Initialize Finite State Machine
        self.machine = Machine(model=self, states=CapstoneRobot.states, initial='searching_pole')

        # Define State Transitions: trigger, source state, destination state
        self.machine.add_transition(trigger='pole_found', source='searching_pole', dest='approaching_pole')
        self.machine.add_transition(trigger='pole_reached', source='approaching_pole', dest='aligning_bell')
        self.machine.add_transition(trigger='aligned', source='aligning_bell', dest='climbing_pole')
        self.machine.add_transition(trigger='bell_detected', source='climbing_pole', dest='striking_bell')
        self.machine.add_transition(trigger='climb_failed', source='climbing_pole', dest='done')
        self.machine.add_transition(trigger='mission_complete', source='striking_bell', dest='done')

    def detect_pole(self):
        ok, frame, metadata = self.ai_camera.read()
        if not ok:
            return None, None

        detections = self.ai_camera.get_detections(
            metadata=metadata,
            labels=self.ai_labels,
            threshold=self.pole_conf_threshold,
        )
        return frame, choose_pole(detections, frame)

    def drive(self, left_speed, right_speed):
        left_speed = max(-1.0, min(1.0, left_speed))
        right_speed = max(-1.0, min(1.0, right_speed))
        self.motors.value = (left_speed, right_speed)

    def turn_in_place(self, direction, seconds, speed=None):
        speed = self.align_turn_speed if speed is None else speed

        if direction == "left":
            self.motors.left(speed)
        else:
            self.motors.right(speed)

        sleep(seconds)
        self.motors.stop()
        sleep(0.2)

    def opposite_direction(self, direction):
        return "left" if direction == "right" else "right"

    def update_preview(self, frame, color_format="rgb"):
        if self.preview_server is None or frame is None:
            return

        # if color_format == "rgb":
        #     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        self.preview_server.update(resize_preview(frame, self.preview_width))

    def close(self):
        if self.preview_server is not None:
            self.preview_server.stop()
            self.preview_server = None
        self.servo.value = None
        self.servo.detach()

    def search_for_pole(self):
        searching_pole.run(self)

    def approach_pole(self):
        approaching_pole.run(self)

    def read_upward_alignment(self):
        return aligning_bell.read_upward_alignment(self)

    def wait_for_bell_side(self):
        return aligning_bell.wait_for_bell_side(self)

    def orbit_until_bell_aligned(self):
        return aligning_bell.orbit_until_bell_aligned(self)

    def center_front_pole_for_climb(self):
        return aligning_bell.center_front_pole_for_climb(self)

    def align_to_bell(self):
        aligning_bell.run(self)
    
    def climb_pole(self):
        climbing_pole.run(self)
    
    def strike_bell(self):
        striking_bell.run(self)

    def run_robot(self):
        try:
            while self.state != 'done':
                
                if self.state == 'searching_pole':
                    print("[STATE] Searching for the pole...")
                    self.search_for_pole()

                elif self.state == 'approaching_pole':
                    print("[STATE] Approaching pole...")
                    self.approach_pole()

                elif self.state == 'aligning_bell':
                    print("[STATE] Aligning to bell...")
                    self.align_to_bell()

                elif self.state == 'climbing_pole':
                    print("[STATE] Climbing pole...")
                    self.climb_pole()

                elif self.state == 'striking_bell':
                    print("[STATE] Striking bell...")
                    self.strike_bell()

            print("[INFO] Robot execution successfully completed.")
        finally:
            self.close()

if __name__ == "__main__":
    # robot = CapstoneRobot()
    # robot.run_robot()
    robot = CapstoneRobot()
    try:
        # print(robot.state)
        # robot.search_for_pole()
        # print(robot.state)
        # robot.state = "approaching_pole"
        # robot.approach_pole()
        # robot.run_robot()
        robot.state = 'aligning_bell'
        robot.align_to_bell()
        # robot.state = "climbing_pole"
        # robot.climb_pole()
    finally:
        robot.close()
