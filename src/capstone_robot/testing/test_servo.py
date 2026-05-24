import os
os.environ['GPIOZERO_PIN_FACTORY'] = 'lgpio' # Modern OS backend

from gpiozero import PWMOutputDevice, Servo
from time import sleep

# Initialize GPIO 18 (Physical Pin 12) as direct Hardware PWM
# servo = PWMOutputDevice(18, frequency=50)
servo = Servo(16)

# Try testing your movements now:
for i in range(1):
    # servo.value = 0.10  # Forward
    # servo.min()
    servo.max()
    print("forward")
    sleep(1.2)
    # servo.value = 0.0 # Stop
    # servo.max()
    # servo.mid()
    # print("stop")
    # sleep(2)
    servo.min()
    print("backward")
    sleep(5)
    # servo.mid()
    # print("stop")
    # sleep(2)
servo.value = None



        # robot.servo.min()  # Move to -1
        # print("min")
        # sleep(1)
        # # # robot.servo.angle = -30
        # # sleep(1.2)
        # robot.servo.mid()
        # print("mid")
        # sleep(1)
        # robot.servo.max()  # Move to 1
        # print("max")
        # sleep(1)

        # while True:
        #     cmd = input("Enter angle -60 to 60, or q: ")

        #     if cmd == "q":
        #         break

        #     angle = float(cmd)
        #     robot.servo.angle = angle
        #     print(f"Moved to {angle}")
        #     sleep(0.5)
        # while True:
        #     for angle in [-30, 0, 30, 0]:
        #         print(angle)
        #         robot.servo.angle = angle
        #         sleep(10)

        # robot.servo.min()
        # print("CW")
        # sleep(5)
        # robot.servo.mid()
        # print("stop")
        # sleep(5)
        # robot.servo.max()
        # print("CCW")
        # sleep(5)

        # while True:
        #     user_input = input("Enter fine-tune value: ")
        #     if user_input.lower() == 'exit':
        #         robot.servo.value = None
        #         break
        #     try:
        #         val = float(user_input)
        #         robot.servo.value = val
        #     except ValueError:
        #         print("Please enter a valid decimal number.")