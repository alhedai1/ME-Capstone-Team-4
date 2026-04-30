from gpiozero import Motor
from signal import pause

# Setup the motor on pins 23 and 24
motor = Motor(forward=23, backward=24)

# Move forward at 50% speed
print("Motor starting: Forward at 50%")
motor.forward(0.5)

# After some logic or time, you can stop it
# motor.stop()
pause()
