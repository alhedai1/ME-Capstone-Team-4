import time
from gpiozero import DistanceSensor # Standard object library for Pi 5

# VCC → Pin 2 (5V Power)
# GND → Pin 6 (Ground)
# Trig → Pin 16 (GPIO 23)
# Echo → Connect to a 1 kΩ resistor

# Plug one leg of the 1 kΩ resistor into the Echo row, and the other leg into a blank row (let's call it Row 10).
# Plug a jumper wire from Row 10 to Pin 18 on the Pi.
# Plug one leg of the 2 kΩ resistor into that exact same Row 10.
# Plug the other leg of the 2 kΩ resistor into the GND rail of your breadboard.

# Initialize sensor using the specific GPIO pins (Not physical pin numbers)
# GPIO 23 is physical pin 16, GPIO 24 is physical pin 18
sensor = DistanceSensor(echo='18', trigger='BOARD16')

print("Testing Ultrasonic Sensor... Press Ctrl+C to exit.")

try:
    while True:
        # gpiozero handles the pulsing, microsecond calculations, 
        # and speed-of-sound conversion automatically.
        # sensor.distance returns the value in meters.
        distance_cm = sensor.distance * 100
        
        print(f"Distance: {distance_cm:.1f} cm")
        
        # Wait half a second before reading again
        time.sleep(0.5)

except KeyboardInterrupt:
    print("\nMeasurement stopped by user.")
