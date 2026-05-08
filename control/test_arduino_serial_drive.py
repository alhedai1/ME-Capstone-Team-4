#!/usr/bin/env python3
import argparse
import time


def clamp(value, low, high):
    return max(low, min(high, value))


def parse_args():
    parser = argparse.ArgumentParser(description="Test Arduino serial motor control from the Raspberry Pi")
    parser.add_argument("--port", default="COM7", help="Arduino serial port")
    parser.add_argument("--baud", type=int, default=115200, help="Arduino serial baud rate")
    parser.add_argument("--armed", action="store_true", help="required before motor commands are sent")
    parser.add_argument("--left", type=float, default=0.18, help="left wheel speed from -1.0 to 1.0")
    parser.add_argument("--right", type=float, default=0.18, help="right wheel speed from -1.0 to 1.0")
    parser.add_argument("--duration", type=float, default=5.0, help="movement duration in seconds")
    return parser.parse_args()


def send(serial_port, command):
    serial_port.write((command + "\n").encode("ascii"))
    serial_port.flush()


def main():
    args = parse_args()

    if not args.armed:
        raise SystemExit("Refusing to move motors. Re-run with --armed when the robot is secured.")
    if args.duration <= 0:
        raise SystemExit("--duration must be positive")

    left = clamp(args.left, -1.0, 1.0)
    right = clamp(args.right, -1.0, 1.0)

    try:
        import serial
    except ImportError as exc:
        raise SystemExit("pyserial is not installed. Install with: python3 -m pip install pyserial") from exc

    ser = serial.Serial(args.port, args.baud, timeout=1.0)
    time.sleep(2.0)

    try:
        print(f"Driving left={left:+.2f} right={right:+.2f} for {args.duration:.1f}s")
        send(ser, f"M {left:.3f} {right:.3f}")
        time.sleep(args.duration)
        send(ser, "S")
        print("Stopped")
    except KeyboardInterrupt:
        print("Keyboard interrupt; stopping.")
        send(ser, "S")
    finally:
        send(ser, "S")
        ser.close()


if __name__ == "__main__":
    main()
