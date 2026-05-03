# Capstone Robot

Robotics capstone project for a pole-climbing robot that detects a pole/bell structure, aligns to it, climbs, and strikes the bell.

The project is still in bring-up. Treat scripts under `scripts/` and `misc/` as prototypes unless they are moved into the ROS2 package.

## Repository Layout

- `docs/` - design notes, state machine, wiring, purchase list, and course PDFs.
- `ros2_ws/` - ROS2 workspace for robot nodes.
- `scripts/vision/` - camera, YOLO, OpenCV, and dataset utility prototypes.
- `scripts/control/` - GPIO motor and actuator test scripts.
- `firmware/` - Arduino motor test sketches.
- `train/` - training commands and generated run output.
- `data/` - local datasets and videos. Large generated data should stay out of git.
- `models/` - local model artifacts. Large model files should stay out of git.

## Local Setup

Create a Python environment outside git-tracked source files:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy the example environment file and fill in local secrets:

```bash
cp .env.example .env
```

Do not commit `.env`.

## ROS2 Build

From the ROS2 workspace:

```bash
cd ros2_ws
colcon build
source install/setup.bash
```

Use `colcon build --symlink-install` during Python-only development if your ROS2/Python environment supports editable installs. The package is currently a skeleton. Prototype code should be moved into ROS2 nodes as interfaces settle.

## Vision Prototypes

Run the OpenCV USB pole tracker:

```bash
python3 scripts/vision/pole_track_cv_usb.py --display --camera 0
```

Run Roboflow labeling after setting `ROBOFLOW_API_KEY`:

```bash
source .env
python3 misc/roboflow/label.py data/extracted_frames/phone1 --output misc/roboflow/results.jsonl
```

Training data paths should be relative to the repo when possible. Avoid absolute paths like `/home/<user>/...` in committed files.

## Motor Bring-Up

Secure the robot before any motor test. The test scripts require `--armed` and default to low power:

```bash
python3 scripts/control/test_magnet.py --armed --speed 0.25 --duration 1.0
python3 scripts/control/test_gripper.py --armed --speed 0.25 --duration 1.0
```

Add a physical power cutoff before autonomous testing.

## Current Design Docs

- [State machine](docs/state_machine.md)
- [Alignment plan](docs/alignment_plan.md)
- [Pi battery wiring](docs/pi_battery_wiring.md)
- [Purchase list](docs/purchase_list.md)
