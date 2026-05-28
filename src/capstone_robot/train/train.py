from ultralytics import YOLO
from pathlib import Path

from capstone_robot.utils import find_repo_root

REPO_ROOT = find_repo_root(__file__)

def main():
    model = YOLO("yolo11n.pt")

    model.train(
        data=REPO_ROOT.joinpath("src/capstone_robot/data/datasets/dataset_pole_final/data.yaml"),
        epochs=100,
        imgsz=640,
        batch=16,
        workers=0,
        project="runs/pole_final",
        name="yolo11n_pole_final_640",
        pretrained=True,
    )


if __name__ == "__main__":
    main()
