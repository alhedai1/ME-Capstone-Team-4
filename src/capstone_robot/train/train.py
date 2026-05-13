from ultralytics import YOLO
from pathlib import Path

from capstone_robot.utils import find_repo_root

REPO_ROOT = find_repo_root(__file__)

def main():
    model = YOLO("yolo11n.pt")

    model.train(
        data=REPO_ROOT.joinpath("src/capstone_robot/data/datasets/dataset_upward_2/data.yaml"),
        epochs=100,
        imgsz=640,
        batch=-1,
        workers=0,
        project="runs/upward_2",
        name="yolo11n_upward_2_640",
        pretrained=True,
    )


if __name__ == "__main__":
    main()
