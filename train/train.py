from ultralytics import YOLO


def main():
    model = YOLO("yolo26n.pt")

    model.train(
        data="data/dataset/data.yaml",
        epochs=100,
        imgsz=640,
        batch=-1,
        project="runs/pole",
        name="yolo26n_640",
        pretrained=True,
    )


if __name__ == "__main__":
    main()
