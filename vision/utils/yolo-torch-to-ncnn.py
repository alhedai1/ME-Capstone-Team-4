#!/usr/bin/env python3
import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Export a YOLO PyTorch model to NCNN")
    parser.add_argument("--model", type=Path, default=Path("train/runs/detect/yolo26n_sz320/weights/best.pt"))
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--half", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(str(args.model))
    exported_model_path = model.export(format="ncnn", half=args.half, imgsz=args.imgsz)
    print(exported_model_path)


if __name__ == "__main__":
    main()
