#!/usr/bin/env python3
import argparse
from pathlib import Path

from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "runs/detect/runs/pole/yolo26n_640/weights/best.pt"
DEFAULT_DATA = REPO_ROOT / "data/dataset/data.yaml"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a trained YOLO model for Raspberry Pi deployment."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="trained YOLO .pt model path; defaults to the newest runs/**/best.pt if missing",
    )
    parser.add_argument(
        "--format",
        default="ncnn",
        help="Ultralytics export format, e.g. ncnn, onnx, openvino, tflite, imx, hailo",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="export image size")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="dataset YAML for calibration")
    parser.add_argument("--half", action="store_true", help="export FP16 when the format supports it")
    parser.add_argument("--int8", action="store_true", help="export INT8 when the format supports it")
    parser.add_argument("--device", default=None, help="export device, e.g. cpu, 0")
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="simplify exported graph when the format supports it",
    )
    return parser.parse_args()


def newest_best_model():
    candidates = sorted(
        (REPO_ROOT / "runs").rglob("best.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def resolve_model_path(model_path):
    if model_path.exists():
        return model_path

    if model_path == DEFAULT_MODEL:
        candidate = newest_best_model()
        if candidate is not None:
            return candidate

    raise SystemExit(f"Model does not exist: {model_path}")


def main():
    args = parse_args()
    model_path = resolve_model_path(args.model)

    export_kwargs = {
        "format": args.format,
        "imgsz": args.imgsz,
    }

    if args.half:
        export_kwargs["half"] = True
    if args.int8:
        export_kwargs["int8"] = True
        if args.data.exists():
            export_kwargs["data"] = str(args.data)
        else:
            raise SystemExit(f"Calibration dataset YAML does not exist: {args.data}")
    if args.device is not None:
        export_kwargs["device"] = args.device
    if args.simplify:
        export_kwargs["simplify"] = True

    print(f"Using model: {model_path}")
    print(f"Export format: {args.format}")
    model = YOLO(str(model_path))
    exported_path = model.export(**export_kwargs)
    print(f"Exported model: {exported_path}")


if __name__ == "__main__":
    main()
