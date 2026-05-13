#!/usr/bin/env bash
set -euo pipefail

yolo detect train data=data/roboflow/dataset/data.yaml model=yolo26n.pt epochs=100 imgsz=320 name=yolo26n_sz320
