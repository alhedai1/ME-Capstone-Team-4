#!/usr/bin/env python3
"""Run the Roboflow bell/pole workflow on one image or a folder of images."""

import argparse
import json
import os
from pathlib import Path
from time import sleep

from inference_sdk import InferenceHTTPClient


DEFAULT_WORKSPACE = "sth-mswrs"
DEFAULT_WORKFLOW = "find-bells-and-poles-2"
DEFAULT_API_URL = "https://detect.roboflow.com"


def parse_args():
    parser = argparse.ArgumentParser(description="Label images with a Roboflow workflow")
    parser.add_argument("input", type=Path, help="image file or folder of images")
    parser.add_argument("--output", type=Path, default=Path("misc/roboflow/results.jsonl"))
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    parser.add_argument("--workflow", default=DEFAULT_WORKFLOW)
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--pause-seconds", type=float, default=1.0)
    return parser.parse_args()


def collect_images(input_path):
    if input_path.is_file():
        return [input_path]

    patterns = ("*.jpg", "*.jpeg", "*.png")
    images = []
    for pattern in patterns:
        images.extend(input_path.glob(pattern))
    return sorted(images)


def chunks(items, size):
    for index in range(0, len(items), size):
        yield items[index : index + size]


def main():
    args = parse_args()
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("Set ROBOFLOW_API_KEY before running this script.")

    images = collect_images(args.input)
    if not images:
        raise SystemExit(f"No images found in {args.input}")

    client = InferenceHTTPClient(api_url=args.api_url, api_key=api_key)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("a", encoding="utf-8") as output_file:
        for batch_index, batch in enumerate(chunks(images, args.batch_size), start=1):
            batch_paths = [str(path) for path in batch]
            print(f"Sending batch {batch_index}: {len(batch_paths)} image(s)")
            record = {"batch_index": batch_index, "images": batch_paths}

            try:
                record["result"] = client.run_workflow(
                    workspace_name=args.workspace,
                    workflow_id=args.workflow,
                    images={"image": batch_paths},
                    use_cache=True,
                )
            except Exception as exc:
                record["error"] = str(exc)

            output_file.write(json.dumps(record) + "\n")
            output_file.flush()
            sleep(args.pause_seconds)


if __name__ == "__main__":
    main()
