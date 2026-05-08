#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = REPO_ROOT / "data/extracted_frames"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/dataset/raw"
DEFAULT_EXCLUDED_FOLDERS = {"bell-bottom"}
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Copy extracted frame images into data/dataset/raw as img1, img2, ..."
    )
    parser.add_argument(
        "append_dir",
        nargs="?",
        type=Path,
        help="folder of images to append to the existing raw dataset",
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--exclude",
        action="append",
        default=sorted(DEFAULT_EXCLUDED_FOLDERS),
        help="folder name to skip; can be passed multiple times",
    )
    return parser.parse_args()


def iter_images_recursive(input_dir):
    for image_path in sorted(input_dir.rglob("*")):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            yield image_path


def iter_images(input_dir, excluded_folders):
    for folder in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        if folder.name in excluded_folders:
            continue

        yield from iter_images_recursive(folder)


def existing_image_numbers(output_dir):
    numbers = []
    if not output_dir.is_dir():
        return numbers

    for image_path in output_dir.iterdir():
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if not image_path.stem.startswith("img"):
            continue

        number_text = image_path.stem.removeprefix("img")
        if number_text.isdigit():
            numbers.append(int(number_text))

    return numbers


def main():
    args = parse_args()

    if args.append_dir is not None:
        if not args.append_dir.is_dir():
            raise SystemExit(f"Append folder does not exist: {args.append_dir}")
        images = list(iter_images_recursive(args.append_dir))
        existing_numbers = existing_image_numbers(args.output_dir)
        start_index = max(existing_numbers, default=0) + 1
        action = "Appended"
        prevent_overwrite = True
    else:
        if not args.input_dir.is_dir():
            raise SystemExit(f"Input folder does not exist: {args.input_dir}")
        excluded_folders = set(args.exclude)
        images = list(iter_images(args.input_dir, excluded_folders))
        start_index = 1
        action = "Copied"
        prevent_overwrite = False

    if not images:
        source_dir = args.append_dir if args.append_dir is not None else args.input_dir
        raise SystemExit(f"No images found in: {source_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    last_output_path = None
    for index, image_path in enumerate(images, start=start_index):
        output_path = args.output_dir / f"img{index}{image_path.suffix.lower()}"
        if prevent_overwrite and output_path.exists():
            raise SystemExit(f"Output file already exists: {output_path}")
        shutil.copy2(image_path, output_path)
        last_output_path = output_path

    print(f"{action} {len(images)} images to: {args.output_dir}")
    print(f"Last image: {last_output_path.name}")


if __name__ == "__main__":
    main()
