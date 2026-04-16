import json
from pathlib import Path

def convert_roboflow_json_to_yolo(json_path, output_dir=None):
    json_path = Path(json_path)

    if output_dir is None:
        output_dir = json_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path, "r") as f:
        data = json.load(f)

    preds = data["predictions"]
    img_w = preds["image"]["width"]
    img_h = preds["image"]["height"]
    detections = preds["predictions"]

    txt_lines = []
    for det in detections:
        class_id = det["class_id"]
        x_center = det["x"] / img_w
        y_center = det["y"] / img_h
        width = det["width"] / img_w
        height = det["height"] / img_h

        txt_lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )

    txt_path = output_dir / (json_path.stem + ".txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(txt_lines))

    print(f"Saved: {txt_path}")

def convert_folder(folder_path, output_dir=None):
    folder = Path(folder_path)
    for json_file in folder.glob("*.json"):
        convert_roboflow_json_to_yolo(json_file, output_dir)

# Example usage:
# convert_roboflow_json_to_yolo("20260403_172554_00000.json")
# convert_folder("./roboflow_jsons", "./labels")

convert_folder("/home/ahmed/Other/capstone/data/roboflow/labels_json", "/home/ahmed/Other/capstone/data/roboflow/labels")