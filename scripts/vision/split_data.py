from pathlib import Path
import random
import shutil

dataset_dir = Path("/home/ahmed/Other/capstone/data/roboflow/dataset/data")   # has images/ and labels/
images_dir = dataset_dir / "images"
labels_dir = dataset_dir / "labels"

out_img_train = images_dir / "train"
out_img_val   = images_dir / "val"
out_lbl_train = labels_dir / "train"
out_lbl_val   = labels_dir / "val"

for d in [out_img_train, out_img_val, out_lbl_train, out_lbl_val]:
    d.mkdir(parents=True, exist_ok=True)

image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

pairs = []
for img_path in images_dir.iterdir():
    if img_path.is_file() and img_path.suffix.lower() in image_exts:
        label_path = labels_dir / (img_path.stem + ".txt")
        if label_path.exists():
            pairs.append((img_path, label_path))

random.seed(42)
random.shuffle(pairs)

train_ratio = 0.8
n_train = int(len(pairs) * train_ratio)

train_pairs = pairs[:n_train]
val_pairs = pairs[n_train:]

for img_path, label_path in train_pairs:
    shutil.move(str(img_path), str(out_img_train / img_path.name))
    shutil.move(str(label_path), str(out_lbl_train / label_path.name))

for img_path, label_path in val_pairs:
    shutil.move(str(img_path), str(out_img_val / img_path.name))
    shutil.move(str(label_path), str(out_lbl_val / label_path.name))

print(f"Train: {len(train_pairs)}")
print(f"Val:   {len(val_pairs)}")