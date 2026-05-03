from ultralytics import YOLO

# Load a YOLO26n PyTorch model
model = YOLO("/home/ahmed/Other/capstone/train/runs/detect/yolo26n_sz320/weights/best.pt")

# Export the model to NCNN format
# exported_model_path = model.export(format="ncnn", half=True, )  # creates 'yolo26n_ncnn_model'
exported_model_path = model.export(format="ncnn", half=True, imgsz=320)  # smaller size

print(exported_model_path)

# Load the exported NCNN model
# ncnn_model = YOLO("best_ncnn_model")

# Run inference
# results = ncnn_model("https://ultralytics.com/images/bus.jpg")