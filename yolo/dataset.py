from ultralytics import YOLO
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Load a model
#model = YOLO("runs/segment/train4/weights/best.pt")

model = YOLO("yolo12n-seg.yaml").load("runs/segment/train/weights/best.pt")


# Train the model
results = model.train(data="open-images-v7.yaml", epochs=100, imgsz=640)