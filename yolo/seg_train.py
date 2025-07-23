from ultralytics import YOLO
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Load a model
#model = YOLO("yolo12n-seg.yaml").load("yolo12n.pt")

model = YOLO("yolo12n-seg.yaml").load("runs/segment/train3/weights/best.pt")

# Train the model
results = model.train(data="coco_sportsball.yaml", epochs=100, imgsz=640, device=2)