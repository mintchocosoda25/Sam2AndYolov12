from ultralytics import YOLO
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Load a model
#model = YOLO("yolo12n-seg.yaml").load("runs/segment/train4/weights/best.pt")
model = YOLO("runs/segment/train/weights/best.pt")

results = model("./1.mp4", task="segment", save=True ,classes=[0])