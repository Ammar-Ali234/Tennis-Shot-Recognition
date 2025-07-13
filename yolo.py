from ultralytics import YOLO

model = YOLO("yolo11n-pose.pt")  # or yolov8s-pose.pt

# Inference on a video
results = model("test.mp4", save=True, conf=0.1)
