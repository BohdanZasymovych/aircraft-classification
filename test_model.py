from ultralytics import YOLO
import os


if __name__ == "__main__":
    os.chdir("/home/bohdan/code/aircraft-classification")

    model = YOLO("runs/detect/train-russian-100ep-artifitial-imgsz704-m/weights/best.pt")

    model.val(
        data="data/dataset-russian-planes-artifitial-v3/data.yaml",
        split="test",
        project="train-russian-100ep-artifitial-imgsz704-m",
        name="test-results",
        exist_ok=True
    )
