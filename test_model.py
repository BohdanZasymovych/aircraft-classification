from ultralytics import YOLO
import os


if __name__ == "__main__":
    os.chdir("/home/bohdan/code/aircraft-classification")

    model = YOLO("runs/detect/train-russian-100ep/weights/best.pt")

    model.val(
        data="data/russian-planes-yolov8-dataset/data.yml",
        split="test",
        project="runs/detect/train-russian-100ep",
        name="test-results",
        exist_ok=True
    )
