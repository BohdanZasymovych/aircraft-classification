import os
import shutil
from utils import clear_folder
from scripts.xml_parser import XMLParser

os.chdir("/home/bohdan/code/aircraft-classification/data")


with open("dataset/ImageSets/Main/train.txt", "r", encoding="utf-8") as file:
    TRAIN_IMAGES: set[int] = set(map(int, file.read().split("\n")))

with open("dataset/ImageSets/Main/test.txt", "r", encoding="utf-8") as file:
    VAL_IMAGES: set[int] = set(map(int, file.read().split("\n")))


def copy_images():
    clear_folder("yolov8-format-dataset/images/train")
    clear_folder("yolov8-format-dataset/images/val")

    for image in os.listdir("dataset/JPEGImages"):
        img_num = int(os.path.basename(image).split('.')[0])

        if img_num in TRAIN_IMAGES:
            shutil.copy(f"dataset/JPEGImages/{image}", "yolov8-format-dataset/images/train")
        elif img_num in VAL_IMAGES:
            shutil.copy(f"dataset/JPEGImages/{image}", "yolov8-format-dataset/images/val")
        else:
            raise ValueError("Image number are not present in any of the sets")

    print("Images copied")


def format_labels():
    clear_folder("yolov8-format-dataset/labels/train")
    clear_folder("yolov8-format-dataset/labels/val")

    Parser = XMLParser()

    for label_file in os.listdir("dataset/Annotations/Horizontal Bounding Boxes"):
        label_num = int(os.path.basename(label_file).split('.')[0])
        label_file = os.path.abspath(f"dataset/Annotations/Horizontal Bounding Boxes/{label_file}")

        if label_num in TRAIN_IMAGES:
            Parser.format_file(label_file, f"yolov8-format-dataset/labels/train/{label_num}.txt")
        elif label_num in VAL_IMAGES:
            Parser.format_file(label_file, f"yolov8-format-dataset/labels/val/{label_num}.txt")
        else:
            raise ValueError("Label number are not present in any of the sets")
    
    print("Labels created")
