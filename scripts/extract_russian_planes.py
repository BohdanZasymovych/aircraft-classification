import os
import shutil
from xml_parser import XMLParser
from utils import clear_folder

SOVIET_RUSSIAN_PLANES_CLASSES = ["A1", "A6", "A12", "A17", "A19", "A20"]


def extract_soviet_russian_planes() -> None:
    clear_folder("russian-planes-yolov8-dataset/all-images")
    clear_folder("russian-planes-yolov8-dataset/all-labels")

    images: set[int] = set()

    for file in os.listdir("original-dataset/Annotations/Horizontal Bounding Boxes"):
        num = int(os.path.basename(file).split('.')[0])
        obj_classes = XMLParser.get_obj_classes(f"original-dataset/Annotations/Horizontal Bounding Boxes/{file}")
        if all(obj_class in SOVIET_RUSSIAN_PLANES_CLASSES for obj_class in obj_classes):
            images.add(num)
    
    print("Number of images containing soviet planes:", len(images))
    
    for file in images:
        shutil.copy(f"original-dataset/JPEGImages/{file}.jpg", "russian-planes-yolov8-dataset/all-images")
        parser = XMLParser()
        parser.format_file(f"original-dataset/Annotations/Horizontal Bounding Boxes/{file}.xml", f"russian-planes-yolov8-dataset/all-labels/{file}.txt")

if __name__ == "__main__":
    os.chdir("/home/bohdan/code/aircraft-classification/data")
    extract_soviet_russian_planes()
