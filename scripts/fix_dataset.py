import os
import xml.etree.ElementTree as ET
from PIL import Image


if __name__ == "__main__":
    os.chdir("/home/bohdan/code/aircraft-classification/data/original-dataset/Annotations/Horizontal Bounding Boxes")
    for file in os.listdir():
        tree = ET.parse(file)
        root = tree.getroot()
        image = Image.open(f"/home/bohdan/code/aircraft-classification/data/original-dataset/JPEGImages/{os.path.basename(file).split('.')[0]}.jpg")
        img_width, img_height = image.size
        root.find("size/width").text = str(img_width)
        root.find("size/height").text = str(img_height)
        tree.write(file)
