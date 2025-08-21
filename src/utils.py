import os
import sys
from ultralytics import YOLO


PATH_TO_MODELS = "/home/bohdan/code/aircraft-classification/runs/detect"
PATH_TO_IMAGES = "/home/bohdan/code/aircraft-classification/test-images"


def get_path() -> str:
    """
    Function asks user to enter path until user enters valid one
    """
    while True:
        path = input(">>> ")
        if os.path.exists(path):
            return path
        print("path doesn't exist, enter valid path")


def get_models() -> list[str]:
    """
    Function returns list of all trained model names.
    Name of a model coinsides with its folder. 
    return: list of names of models. 
    """
    models = list(sorted(filter(lambda x: x.startswith("train-"), os.listdir(PATH_TO_MODELS))))
    return models


def get_images() -> list[str]:
    """
    Function returns list of names of all images in folder test-images
    """
    images = list(sorted(os.listdir(PATH_TO_IMAGES)))
    return images


def choose_model():
    """
    Function asks user to chose model.
    return: tuple with model object and name of a model.
    """
    models = get_models()
    print("Enter number to chose a model")
    print("\n".join(f"{i}: {model}" for i, model in enumerate(models, start=1)))
    model_name = models[int(input(">>> ").strip())-1]
    model = YOLO(os.path.join(PATH_TO_MODELS, model_name, "weights/best.pt"))
    return model, model_name


def choose_image() -> str:
    """
    Function asks user to chose image to process.
    return: str, full path to an image.
    """
    images = get_images()
    print("Enter number to chose an image")
    print("0: Enter full path")
    print("\n".join(f"{i}: {img}" for i, img in enumerate(images, start=1)))
    image_num = int(input(">>> ").strip())

    if image_num == 0:
        image_path == input("Enter full path to an image\n>>> ")
        if not os.path.exists():
            print("Path does not exist")
            sys.exit()
    else:
        image_name = images[image_num-1]
        image_path = os.path.join(PATH_TO_IMAGES, image_name)

    return image_path
