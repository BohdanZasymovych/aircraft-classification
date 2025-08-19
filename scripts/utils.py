import os
import shutil
from typing import Iterable
import json
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def add_gaussian_noise(img: Image.Image, mean=0, std=25) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(mean, std, arr.shape)
    noisy_arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_arr)


def clear_folder(path_to_folder: str) -> None:
    """
    Funcion deletes all files from a folder
    """
    for file in os.listdir(path_to_folder):
        filepath = os.path.join(path_to_folder, file)
        os.remove(filepath)


def copy_files(path_to_folder: str, file_names: Iterable[str], destination_folder: str) -> None:
    """
    Function copies all files from the folder which names are in the list to the destination foolder.
    Names and extentions of the files are preserved.
    path_to_folder: path to the folder from which files will be copied.
    file_names: Iterable containing all file names (without extension) which will be copied.
    destination_folder: path to the folder to which files will be copied. If folder does not exist it will be created.
    """
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder, exist_ok=True)

    for file in os.listdir(path_to_folder):
        if os.path.splitext(os.path.basename(file))[0] in file_names:
            shutil.copy(os.path.join(path_to_folder, file), destination_folder)


def split_dataset_train_val(dataset: list[int], train_size: float) -> tuple[list[int], list[int]]:
    """
    Function splits dataset represented as list to train and validation sets randomly.
    train_size parameter deffiens proportion in which dataset will be splited,
    it has to be two non-negative numbers adding up to one.
    return: tuple containing two sorted lists with training and validation sets. 
    """
    if not 0 <= train_size <= 1:
        raise ValueError("train_size parameter has to be non-negative number less or equal to one")
    
    train, val = train_test_split(dataset, train_size=train_size, random_state=15)

    return train, val


def split_dataset_files(path_to_labels: str, path_to_images: str, destination_path: str, train_size: float) -> None:
    dataset = [os.path.splitext(os.path.basename(file))[0] for file in os.listdir(path_to_labels)]
    train, val = split_dataset_train_val(dataset, train_size=train_size)
    copy_files(path_to_folder=path_to_labels, file_names=train, destination_folder=os.path.join(destination_path, "labels", "train"))
    copy_files(path_to_folder=path_to_labels, file_names=val, destination_folder=os.path.join(destination_path, "labels", "val"))
    copy_files(path_to_folder=path_to_images, file_names=train, destination_folder=os.path.join(destination_path, "images", "train"))
    copy_files(path_to_folder=path_to_images, file_names=val, destination_folder=os.path.join(destination_path, "images", "val"))


def plot_classes_presence(path_to_folder: str, path_to_class_mapping: str, plot: bool=True, returns: bool=True) -> None | dict:
    """
    Function plots histogram showing presence of diffrent classes in the set and returns number of .
    path_to_folder: path to folder where labels for classes is stored,
    all files has to be .txt files foemated for YOLOv8 model.
    path_to_class_mapping: path to .json file mapping class id to class name.
    """

    with open(path_to_class_mapping, "r", encoding="utf-8") as file:
        class_mapping: dict = json.load(file)

    os.chdir(path_to_folder)

    classes_presence: dict[str: int] = {class_name: 0 for class_name in class_mapping.values()}

    for file in os.listdir(path_to_folder):
        with open(file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                if line:
                    classes_presence[class_mapping[line[0]]] += 1
    
    if plot:
        classes = classes_presence.keys()
        values = classes_presence.values()

        plt.bar(classes, values)

        plt.xlabel("Plane classes")
        plt.ylabel("Instances")
        plt.title("Classes presence in the dataset")
        plt.show()
    
    if returns:
        return classes_presence

    return None


def clamp(value, min_val, max_val):
    """Function clamps the value to be within the specified range."""
    return max(min_val, min(value, max_val))
