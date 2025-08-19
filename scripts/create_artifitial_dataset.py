import os
from time import time
from artifitial_images_generator import ImageCreator
from utils import split_dataset_files

PATH_TO_REPOSITORY = "/home/bohdan/code/aircraft-classification/"
PATH_TO_ARBASE_IMAGES_FOLDER = "data/artifitial-data/airbase-images-v2"
PATH_TO_PLANE_IMAGES_FOLDER = "data/artifitial-data/plane-images"
PATH_TO_SAVE = "data/artifitial-data/artifitial-images-dataset-v3"


if __name__ == "__main__":
    t1 = time()
    os.chdir(PATH_TO_REPOSITORY)
    image_creator = ImageCreator(path_to_airbase_images_folder=PATH_TO_ARBASE_IMAGES_FOLDER,
                                path_to_plane_images_folder=PATH_TO_PLANE_IMAGES_FOLDER,
                                path_to_save=PATH_TO_SAVE,
                                start_index=4270)
    image_creator.generate_dataset(number_of_images=1730)
    # split_dataset_files(os.path.join(PATH_TO_SAVE, "labels"), os.path.join(PATH_TO_SAVE, "images"), "data/dataset-russian-planes-artifitial-v3", 0.80)
    t2 = time()
    print(t2-t1)