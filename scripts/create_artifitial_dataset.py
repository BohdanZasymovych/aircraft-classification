import os
from time import time
from artifitial_image_generator import ImageCreator
from utils import split_dataset_files

PATH_TO_REPOSITORY = "/home/bohdan/code/aircraft-classification/"
PATH_TO_ARBASE_IMAGES_FOLDER = "data/artifitial-data/airbase-images"
PATH_TO_PLANE_IMAGES_FOLDER = "data/artifitial-data/plane-images-v3"
PATH_TO_SAVE = "data/artifitial-data/combined-images-dataset"


if __name__ == "__main__":
    t1 = time()
    os.chdir(PATH_TO_REPOSITORY)
    # image_creator = ImageCreator(path_to_airbase_images_folder=PATH_TO_ARBASE_IMAGES_FOLDER,
    #                             path_to_plane_images_folder=PATH_TO_PLANE_IMAGES_FOLDER,
    #                             path_to_save=PATH_TO_SAVE,
    #                             start_index=4000)
    # image_creator.generate_dataset(number_of_images=3000)
    split_dataset_files(os.path.join(PATH_TO_SAVE, "labels"), os.path.join(PATH_TO_SAVE, "images"), "data/dataset-russian-planes-combined", train_size=0.85)
    t2 = time()
    print(t2-t1)