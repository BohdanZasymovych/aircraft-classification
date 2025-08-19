import os
from utils import split_dataset_files


if __name__ == "__main__":
    os.chdir("/home/bohdan/code/aircraft-classification/data")
    split_dataset_files("artifitial-data/artifitial-images-dataset-v2/labels", "artifitial-data/artifitial-images-dataset-v2/images", "dataset-russian-planes-artifitial-v2", 0.80)