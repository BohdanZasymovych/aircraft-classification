from ultralytics import YOLO
from src.utils import choose_model, get_path
import os
import yaml

PATH_TO_REPOSITORY = "/home/bohdan/code/aircraft-classification"


if __name__ == "__main__":
    os.chdir(PATH_TO_REPOSITORY)

    model, model_name = choose_model()
    path_to_model = os.path.join("runs", "detect", model_name, "weights", "best.pt")
    path_to_args = os.path.join("runs", "detect", model_name, "args.yaml")

    path_to_data = None
    with open(path_to_args, "r", encoding="utf-8") as file:
        path_to_data = yaml.safe_load(file)["data"]
    
    with open(path_to_data, "r", encoding="utf-8") as file:
        content = yaml.safe_load(file)
        if "test" not in content:
            raise ValueError("Model doesn't have test set")

    path_to_save = os.path.split(path_to_data)[0]
    results_folder_name = f"test-results-{model_name}"

    model.val(
        data=path_to_data,
        split="test",
        project=path_to_save,
        name=results_folder_name,
        exist_ok=True
    )
