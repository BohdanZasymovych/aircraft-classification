import os
from src.utils import choose_model, choose_image


PATH_TO_SAVE = "/home/bohdan/code/aircraft-classification/test-results"


model, model_name = choose_model()
path_to_image = choose_image()

results = model(path_to_image)

results[0].show()
# results[0].save(filename=os.path.join(PATH_TO_SAVE, f"{os.path.splitext(os.path.basename(path_to_image))[0]}-{model_name}.jpg"))
