import os
from pprint import pprint
from src.utils import choose_model, choose_image


PATH_TO_SAVE = "/home/bohdan/code/projects/aircraft-classification/detection-results"


model, model_name = choose_model()
path_to_image = choose_image()

results = model.predict(source=path_to_image, conf=0.25, iou=0.5)

results[0].save(filename=os.path.join(PATH_TO_SAVE, f"{os.path.splitext(os.path.basename(path_to_image))[0]}-{model_name}.jpg"))

prediction_results = []
for result in results:
    boxes = result.boxes
    if boxes is not None:
        for i, box in enumerate(boxes):
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            coords = box.xyxy[0].tolist()
            x1, y1, x2, y2 = coords
            center = (round((x1+x2)/2), round((y1+y2)/2))
            prediction_results.append({"class_name": class_name,
                                       "confidence": confidence,
                                       "center_coordinates": center})
pprint(prediction_results)

results[0].show()