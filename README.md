# aircraft-classification

## Summary

This repository contains code, scripts, and experiments for detecting and classifying Russian military aircraft in satellite images of airbases. It also includes scripts and data for creating artificial datasets.

## Project structure

- ```main.py``` — script to recognize planes in an image using a selectable model

- ```test_model.py``` — script to test model performance if a training set is available

- ```data/```
  - ```artificial-data/``` — assets used to build synthetic datasets
    - ```airbase-images.zip``` — archive with airbase images and size labels
    - ```plane-images.zip``` — archive with all types of planes and size labels
    - ```airbase-images/```
      - ```images/``` — folder with raw images of airbases (images do not contain aircraft)
      - ```labels/``` — folder with labels for airbase images containing their real size (in meters) and points where planes can be placed
  - ```datasets/``` — folder containing dataset folders
    - ```dataset-folder```
      - ```test-results``` — folder with results of tests on the test set
      - ```class_names_mapping.json``` — mapping of class id to its name
      - ```config.yaml``` — config with training and augmentation parameters used to train the corresponding model
      - ```data.yaml``` — data file for training YOLO model

- ```runs/detect``` — folder with trained models

- ```test-images/``` — folder containing images of airbases for testing

- ```detection-results/``` — folder where processed images with classified planes are saved

- ```scripts/``` — dataset creation, preprocessing, and utilities
  - ```artificial_image_generator.py``` — classes used to create artificial satellite images
  - ```set_plane_locations.py``` — module with class used to set spawn locations for planes on images of empty airbases for generating artificial satellite images
  - ```utils.py``` — utilities used in other scripts 
  - ```format_yolov8.py``` — module with functions to convert the initial dataset with .xml labels to YOLO format
  - ```xml_parser.py``` — module with class used to convert the initial dataset with .xml labels to YOLO format

- ```src/```
  - ```utils.py``` — shared helper functions used by ```main.py``` and ```test_model.py```

## Installation

Recommended: create a Python venv and install dependencies listed in `requirements.txt` with the following commands (for Ubuntu):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage
```bash
cd aircraft-classification
source .venv/bin/activate
```

To launch the main script, where you can choose an image from `test-images` to detect planes or enter your own path, and select a model to use:
```bash
python3 main.py
```

To launch the script that tests model performance (if a test set is available). You can choose which model to test:
```bash
python3 test_model.py
```

## Dataset
The initial dataset with real images was taken from Kaggle: [Military Aircraft Recognition dataset](https://www.kaggle.com/datasets/khlaifiabilel/military-aircraft-recognition-dataset)  
This dataset contains 3,842 images, 20 types, and 22,341 instances annotated with horizontal and oriented bounding boxes.

The `plane-images.zip` archive from `data/artificial-datasets/` contains images and labels for the following types of aircraft:
- A-50
- Il-76
- L-39
- MiG-29
- MiG-31
- Su-24
- Su-25
- Su-27/35
- Su-30/34
- Tu-160
- Tu-22
- Tu-95

Su-27 and Su-35, as well as Su-30 and Su-34, were combined into two classes instead of four due to their small visual differences and the complexity of distinguishing them for a model.

## Artificial dataset generation
There is a lack of real satellite images of military planes of different classes, so I created scripts to generate artificial imagery.  
I used public satellite imagery from Google Earth and created PNG images of planes of different types with transparent backgrounds. Airbase images were also obtained from Google Earth.

### Aircraft placement
To create an artificial image of an airbase, planes are placed on randomly chosen points from a set defined for each airbase using the ```scripts/set_plane_locations.py``` module. There are two types of points: small (only planes defined as small can be placed) and big (all planes can be placed). Aircraft are placed with small random noise in position to make images more distinct.

### Shadows
Shadows are added to make images more realistic. A random shift is chosen for all shadows on an image, and all shadows are placed with this shift relative to the plane. The shadow of a plane is obtained by lowering the saturation and brightness of its image and applying blur to the alpha channel to make the shadow more realistic.

### Augmentation
For each airbase image, new parameters for the plane and shadow images (saturation, brightness, contrast, blur, sharpness) are randomly generated and applied. After placing planes, Gaussian noise and blur are added to the final image. Augmentation increases dataset diversity and helps prevent model overfitting.

## Models
Project includes 5 trained models:

- **train-all-20cls-60ep-s** — recognizes 20 classes of aircraft (Russian and NATO) from the initial Kaggle dataset. Trained for 60 epochs, YOLOv8s.

- **train-russian-11cls-76ep-artificial-imgsz800-m** — recognizes 11 classes of Russian aircraft (A-50, Il-76, MiG-29, MiG-31, Su-24, Su-25, Su-27/35, Su-30/34, Tu-22, Tu-95, Tu-160). Trained for 76 epochs, image size 800, YOLOv8m.

- **train-russian-6cls-60ep-s** — recognizes 6 classes of Russian aircraft (Su-24, Su-27/35, Su-30/34, Tu-22, Tu-95, Tu-160). Trained on 1,167 real images from the initial dataset containing only Russian planes. Trained for 60 epochs, YOLOv8s.

- **train-russian-6cls-100ep-artificial-imgsz800-m** — recognizes 6 classes of Russian aircraft (Su-24, Su-27/35, Su-30/34, Tu-22, Tu-95, Tu-160). Trained on 4,000 artificially generated images. Trained for 100 epochs, image size 800, YOLOv8m.

- **train-russian-6cls-100ep-combined-imgsz800-m** — recognizes 6 classes of Russian aircraft (Su-24, Su-27/35, Su-30/34, Tu-22, Tu-95, Tu-160). Trained on 4,176 images (1,176 real, 3,000 artificial). Trained for 100 epochs, image size 800, YOLOv8m.

## Detection results
### Below are examples of aircraft detection on Saky and Engels airbases for each model

### **train-all-20cls-60ep-s**:
<table><tr>
  <td><img src="detection-results/saky_airbase-train-all-20cls-60ep-s.jpg" width="400"></td>
  <td><img src="detection-results/engels_airbase5-train-all-20cls-60ep-s.jpg" width="400"></td>
</tr></table>

### **train-russian-11cls-76ep-artificial-imgsz800-m**:
<table><tr>
  <td><img src="detection-results/saky_airbase-train-russian-11cls-76ep-artifitial-imgsz800-m.jpg" width="400"></td>
  <td><img src="detection-results/engels_airbase5-train-russian-11cls-76ep-artifitial-imgsz800-m.jpg" width="400"></td>
</tr></table>

### **train-russian-6cls-100ep-artificial-imgsz800-m**:
<table><tr>
  <td><img src="detection-results/saky_airbase-train-russian-6cls-100ep-artifitial-imgsz800-m.jpg" width="400"></td>
  <td><img src="detection-results/engels_airbase5-train-russian-6cls-100ep-artifitial-imgsz800-m.jpg" width="400"></td>
</tr></table>

### **train-russian-6cls-60ep-s**:
<table><tr>
  <td><img src="detection-results/saky_airbase-train-russian-6cls-60ep-s.jpg" width="400"></td>
  <td><img src="detection-results/engels_airbase5-train-russian-6cls-60ep-s.jpg" width="400"></td>
</tr></table>

### **train-russian-6cls-100ep-combined-imgsz800-m**:
<table><tr>
  <td><img src="detection-results/saky_airbase-train-russian-6cls-100ep-combined-imgsz800-m.jpg" width="400"></td>
  <td><img src="detection-results/engels_airbase5-train-russian-6cls-100ep-combined-imgsz800-m.jpg" width="400"></td>
</tr></table>

## Performance comparison
Performance comparison is made only for the 3 models that classify 6 classes of planes, for a fair evaluation. All models were tested on the same set containing 83 real images with instances of all classes.

### **train-russian-6cls-100ep-artificial-imgsz800-m**:

<table><tr>
  <td><img src="data/datasets/russian-planes-6cls-artifitial/test-results-train-russian-6cls-100ep-artifitial-imgsz800-m/BoxF1_curve.png" width="400"></td>
  <td><img src="data/datasets/russian-planes-6cls-artifitial/test-results-train-russian-6cls-100ep-artifitial-imgsz800-m/BoxP_curve.png" width="400"></td>
</tr></table>

<table><tr>
  <td><img src="data/datasets/russian-planes-6cls-artifitial/test-results-train-russian-6cls-100ep-artifitial-imgsz800-m/BoxPR_curve.png" width="400"></td>
  <td><img src="data/datasets/russian-planes-6cls-artifitial/test-results-train-russian-6cls-100ep-artifitial-imgsz800-m/BoxR_curve.png" width="400"></td>
</tr></table>

![Normalized confusion matrix](data/datasets/russian-planes-6cls-artifitial/test-results-train-russian-6cls-100ep-artifitial-imgsz800-m/confusion_matrix_normalized.png)

### **train-russian-6cls-60ep-s**:
<table><tr>
  <td><img src="data/datasets/russian-planes-6cls/test-results-train-russian-6cls-60ep-s/BoxF1_curve.png" width="400"></td>
  <td><img src="data/datasets/russian-planes-6cls/test-results-train-russian-6cls-60ep-s/BoxP_curve.png" width="400"></td>
</tr></table>

<table><tr>
  <td><img src="data/datasets/russian-planes-6cls/test-results-train-russian-6cls-60ep-s/BoxPR_curve.png" width="400"></td>
  <td><img src="data/datasets/russian-planes-6cls/test-results-train-russian-6cls-60ep-s/BoxR_curve.png" width="400"></td>
</tr></table>

![Normalized confusion matrix](data/datasets/russian-planes-6cls/test-results-train-russian-6cls-60ep-s/confusion_matrix_normalized.png)

### **train-russian-6cls-100ep-combined-imgsz800-m**:

<table><tr>
  <td><img src="data/datasets/russian-planes-6cls-combined/test-results-train-russian-6cls-100ep-combined-imgsz800-m/BoxF1_curve.png" width="400"></td>
  <td><img src="data/datasets/russian-planes-6cls-combined/test-results-train-russian-6cls-100ep-combined-imgsz800-m/BoxP_curve.png" width="400"></td>
</tr></table>

<table><tr>
  <td><img src="data/datasets/russian-planes-6cls-combined/test-results-train-russian-6cls-100ep-combined-imgsz800-m/BoxPR_curve.png" width="400"></td>
  <td><img src="data/datasets/russian-planes-6cls-combined/test-results-train-russian-6cls-100ep-combined-imgsz800-m/BoxR_curve.png" width="400"></td>
</tr></table>

![Normalized confusion matrix](data/datasets/russian-planes-6cls-combined/test-results-train-russian-6cls-100ep-combined-imgsz800-m/confusion_matrix_normalized.png)

## Sources
Kaggle, Military Aircraft Recognition dataset: https://www.kaggle.com/datasets/khlaifiabilel/military-aircraft-recognition-