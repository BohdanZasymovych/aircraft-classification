import cv2
import os

def show_yolov8_annotation(image_path: str, labels_folder: str, class_names=None):
    """
    Show image with YOLOv8 boxes and class IDs.

    Parameters:
        image_path (str): Path to image file.
        labels_folder (str): Path to folder with YOLOv8 txt annotations.
        class_names (list[str] | None): Optional list of class names. 
                                        If None, class IDs will be shown.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    h, w = img.shape[:2]

    # annotation file assumed to have same basename as image
    label_path = os.path.join(
        labels_folder, os.path.splitext(os.path.basename(image_path))[0] + ".txt"
    )
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Annotation not found: {label_path}")

    with open(label_path, "r") as f:
        for line in f:
            cls, x_c, y_c, bw, bh = map(float, line.strip().split())
            cls = int(cls)

            # convert from YOLOv8 normalized format to pixel coords
            x1 = int((x_c - bw / 2) * w)
            y1 = int((y_c - bh / 2) * h)
            x2 = int((x_c + bw / 2) * w)
            y2 = int((y_c + bh / 2) * h)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = class_names[cls] if class_names else str(cls)
            cv2.putText(
                img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2
            )

    cv2.imshow("YOLOv8 Annotations", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    show_yolov8_annotation("/home/bohdan/code/aircraft-classification/data/dataset-russian-planes-artifitial-v2/images/train/0.png",
                           "/home/bohdan/code/aircraft-classification/data/dataset-russian-planes-artifitial-v2/labels/train/")