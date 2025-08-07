import xml.etree.ElementTree as ET
from dataclasses import dataclass


@dataclass
class LabeledObject:
    class_id: int
    center_x: int
    center_y: int
    width: int
    height: int

    def __str__(self):
        return f"{self.class_id} {self.center_x} {self.center_y} {self.width} {self.height}"

class XMLParser:
    def __init__(self):
        self.__refresh()

    def __refresh(self):
        self.__img_width: int = None
        self.__img_height: int = None
        self.__objects: list[LabeledObject] = []

    def __parse_file(self, filepath: str):
        self.__root = ET.parse(filepath).getroot()
        self.__img_width = int(self.__root.findtext("size/width"))
        self.__img_height = int(self.__root.findtext("size/height")) 

    def __normalize_data(self, xmin: int, ymin: int, xmax: int, ymax: int) -> tuple[int, int]:
        width = (xmax-xmin)/self.__img_width
        height = (ymax-ymin)/self.__img_height
        return width, height

    def __find_center(self, xmin: int, ymin: int, xmax: int, ymax: int) -> tuple[int, int]:
        x = (xmax+xmin)/2/self.__img_width
        y = (ymax+ymin)/2/self.__img_height
        return x, y
    
    def __parse_objects(self):
        for obj in self.__root.findall("object"):
            class_id = int(obj.findtext("name")[1:])-1
            xmin = int(obj.findtext("bndbox/xmin"))
            ymin = int(obj.findtext("bndbox/ymin"))
            xmax = int(obj.findtext("bndbox/xmax"))
            ymax = int(obj.findtext("bndbox/ymax"))

            width, height = self.__normalize_data(xmin, ymin, xmax, ymax)
            center_x, center_y = self.__find_center(xmin, ymin, xmax, ymax)
            self.__objects.append(LabeledObject(class_id, center_x, center_y, width, height))
    
    def __process_file(self, filepath: str):
        self.__parse_file(filepath)
        self.__parse_objects()
    
    @classmethod
    def get_obj_classes(cls, filepath: str) -> set[str]:
        """
        Function returns set of all classes of objects present in the image.
        """
        root = ET.parse(filepath).getroot()
        obj_clases = {obj.findtext("name") for obj in root.findall("object")}
        return obj_clases


    def format_file(self, file_to_format: str, file_to_write: str) -> None:
        """
        Function parses .xml file, formats its data to YOLOv8 format
        and writes formated data to a given file.
        """
        self.__process_file(file_to_format)

        with open(file_to_write, "w", encoding="utf-8") as file:
            file.write("\n".join(str(obj) for obj in self.__objects))

        self.__refresh()

