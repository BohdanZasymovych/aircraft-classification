import os
import random
import json
from time import time
from PIL import Image, ImageEnhance, ImageFilter
from utils import add_gaussian_noise


CLASSES_NAMES_MAPPING_PATH = "data/artifitial-data/plane-images-v3/class_names_to_id_mapping.json"

with open(CLASSES_NAMES_MAPPING_PATH, "r", encoding="utf-8") as file:
    CLASSES_NAMES_MAPPING = json.load(file)


class PlaneImage:
    def __init__(self, image: Image, width: float, plane_id: str):
        self.image = image.copy()
        self.class_index = plane_id
        self.__meter_width = width
        self.__pixel_size: tuple = None

    @property
    def pixel_width(self) -> int:
        """Property returns pixel width of the plane image."""
        return self.__pixel_size[0]

    @property
    def pixel_height(self) -> int:
        """Property returns pixel height of the plane image."""
        return self.__pixel_size[1]

    def calculate_pixel_size(self, airbase_pixel_width: int, airbase_meter_width: float):
        """
        Function calculates plane size in pixels of the airbase image to preserve real scale
        and sets tuple with width and height to __pixel_size attribute.
        """
        airbase_pixel_size = airbase_pixel_width/airbase_meter_width

        image_pixel_width, image_pixel_height = self.image.size
        height_to_width_plane_ratio = image_pixel_height/image_pixel_width

        plane_pixel_width = round(self.__meter_width*airbase_pixel_size)
        plane_pixel_height = round(plane_pixel_width*height_to_width_plane_ratio)

        self.__pixel_size = (plane_pixel_width, plane_pixel_height)
    
    def get_shadow_image(self, shadow_parameters: dict) -> Image:
        """
        Function takes shadow parameters, makes small variation to them, applies this parameters to an image to make shade image.
        return: PIL.Image object of the shadow.
        """
        # shadow_saturation = random.uniform(0.95, 1.05) * shadow_parameters["saturation"]
        # shadow_brightness = random.uniform(0.95, 1.05) * shadow_parameters["brightness"]
        # shadow_contrast = random.uniform(0.95, 1.05) * shadow_parameters["contrast"]
        # shadow_sharpness = random.uniform(0.95, 1.05) * shadow_parameters["sharpness"]
        # shadow_blur = random.uniform(0.95, 1.05) * shadow_parameters["blur"]
        # shadow_transparency = random.uniform(0.95, 1.05) * shadow_parameters["transparency"]

        r, g, b, a = self.image.split()
        a = a.point(lambda p: p*shadow_parameters["transparency"])
        shadow = Image.merge("RGBA", (r, g, b, a))

        shadow = ImageEnhance.Color(shadow).enhance(shadow_parameters["saturation"])
        shadow = ImageEnhance.Brightness(shadow).enhance(shadow_parameters["brightness"])
        shadow = ImageEnhance.Contrast(shadow).enhance(shadow_parameters["contrast"])
        shadow = ImageEnhance.Sharpness(shadow).enhance(shadow_parameters["sharpness"])
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=shadow_parameters["blur"]))

        shadow = shadow.resize(self.__pixel_size)

        return shadow
    
    def get_plane_image(self) -> Image:
        """
        Function makes small variation to an original image
        return: PIL.Image object.
        """
        image_saturation = random.uniform(0.8, 1.2)
        image_brightness = random.uniform(0.8, 1.2)
        image_contrast = random.uniform(0.8, 1.2)
        image_sharpness = random.uniform(0.4, 1.6)

        image = self.image

        r, g, b, a = image.split()
        a = a.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 3))) # Makes edges of plane blured and not too clear 
        image = Image.merge("RGBA", (r, g, b, a))

        image = ImageEnhance.Color(image).enhance(image_saturation)
        image = ImageEnhance.Brightness(image).enhance(image_brightness)
        image = ImageEnhance.Contrast(image).enhance(image_contrast)
        image = ImageEnhance.Sharpness(image).enhance(image_sharpness)

        image = image.resize(self.__pixel_size)

        return image


class AirbaseImage:
    def __init__(self, airbase_image: str, airbase_width: float, spawn_points: list, planes: list):
        self.image: Image = airbase_image.copy()
        self.labels: list[str] = None
        self.__pixel_size: tuple = self.image.size
        self.__airbase_width: float = airbase_width

        self.__spawn_points: list[tuple[int, int, str]] = spawn_points
        self.__modify_spawn_points()

        self.__planes: list[PlaneImage] = planes
        self.__calculate_plane_sizes()

        self.__shadow_parameters: dict = None
        self.__generate_shadow_parameters()

    def __calculate_plane_sizes(self):
        """
        Function resizes all planes to the size that corresponds to the airbase image size to preserve proportions.
        """
        airbase_pixel_width = self.__pixel_size[0]
        airbase_meter_width = self.__airbase_width
        for plane in self.__planes:
            plane.calculate_pixel_size(airbase_pixel_width, airbase_meter_width)

    def __modify_spawn_points(self):
        """
        Function slightly varies spawn points location coordinates and removes plane type from point data.
        """
        for i, spawn_point in enumerate(self.__spawn_points):
            x, y, _ = spawn_point
            x += random.randint(-3, 3)
            y += random.randint(-3, 3)
            self.__spawn_points[i] = (x, y)

    def __generate_shadow_parameters(self):
        """
        Function randomly generates shadow parameters.
        """
        self.__shadow_parameters = {
            "saturation": random.uniform(0.05, 1),
            "brightness": random.uniform(0.1, 0.7),
            "contrast": random.uniform(0.1, 0.6),
            "sharpness": random.uniform(0, 0.7),
            "blur": random.uniform(2, 10),
            "transparency": random.uniform(0.55, 0.95),
            "shadow_offset": (random.randint(-6, 6), random.randint(-6, 6))
        }

    def __generate_plane_label(self, x_center: int, y_center: int, plane_image: PlaneImage) -> str:
        """
        Function generates label for one plane placed on the airbase.
        return: string with label in YOLOv8 format.
        """
        class_index = plane_image.class_index
        x_center /= self.__pixel_size[0]
        y_center /= self.__pixel_size[1]
        width = plane_image.pixel_width / self.__pixel_size[0]
        height = plane_image.pixel_height / self.__pixel_size[1]

        return f"{class_index} {x_center} {y_center} {width} {height}"

    def generate_labels(self) -> list[str]:
        """
        Function generates labels for all planes placed on the airbase.
        return: list of strings with labels in YOLOv8 format.
        """
        labels = []
        for spawn_point, plane_image in zip(self.__spawn_points, self.__planes):
            x_center, y_center = spawn_point
            label = self.__generate_plane_label(x_center, y_center, plane_image)
            labels.append(label)

        return labels

    def __place_shadow(self, spawn_point: tuple[int, int], plane_image: PlaneImage):
        """
        Function places shadow of the plane on the airbase image.
        :param spawn_point: tuple containing x, y.
        :param plane_image: PlaneImage object.
        :param shadow_parameters: dict containing shadow parameters.
        """
        shadow = plane_image.get_shadow_image(self.__shadow_parameters)
        x, y = spawn_point
        x -= plane_image.pixel_width // 2
        y -= plane_image.pixel_height // 2
        x_offset, y_offset = self.__shadow_parameters["shadow_offset"]
        self.image.paste(shadow, (x + x_offset, y + y_offset), shadow)
    
    def __place_shadows(self):
        """
        Function places shadows of all planes on the airbase image.
        """
        for spawn_point, plane_image in zip(self.__spawn_points, self.__planes):
            self.__place_shadow(spawn_point, plane_image)
    
    def __place_plane(self, spawn_point: tuple[int, int], plane_image: PlaneImage) -> str:
        """
        Function places plane on the airbase image and generates label in YOLOv8 format for it.
        :param spawn_point: tuple containing x, y coordinates.
        :param plane_image: PlaneImage object.
        :return: string with label in YOLOv8 format.
        """
        x, y = spawn_point
        label = self.__generate_plane_label(x, y, plane_image)

        plane_image_pixel_width = plane_image.pixel_width
        plane_image_pixel_height = plane_image.pixel_height
        x -= plane_image_pixel_width // 2
        y -= plane_image_pixel_height // 2
        plane_image = plane_image.get_plane_image()
        self.image.paste(plane_image, (x, y), plane_image)
    
        return label

    def __place_planes(self) -> list[str]:
        """
        Function places all planes on the airbase image and generate labels for them.
        return: list of strings with labels in YOLOv8 format.
        """
        labels = []
        for spawn_point, plane_image in zip(self.__spawn_points, self.__planes):
            label = self.__place_plane(spawn_point, plane_image)
            labels.append(label)
        
        return labels
    
    @staticmethod
    def __add_interference(image):
        """
        Function randomly blurs or sharpens an image and with 0.5 probability adds gaussian noise
        with 0 mean and standart deviation being random value from 0 to 20.
        """
        if random.random() < 0.5:
            image_sharpness = 0.1 + 0.7*random.random()
        else:
            image_sharpness = 1.2 + 0.7*random.random()

        image = ImageEnhance.Sharpness(image).enhance(image_sharpness)
        if random.random() > 0.5:
            image = add_gaussian_noise(image, std=random.uniform(2, 20))
        
        return image
    
    def create_image(self) -> list[str]:
        """
        Function creates an image of the airbase with planes and shadows and sets list of labels for the planes to labels attribute.
        Method add_interference is applied to the image with 0.5 probability.
        """
        self.__place_shadows()
        labels = self.__place_planes()
        self.labels = labels

        self.image = self.__add_interference(self.image)
    
    def save_image(self, path_to_save: str, image_number: int) -> None:
        """
        Function saves image and labels to the specified paths.
        :param path_to_save_image: path to save the image.
        :param path_to_save_label: path to save the labels.
        """
        path_to_save_image = os.path.join(path_to_save, "images", f"{image_number}.png")
        path_to_save_label = os.path.join(path_to_save, "labels", f"{image_number}.txt")

        self.image.save(path_to_save_image)
        print(f"Saved image to {path_to_save_image}")
        with open(path_to_save_label, "w", encoding="utf-8") as file:
            file.write("\n".join(self.labels))


class ImageCreator:
    """
    Class used to create artifitial image of planes on the airbase
    with create_image method and save this image.
    """
    SMALL_PLANES = {"su24", "su27_su35", "su30_su34"}

    def __init__(self, path_to_airbase_images_folder: str, path_to_plane_images_folder: str, path_to_save: str, start_index: int=0):
        self.__path_to_airbase_images_folder: str = path_to_airbase_images_folder
        self.__path_to_plane_images_folder: str = path_to_plane_images_folder
        self.__airbase_image: AirbaseImage = None
        self.__path_to_save: str = path_to_save
        self.__image_number: int = start_index

        path_to_save_images: str = os.path.join(path_to_save, "images")
        if not os.path.exists(path_to_save_images):
            os.makedirs(path_to_save_images, exist_ok=True)

        path_to_save_labels: str = os.path.join(path_to_save, "labels")
        if not os.path.exists(path_to_save_labels):
            os.makedirs(path_to_save_labels, exist_ok=True)

        self.__airbases_images: list = []
        self.__airbases_labels: list = []
        self.__read_airbases_data()

        self.__planes_images: dict = {}
        self.__planes_labels: dict = {}
        self.__read_planes_data()
        self.__prerotate_planes()

        self.__small_planes_keys: list = []
        self.__all_planes_keys: list = []
        self.__setup_plane_selection_lists()

    def __reset(self):
        self.__airbase_image: AirbaseImage = None
        self.__image_number += 1

    def create_image(self):
        """
        Function creates image and saves it to the specified location.
        """
        airbase_image_name = self.__choose_airbase_image()
        airbase_image, (airbase_width, plane_spawn_points) = airbase_image_name

        plane_spawn_points = self.__choose_points(plane_spawn_points)
        plane_images = self.__choose_plane_images(plane_spawn_points)

        self.__airbase_image = AirbaseImage(airbase_image, airbase_width, plane_spawn_points, plane_images)
        self.__airbase_image.create_image()
        self.__airbase_image.save_image(self.__path_to_save, self.__image_number)

        self.__reset()

    def generate_dataset(self, number_of_images: int):
        """
        Function generates specified number of artifitial images of planes on the airbase
        and saves them to the specified location.
        :param number_of_images: int value of number of images to generate.
        """
        for _ in range(number_of_images):
            self.create_image()

    def __prerotate_planes(self):
        """
        Function creates rotated versions of each plane image in the __planes_images dictionary to save time during image generation.
        It rotates each plane image by 0, 45, 90, 135, 180, 225, 270, and 315 degrees.
        The rotated images are stored in a list for each plane name in the __planes_images dictionary.
        """
        ANGLES = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
        for plane_name, plane_image in self.__planes_images.items():
            self.__planes_images[plane_name] = []
            for angle in ANGLES:
                rotated_image = plane_image.rotate(angle, expand=True)
                self.__planes_images[plane_name].append(rotated_image)

    def __setup_plane_selection_lists(self):
        """
        Function sets up lists of plane types for selection.
        It creates a list of small planes and a list of all planes.
        """
        self.__all_planes_keys = list(self.__planes_labels.keys())
        self.__small_planes_keys = [key for key in self.__planes_labels.keys() if key[:-2] in self.SMALL_PLANES]

    def __read_airbase_label(self, airbase_image_name: str) -> tuple[int, list[tuple]]:
        """
        Function reads airbase image label.
        return: tuple containing int value of airbase width in meters
        and list of tuple containing list of spawn points for aircrafts
        """
        def process_point_line(line: str) -> tuple[int, int, str]:
            """
            Function converts string line to tuple
            return: x - int coordinate, y - int coordinate, plane type - str
            """
            line = line.strip().split(",")
            return int(line[0]), int(line[1]), line[2]

        img_num = os.path.splitext(airbase_image_name)[0]
        path_to_label = os.path.join(self.__path_to_airbase_images_folder, "labels", f"{img_num}.txt")

        with open(path_to_label, "r", encoding="utf-8") as file:
            width = int(file.readline().strip().split(",")[0])
            spawn_points = tuple(map(process_point_line, file.readlines()))
        
        return width, spawn_points
    
    def __read_airbases_data(self):
        path_to_airbase_images = os.path.join(self.__path_to_airbase_images_folder, "images")

        sorted_airbase_images = sorted(os.listdir(path_to_airbase_images), key= lambda x: int(os.path.splitext(x)[0]))
        for image_file in sorted_airbase_images:
            airbase_image_name = image_file
            path_to_image = os.path.join(path_to_airbase_images, airbase_image_name)
            airbase_image = Image.open(path_to_image)
            width, spawn_points = self.__read_airbase_label(airbase_image_name)

            self.__airbases_images.append(airbase_image)
            self.__airbases_labels.append((width, spawn_points))

    def __read_planes_data(self):
        """
        Function reads plane images and labels from the file and adds them to the __palne_images __planes_labels dictionary.
        """
        for folder in os.listdir(self.__path_to_plane_images_folder):
            if not os.path.isdir(os.path.join(self.__path_to_plane_images_folder, folder)):
                continue

            path_to_labels = os.path.join(self.__path_to_plane_images_folder, folder, "size-labels")
            path_to_images = os.path.join(self.__path_to_plane_images_folder, folder, "no-background")

            for image_file in os.listdir(path_to_images):
                key = os.path.splitext(image_file)[0]
                value = Image.open(os.path.join(path_to_images, image_file))
                self.__planes_images[key] = value

            for label_file in os.listdir(path_to_labels):
                key = os.path.splitext(label_file)[0]
                with open(os.path.join(path_to_labels, label_file), "r", encoding="utf-8") as file:
                    value = float(file.read().strip())

                self.__planes_labels[key] = value

    def __choose_airbase_image(self) -> str:
        """
        Function chooses image of an airbase and returns its name .
        """
        airbase_image_id = random.randint(0, len(self.__airbases_images)-1)
        airbase_image = self.__airbases_images[airbase_image_id]
        airbase_label = self.__airbases_labels[airbase_image_id]
        return airbase_image, airbase_label

    def __choose_points(self, points: tuple) -> list:
        """
        Function chooses random number of points from given list and returns them. 
        """
        n = random.randint(1, len(points))
        chosen_points = random.sample(points, n)
        return chosen_points

    def __choose_plane_image(self, plane_type: str) -> PlaneImage:
        """
        Function chooses plane image from given filder according to a type (s-small, b-small or big)
        return: PlaneImage object
        """
        match plane_type:
            case "s":
                plane_file_name = random.choice(self.__small_planes_keys)
                plane_image = random.choice(self.__planes_images[plane_file_name])
                plane_width_label = self.__planes_labels[plane_file_name]
                plane_id = CLASSES_NAMES_MAPPING[plane_file_name[:plane_file_name.rfind("_")]]
            case "b":
                plane_file_name = random.choice(self.__all_planes_keys)
                plane_image = random.choice(self.__planes_images[plane_file_name])
                plane_width_label = self.__planes_labels[plane_file_name]
                plane_id = CLASSES_NAMES_MAPPING[plane_file_name[:plane_file_name.rfind("_")]]
            case _:
                raise ValueError("Incorrect plane type")

        return PlaneImage(plane_image, plane_width_label, plane_id)
    
    def __choose_plane_images(self, points: list):
        """
        Function chooses planes to place on each point and sets list of them to the __planes attribute.
        """
        planes = []
        for _, _, plane_type in points:
            planes.append(self.__choose_plane_image(plane_type))

        return planes


if __name__ == "__main__":
    os.chdir("/home/bohdan/code/aircraft-classification")

    time_start = time()
    image_creator = ImageCreator(path_to_plane_images_folder="data/artifitial-data/plane-images",
                                 path_to_airbase_images_folder="data/artifitial-data/airbase-images",
                                 path_to_save="data/artifitial-data/artifitial-images-dataset-v2",
                                 start_index=0)
    image_creator.generate_dataset(number_of_images=6000)
    time_end = time()
    print(f"Generated images in {time_end - time_start:.2f} seconds.")
