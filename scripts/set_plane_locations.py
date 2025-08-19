import cv2
import os


class ImageWindow:
    def __init__(self, image_name: str):
        self.__points: list[tuple[int, int, str]] = []

        image_number: str = os.path.splitext(os.path.basename(image_name))[0]
        self.__label_path = os.path.join("labels", f"{image_number}.txt")

        self.__image = cv2.imread(os.path.join("images", image_name))
        cv2.namedWindow("Image", cv2.WINDOW_GUI_NORMAL)
        cv2.setWindowProperty("Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Image", self.__image)
        cv2.setMouseCallback("Image", self.__mouse_button_click_event, self)

    @staticmethod
    def __mouse_button_click_event(event, x, y, flags, param):
        self = param

        if event == cv2.EVENT_LBUTTONDOWN:
            self.__points.append((str(x), str(y), "s"))
            cv2.circle(self.__image, center=(x, y), radius=3, color=(255, 0, 0), thickness=-1)
            cv2.imshow("Image", self.__image)

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.__points.append((str(x), str(y), "b"))
            cv2.circle(self.__image, center=(x, y), radius=3, color=(0, 0, 255), thickness=-1)
            cv2.imshow("Image", self.__image)
    

    def __save_points(self):
        with open(self.__label_path, "a", encoding="utf-8") as file:
            file.write("\n".join(",".join(point) for point in self.__points))

    def __close(self):
        cv2.destroyAllWindows()

    def run(self):
        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                self.__save_points()
                print(f"Saved {len(self.__points)} points to {self.__label_path}")
                self.__close()
                break
            elif key == ord('q'):
                self.__close()
                break


class MarkerApp:
    def __init__(self, path_to_airbase_images_directory: str):
        os.chdir(path_to_airbase_images_directory)

    def run(self):
        for file in sorted(os.listdir("images"), key=lambda x: int(os.path.splitext(x)[0])):
            image_window = ImageWindow(file)
            image_window.run()


if __name__ == "__main__":
    marker_app = MarkerApp("data/artifitial-data/airbase-images-v2")
    marker_app.run()
