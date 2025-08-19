import os
from PIL import Image, ImageEnhance, ImageFilter
from utils import add_gaussian_noise


if __name__ == "__main__":
    os.chdir("/home/bohdan/code/aircraft-classification")

    background = Image.open("data/artifitial-data/airbase-images-v1/images/1.png")
    overlay_img = Image.open("data/artifitial-data/plane-images/su-24/no-background/su24_1.png")
    overlay_img_original = overlay_img.copy()
    overlay_img_original = overlay_img_original.resize((100, 200))
    r, g, b, a = overlay_img_original.split()
    a = a.filter(ImageFilter.GaussianBlur(radius=3))
    # a = a.point(lambda p: p*0.85)

    # overlay_img = overlay_img.resize((20, 40))
    overlay_img_original = Image.merge("RGBA", (r, g, b, a))
    overlay_img = ImageEnhance.Color(overlay_img).enhance(0.3)
    overlay_img = ImageEnhance.Brightness(overlay_img).enhance(0.3)
    overlay_img = ImageEnhance.Contrast(overlay_img).enhance(0.6)
    overlay_img = ImageEnhance.Sharpness(overlay_img).enhance(0)

    background = ImageEnhance.Sharpness(background).enhance(1)
    # background = add_gaussian_noise(background, std=20)
    # background.paste(overlay_img, (441, 121), overlay_img)
    background.paste(overlay_img_original, (440, 120), overlay_img_original)

    background.show()