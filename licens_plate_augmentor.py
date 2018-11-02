import random

import Augmentor
import cv2
import imutils
import numpy as np


class LicensePlateAugmentor:
    def __init__(self, img_w, img_h):
        self.augmentor = Augmentor.Pipeline()

        self.augmentor.zoom_random(probability=0.8, percentage_area=0.998)
        self.augmentor.skew_top_bottom(probability=0.8, magnitude=0.05)
        self.augmentor.skew_left_right(probability=0.8, magnitude=0.2)
        # self.augmentor.skew_tilt(probability=1, magnitude=0.2)
        self.augmentor.rotate(probability=0.7, max_left_rotation=5, max_right_rotation=5)
        self.augmentor.gaussian_distortion(probability=1, grid_width=1, grid_height=1,
                                           magnitude=1, corner="bell", method="in",
                                           mex=0.5, mey=0.5, sdx=0.05, sdy=0.05)

        self.OUTPUT_SHAPE = img_h, img_w

    def __get_background_image__(self):
        bg = cv2.imread('D:/development/datasets/SUN397/c/campus/sun_akgyyhdnnpenxrwv.jpg')
        bg = cv2.cvtColor(bg, cv2.COLOR_RGB2BGRA)
        return bg

    def generate_bg(self, num_bg_images):
        found = False
        while not found:
            fname = "bgs/{:08d}.jpg".format(random.randint(0, num_bg_images - 1))
            bg = cv2.imread(fname, cv2.IMREAD_GRAYSCALE) / 255.
            if (bg.shape[1] >= self.OUTPUT_SHAPE[1] and
                    bg.shape[0] >= self.OUTPUT_SHAPE[0]):
                found = True

        x = random.randint(0, bg.shape[1] - self.OUTPUT_SHAPE[1])
        y = random.randint(0, bg.shape[0] - self.OUTPUT_SHAPE[0])
        bg = bg[y:y + self.OUTPUT_SHAPE[0], x:x + self.OUTPUT_SHAPE[1]]

        return bg

    def __get_random_background__(self):
        background = self.__get_background_image__()  # .copy()
        x = random.randint(0, background.shape[1] - self.OUTPUT_SHAPE[1])
        y = random.randint(0, background.shape[0] - self.OUTPUT_SHAPE[0])
        background = background[y:y + self.OUTPUT_SHAPE[0], x:x + self.OUTPUT_SHAPE[1]]
        return background

    def __augment_plate__(self, plate):
        # resize plate image to smaller width than output width
        plate = imutils.resize(plate, width=(self.OUTPUT_SHAPE[1] - 10))
        # convert to grayscale image with alpha channel for transparency
        plate = cv2.cvtColor(plate, cv2.COLOR_GRAY2RGBA)

        background = self.__get_random_background__()

        # place plate image centered on background
        x = (background.shape[0] - plate.shape[0]) // 2
        y = (background.shape[1] - plate.shape[1]) // 2
        rows, cols, channels = plate.shape
        background[x:x + rows, y:y + cols] = plate

        # convert to grayscale image without alpha channel
        image = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
        return image

    def generator(self, images, labels):
        plates = []
        for i, image in enumerate(images):
            plates.append(self.__augment_plate__(image))
        plates = np.asarray(plates)
        generator = self.augmentor.keras_generator_from_array(plates, labels, batch_size=len(plates))
        return next(generator)
