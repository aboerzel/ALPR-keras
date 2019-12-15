import math
import random

import cv2
import numpy as np


class LicensePlateImageAugmentor:
    def __init__(self, img_w, img_h, background_images):

        self.OUTPUT_SHAPE = img_h, img_w
        self.background_images, _ = background_images

    def __get_random_background_image__(self):
        index = random.randint(0, len(self.background_images) - 1)
        return self.background_images[index]

    def __generate_background_image__(self):
        background = self.__get_random_background_image__()
        x = random.randint(0, background.shape[1] - self.OUTPUT_SHAPE[1])
        y = random.randint(0, background.shape[0] - self.OUTPUT_SHAPE[0])
        background = background[y:y + self.OUTPUT_SHAPE[0], x:x + self.OUTPUT_SHAPE[1]]
        return background

    @staticmethod
    def __euler_to_mat__(yaw, pitch, roll):
        # Rotate clockwise about the Y-axis
        c, s = math.cos(yaw), math.sin(yaw)
        M = np.matrix([[c, 0., s],
                       [0., 1., 0.],
                       [-s, 0., c]])

        # Rotate clockwise about the X-axis
        c, s = math.cos(pitch), math.sin(pitch)
        M = np.matrix([[1., 0., 0.],
                       [0., c, -s],
                       [0., s, c]]) * M

        # Rotate clockwise about the Z-axis
        c, s = math.cos(roll), math.sin(roll)
        M = np.matrix([[c, -s, 0.],
                       [s, c, 0.],
                       [0., 0., 1.]]) * M

        return M

    def __make_affine_transform__(self, from_shape, to_shape,
                                  min_scale, max_scale,
                                  scale_variation=1.0,
                                  rotation_variation=1.0,
                                  translation_variation=1.0):

        from_size = np.array([[from_shape[1], from_shape[0]]]).T
        to_size = np.array([[to_shape[1], to_shape[0]]]).T

        scale = random.uniform((min_scale + max_scale) * 0.5 -
                               (max_scale - min_scale) * 0.5 * scale_variation,
                               (min_scale + max_scale) * 0.5 +
                               (max_scale - min_scale) * 0.5 * scale_variation)
        if scale > max_scale or scale < min_scale:
            raise Exception("out_of_bounds")

        roll = random.uniform(-0.3, 0.3) * rotation_variation
        pitch = random.uniform(-0.2, 0.2) * rotation_variation
        yaw = random.uniform(-1.2, 1.2) * rotation_variation

        # Compute a bounding box on the skewed input image (`from_shape`).
        M = self.__euler_to_mat__(yaw, pitch, roll)[:2, :2]
        h, w = from_shape
        corners = np.matrix([[-w, +w, -w, +w],
                             [-h, -h, +h, +h]]) * 0.5
        skewed_size = np.array(np.max(M * corners, axis=1) -
                               np.min(M * corners, axis=1))

        # Set the scale as large as possible such that the skewed and scaled shape
        # is less than or equal to the desired ratio in either dimension.
        scale *= np.min(to_size / skewed_size)

        # Set the translation such that the skewed and scaled image falls within
        # the output shape's bounds.
        trans = (np.random.random((2, 1)) - 0.5) * translation_variation
        trans = ((2.0 * trans) ** 5.0) / 2.0
        if np.any(trans < -0.5) or np.any(trans > 0.5):
            raise Exception("out_of_bounds")

        trans = (to_size - skewed_size * scale) * trans

        center_to = to_size / 2.
        center_from = from_size / 2.

        M = self.__euler_to_mat__(yaw, pitch, roll)[:2, :2]
        M *= scale
        M = np.hstack([M, trans + center_to - M * center_from])

        return M

    @staticmethod
    def __gaussian_noise__(image, sigma=1):
        mean = 0.0
        gauss = np.random.normal(mean, sigma, image.shape)
        image = image + gauss
        return image

    @staticmethod
    def normalize_image(image):
        # normalize image data between 0 and 1
        image = (image - image.min()) / (image.max() - image.min())
        return image

    def generate_plate_image(self, plate):
        bi = self.__generate_background_image__()

        M = self.__make_affine_transform__(
            from_shape=plate.shape,
            to_shape=bi.shape,
            min_scale=0.9,
            max_scale=1.0,
            rotation_variation=0.8,
            scale_variation=1.0,
            translation_variation=0.0)

        plate_mask = np.ones(plate.shape)
        plate = cv2.warpAffine(plate, M, (bi.shape[1], bi.shape[0]))
        plate_mask = cv2.warpAffine(plate_mask, M, (bi.shape[1], bi.shape[0]))

        out = plate * plate_mask + bi * (1.0 - plate_mask)

        out = self.__gaussian_noise__(out, 15)
        out = self.normalize_image(out)
        return out
