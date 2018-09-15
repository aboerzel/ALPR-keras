import cv2
import imutils
import random


class RandomRotatePreprocessor:
    def __init__(self, width, height, min_angle, max_angle, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.min = min_angle
        self.max = max_angle
        self.inter = inter

    def preprocess(self, image):
        angle = random.randint(self.min, self.max)
        rotated = imutils.rotate_bound(image, angle)
        return cv2.resize(rotated, (self.width, self.height), interpolation=self.inter)
