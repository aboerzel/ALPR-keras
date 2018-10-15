# import the necessary packages
import cv2


class MeanPreprocessor:
    def __init__(self, mean):
        # store the Red, Green, and Blue channel averages across a
        # training set
        self.mean = mean

    def preprocess(self, image):
        # split the image into its respective Red, Green, and Blue
        # channels
        M = cv2.split(image.astype("float32"))

        # subtract the means for each channel
        M -= self.mean

        # merge the channels back together and return the image
        return cv2.merge(M)
