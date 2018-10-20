# import the necessary packages
from keras.preprocessing.image import img_to_array
import numpy as np

class ImageToArrayPreprocessor:
    def __init__(self, dataFormat=None):
        # store the image data format
        self.dataFormat = dataFormat

    def preprocess(self, image):
       # image = image.astype(np.float32)
      #  image /= 255
        # apply the Keras utility function that correctly rearranges the dimensions of the image
        return img_to_array(image, data_format=self.dataFormat)
