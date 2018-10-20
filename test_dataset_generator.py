import os
import imgaug as ia
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

from config import anpr_config as config
from pyimagesearch.io import HDF5DatasetGenerator
from pyimagesearch.preprocessing import SimplePreprocessor

ia.seed(1)

os.chdir(os.path.join("D:/development/cv/ANPR-keras"))

aug = ImageDataGenerator(rotation_range=18, zoom_range=0.15,
                         width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
                         horizontal_flip=True, fill_mode="nearest")

sp = SimplePreprocessor(config.IMAGE_WIDTH, config.IMAGE_HEIGHT)
trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, config.BATCH_SIZE, preprocessors=[sp], aug=aug)

inputs, outputs = trainGen.generator().__next__()
trainGen.close()

cols = 8
rows = len(inputs["data"]) // cols

f, axarr = plt.subplots(rows, cols, figsize=(15, 32))
for r in range(rows):
    for c in range(cols):
        image = inputs["data"][r + c].reshape(32, 160)
        axarr[r, c].axis("off")
        # axarr[r, c].imshow(image, cmap='Greys_r')
        axarr[r, c].imshow(image, cmap='gray')

plt.show()
