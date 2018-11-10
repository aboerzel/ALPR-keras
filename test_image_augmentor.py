import random

import h5py
import matplotlib.pyplot as plt
import numpy as np

from config import alpr_config as config
from license_plate_image_augmentor import LicensePlateImageAugmentor

batch_size = 6

trainData = h5py.File(config.TRAIN_HDF5)
images = np.array(trainData["images"])
labels = np.array(trainData["labels"])
trainData.close()

n = random.randint(0, len(images) - batch_size)
images = images[n:n + batch_size]
labels = labels[n:n + batch_size]

augmentor = LicensePlateImageAugmentor(config.IMAGE_WIDTH, config.IMAGE_HEIGHT, config.SUN397_HDF5)

cols = 2
rows = len(images) // cols

image_index = 0
fig, axarr = plt.subplots(rows, cols, figsize=(15, 50))
for r in range(rows):
    for c in range(cols):
        image = images[image_index]
        image = augmentor.generate_plate_image(image)
        axarr[r, c].title.set_text(labels[image_index])
        axarr[r, c].axis("off")
        axarr[r, c].imshow(image, cmap='gray')
        image_index += 1

plt.show()
