import h5py
import numpy as np
import random
import matplotlib.pyplot as plt
from config import alpr_config as config
from licens_plate_augmentor import LicensePlateAugmentor

batch_size = 32

trainData = h5py.File(config.TRAIN_HDF5)
images = np.array(trainData["images"])
labels = np.array(trainData["labels"])
trainData.close()

n = random.randint(0, len(images) - batch_size)
images = images[n:n + batch_size]
labels = labels[n:n + batch_size]

aug = LicensePlateAugmentor(config.IMAGE_WIDTH, config.IMAGE_HEIGHT)
images, labels = aug.generator(images, labels)

cols = 6
rows = len(images) // cols

image_index = 0
f, axarr = plt.subplots(rows, cols, figsize=(15, 50))
for r in range(rows):
    for c in range(cols):
        image = images[image_index].reshape(images[image_index].shape[:2])
        axarr[r, c].axis("off")
        # axarr[r, c].imshow(image, cmap='Greys_r')
        axarr[r, c].imshow(image, cmap='gray')

        image_index += 1

plt.show()
