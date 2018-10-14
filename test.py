from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from pyimagesearch.datasets.LicensePlateDatasetLoader import LicensePlateDatasetLoader
from pyimagesearch.preprocessing import RandomRotatePreprocessor
from pyimagesearch.preprocessing import RandomGaussianNoisePreprocessor
import matplotlib.pyplot as plt
import numpy as np

img_w = 169
img_h = 32
batch_size = 32
pool_size = 2

preprocessors = [
    RandomRotatePreprocessor(-5, 5, img_w, img_h),
    RandomGaussianNoisePreprocessor(15)]

train_dataset = LicensePlateDatasetLoader(img_w, img_h, pool_size, batch_size)
train_dataset.load("data/train")

data = np.array(train_dataset.samples);

(trainX, testX, trainY, testY) = train_test_split(data[0], data[1],
                                                  test_size=0.25, random_state=42)

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

data = aug.flow(trainX, trainY, batch_size=32)

plt.axis("off")
# plt.imshow(data[0], cmap='gray')
plt.imshow(data[0])
plt.show()
