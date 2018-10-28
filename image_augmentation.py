import h5py
import imutils
import numpy as np
import Augmentor
import cv2
import random
import matplotlib.pyplot as plt
from config import alpr_config as config

trainData = h5py.File(config.TRAIN_HDF5)
images = np.array(trainData["images"])
labels = np.array(trainData["labels"])
image = images[0]
label = labels[0]
trainData.close()

images = images[:100]
labels = labels[:100]

plate = images[0]
plate = imutils.resize(plate, width=(config.IMAGE_WIDTH-30))
plate = cv2.cvtColor(plate, cv2.COLOR_GRAY2RGBA)

# plt.axis("off")
# plt.imshow(plate, cmap='gray')
# plt.show()

bg = cv2.imread('D:/development/datasets/SUN397/c/campus/sun_akgyyhdnnpenxrwv.jpg')
# plate = cv2.imread('D:/development/datasets/alpr/images/AA-KC7866.png')
bg = cv2.cvtColor(bg, cv2.COLOR_RGB2BGRA)

OUTPUT_SHAPE = config.IMAGE_HEIGHT, config.IMAGE_WIDTH

x = random.randint(0, bg.shape[1] - OUTPUT_SHAPE[1])
y = random.randint(0, bg.shape[0] - OUTPUT_SHAPE[0])
bg = bg[y:y + OUTPUT_SHAPE[0], x:x + OUTPUT_SHAPE[1]]

# Now create a mask of logo and create its inverse mask also
img2gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)

# Take only region of logo from logo image.
plate = cv2.bitwise_and(plate, plate, mask=mask)
black_image = np.zeros(plate.shape, np.uint8)

# Put logo in ROI and modify the main image
plate = cv2.add(black_image, plate)

x = (bg.shape[0] - plate.shape[0]) // 2
y = (bg.shape[1] - plate.shape[1]) // 2
rows, cols, channels = plate.shape
bg[x:x + rows, y:y + cols] = plate

image = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
label = "Test"

p = Augmentor.Pipeline()

#p.zoom_random(probability=1, percentage_area=0.998)
p.skew_left_right(probability=1, magnitude=0.2)
p.skew_top_bottom(probability=1, magnitude=0.05)
# p.skew_tilt(probability=1, magnitude=0.2)
p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)

g = p.keras_generator_from_array([image], [label], batch_size=30)

images, labels = next(g)

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
