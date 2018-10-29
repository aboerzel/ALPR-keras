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

n = random.randint(0, 15000)
images = images[n:n + 30]
labels = labels[n:n + 30]

OUTPUT_SHAPE = config.IMAGE_HEIGHT, config.IMAGE_WIDTH

bg = cv2.imread('D:/development/datasets/SUN397/c/campus/sun_akgyyhdnnpenxrwv.jpg')
bg = cv2.cvtColor(bg, cv2.COLOR_RGB2BGRA)


def preprocess(image):
    plate = imutils.resize(image, width=(config.IMAGE_WIDTH - 10))
    plate = cv2.cvtColor(plate, cv2.COLOR_GRAY2RGBA)

    x = random.randint(0, bg.shape[1] - OUTPUT_SHAPE[1])
    y = random.randint(0, bg.shape[0] - OUTPUT_SHAPE[0])
    background = bg.copy()
    background = background[y:y + OUTPUT_SHAPE[0], x:x + OUTPUT_SHAPE[1]]

    # create a mask of plate and create its inverse mask also
    plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(plate_gray, 10, 255, cv2.THRESH_BINARY)

    # Take only region of plate from plate image.
    plate = cv2.bitwise_and(plate, plate, mask=mask)
    black_image = np.zeros(plate.shape, np.uint8)

    # Put plate in ROI and modify the output image
    plate = cv2.add(black_image, plate)

    x = (background.shape[0] - plate.shape[0]) // 2
    y = (background.shape[1] - plate.shape[1]) // 2
    rows, cols, channels = plate.shape
    background[x:x + rows, y:y + cols] = plate

    image = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

    return image


plates = []
for i, image in enumerate(images):
    plates.append(preprocess(images[i]))
plates = np.asarray(plates)

p = Augmentor.Pipeline()

p.zoom_random(probability=0.8, percentage_area=0.998)
p.skew_top_bottom(probability=0.8, magnitude=0.05)
p.skew_left_right(probability=0.8, magnitude=0.2)
# p.skew_tilt(probability=1, magnitude=0.2)
p.rotate(probability=0.7, max_left_rotation=5, max_right_rotation=5)

p.gaussian_distortion(probability=1, grid_width=4, grid_height=4, magnitude=1, corner="bell", method="in",
                      mex=0.5, mey=0.5, sdx=0.05, sdy=0.05)

g = p.keras_generator_from_array(plates, labels, batch_size=30)

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
