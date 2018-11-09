import matplotlib.pyplot as plt
import numpy as np
from config import alpr_config as config
from licence_plate_dataset_generator import LicensePlateDatasetGenerator
from pyimagesearch.preprocessing import RandomGaussianNoisePreprocessor

trainGen = LicensePlateDatasetGenerator(config.TRAIN_HDF5, config.IMAGE_WIDTH, config.IMAGE_HEIGHT,
                                        config.POOL_SIZE, config.MAX_TEXT_LEN, 20)

inputs, outputs = next(trainGen.generator())
trainGen.close()

cols = 2
rows = len(inputs["input"]) // cols

# rgnp = RandomGaussianNoisePreprocessor(1)

for i, img in enumerate(inputs["input"]):
    img = img.T.reshape(config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
    #     #img = cv2.GaussianBlur(img, (3, 3), 5)
    #     #img = cv2.medianBlur(img, 3, 5)
    #     #img = rgnp.preprocess(img)
    #
    #     row, col = img.shape
    #     mean = 0.0
    #     var = 0.02
    #     sigma = var ** 0.5
    #     #gauss = np.array(img.shape)
    #     gauss = np.random.normal(mean, sigma, (row, col))
    #     gauss = gauss.reshape(row, col)
    #
    #
    plt.title(inputs["labels"][i])
    plt.axis("off")
    plt.imshow(img, cmap='Greys_r')
    # img = img + gauss
    plt.show()

image_index = 0
f, axarr = plt.subplots(rows, cols, figsize=(15, 10))
for r in range(rows):
    for c in range(cols):
        image = inputs["input"][image_index].T.reshape(config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
        axarr[r, c].axis("off")
        # axarr[r, c].imshow(image, cmap='Greys_r')
        axarr[r, c].imshow(image, cmap='gray')
        image_index += 1

plt.show()
