import os
import imgaug as ia
import matplotlib.pyplot as plt

from config import alpr_config as config
from licence_plate_dataset_generator import LicensePlateDatasetGenerator
from pyimagesearch.preprocessing import SimplePreprocessor

ia.seed(1)

os.chdir(os.path.join(config.PROJECT_ROOT_PATH))

sp = SimplePreprocessor(config.IMAGE_WIDTH, config.IMAGE_HEIGHT)
trainGen = LicensePlateDatasetGenerator(config.TRAIN_HDF5, config.IMAGE_WIDTH, config.IMAGE_HEIGHT, config.POOL_SIZE,
                                        config.MAX_TEXT_LEN, config.BATCH_SIZE, preprocessors=[sp])

inputs, outputs = next(trainGen.generator())
trainGen.close()

cols = 6
rows = len(inputs["input"]) // cols

image_index = 0
f, axarr = plt.subplots(rows, cols, figsize=(10, 32))
for r in range(rows):
    for c in range(cols):
        image = inputs["input"][image_index].T.reshape(config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
        axarr[r, c].axis("off")
        axarr[r, c].imshow(image, cmap='Greys_r')
        # axarr[r, c].imshow(image, cmap='gray')
        image_index += 1

plt.show()
