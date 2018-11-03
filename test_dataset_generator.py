import matplotlib.pyplot as plt

from config import alpr_config as config
from licence_plate_dataset_generator import LicensePlateDatasetGenerator

trainGen = LicensePlateDatasetGenerator(config.TRAIN_HDF5, config.IMAGE_WIDTH, config.IMAGE_HEIGHT,
                                        config.POOL_SIZE, config.MAX_TEXT_LEN, config.BATCH_SIZE)

inputs, outputs = next(trainGen.generator())
trainGen.close()

cols = 8
rows = len(inputs["input"]) // cols

# for img in inputs["input"]:
#     img = img.T.reshape(config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
#     plt.axis("off")
#     plt.imshow(img, cmap='Greys_r')
#     plt.show()

image_index = 0
f, axarr = plt.subplots(rows, cols, figsize=(15, 10))
for r in range(rows):
    for c in range(cols):
        image = inputs["input"][image_index].T.reshape(config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
        axarr[r, c].axis("off")
        axarr[r, c].imshow(image, cmap='Greys_r')
        # axarr[r, c].imshow(image, cmap='gray')
        image_index += 1

plt.show()
