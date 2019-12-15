import matplotlib.pyplot as plt

from config import alpr_config as config
from license_plate_image_augmentor import LicensePlateImageAugmentor
from pyimagesearch.io import Hdf5DatasetLoader

batch_size = 6

loader = Hdf5DatasetLoader()
background_images = loader.load(config.SUN397_HDF5, shuffle=True, max_items=10000)
images, labels = loader.load(config.TRAIN_HDF5, shuffle=True, max_items=batch_size)

augmentor = LicensePlateImageAugmentor(config.IMAGE_WIDTH, config.IMAGE_HEIGHT, background_images)

cols = 2
rows = len(images) // cols

image_index = 0
fig, axarr = plt.subplots(rows, cols, figsize=(15, 50))
for r in range(rows):
    for c in range(cols):
        image = images[image_index]
        image = augmentor.generate_plate_image(image)
        axarr[r, c].axis("off")
        axarr[r, c].title.set_text(labels[image_index])
        axarr[r, c].imshow(image, cmap='gray')
        image_index += 1

plt.show()
