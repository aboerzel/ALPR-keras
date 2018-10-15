import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np

from config import anpr_config as config

db = h5py.File(config.TRAIN_HDF5)
images = db["images"]
labels = db["labels"]

image = images[0]
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.imwrite("D:/development/cv/datasets/anpr/test.png", image)
#img = cv2.imread("D:/development/cv/datasets/anpr/train/ED-KS5784.png")
#image = cv2.resize(image, (config.IMAGE_WIDTH, config.IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)

plt.axis("off")
#plt.imshow(image, cmap='gray')
#plt.imshow(image)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.show()
# plt.axis("off")
# plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
# #plt.imshow(image1, cmap='gray')
# plt.show()
