import argparse
import os

import cv2
import numpy as np
import progressbar
from imutils import paths
from sklearn.model_selection import train_test_split

from config import config
from pyimagesearch.io import HDF5DatasetWriter

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image_data", default=config.TRAIN_IMAGES, help="image data path")
ap.add_argument("-t", "--train_data", default=config.TRAIN_HDF5, help="train dataset hdf5-file")
ap.add_argument("-v", "--test_data", default=config.TEST_HDF5, help="test dataset hdf5-file")
ap.add_argument("-i", "--items", default=1000, type=int, help="max images")
args = vars(ap.parse_args())

trainPaths = list(paths.list_images(args['image_data']))
trainLabels = [p.split(os.path.sep)[-1].split(".")[0].split('#')[1] for p in trainPaths]

# perform stratified sampling from the training set to build the
# testing split from the training data
split = train_test_split(trainPaths, trainLabels, test_size=0.20, random_state=42)
(trainPaths, testPaths, trainLabels, testLabels) = split

# construct a list pairing the training, validation, and testing
# image paths along with their corresponding labels and output HDF5 files
datasets = [
    ("train", trainPaths, trainLabels, args['train_data']),
    ("test", testPaths, testLabels, args['test_data'])]

# original size of generated license plate images
IMAGE_WIDTH = 151
IMAGE_HEIGHT = 32

# loop over the images tuples
for (dType, paths, labels, outputPath) in datasets:
    # create HDF5 writer
    print("[INFO] building {}...".format(outputPath))
    writer = HDF5DatasetWriter((len(paths), IMAGE_HEIGHT, IMAGE_WIDTH), outputPath)

    # initialize the progress bar
    widgets = ["Building Dataset: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(paths), widgets=widgets)
    pbar.start()

    # loop over the image paths
    for (i, (path, label)) in enumerate(zip(paths, labels)):
        # load the image and process it
        # image = cv2.imread(path, cv2.IMREAD_COLOR)  # don't use imread because bug with utf-8 paths
        stream = open(path, "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        image = cv2.imdecode(numpyarray, cv2.IMREAD_GRAYSCALE)

        # check image size
        if not image.shape == (IMAGE_HEIGHT, IMAGE_WIDTH):
            print("image with wrong size: %s" % path)
            continue

        # check number length
        if len(label) > 10:
            print("image with wrong label: %s - %s" % (path, label))
            continue

        # add the image and label # to the HDF5 images
        writer.add([image], [label])
        pbar.update(i)

    # close the HDF5 writer
    pbar.finish()
    writer.close()
