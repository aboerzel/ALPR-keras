import argparse
import math
import random

import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from keras.optimizers import SGD
from sklearn.metrics import accuracy_score

from config import alpr_config as config
from label_codec import LabelCodec
from license_plate_image_augmentor import LicensePlateImageAugmentor
from pyimagesearch.io import Hdf5DatasetLoader

ap = argparse.ArgumentParser()
ap.add_argument("-", "--items", default=1000, type=int, help="number of images to evaluate")
args = vars(ap.parse_args())


def preprocess(image, width, height):
    # make image smaller than than the output size (along width) keeping the aspect ratio
    image = imutils.resize(image, width=(width - 10))

    # determine the padding values for the width and height to obtain the target dimensions
    padW = int((width - image.shape[1]) / 2.0)
    padH = int((height - image.shape[0]) / 2.0)

    # pad the image then apply one more resizing to handle any rounding issues
    image = cv2.copyMakeBorder(image, padH, padH, padW, padW, cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))

    # normalize data
    image = image.astype("float") / 255.

    # return the pre-processed image
    return image


def show_image(image, label, pred_text):
    plt.axis("off")
    plt.title('True: {}\nPred: {}'.format(label, pred_text), loc='left')
    plt.imshow(image, cmap='gray')
    plt.show()


sess = tf.Session()
K.set_session(sess)

model = load_model(config.MODEL_PATH, compile=False)

optimizer = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)

net_inp = model.get_layer(name='input').input
net_out = model.get_layer(name='softmax').output

# load test data from dataset
loader = Hdf5DatasetLoader()
images, labels = loader.load(config.TEST_HDF5, shuffle=True, max_items=args["items"])

augmentor = LicensePlateImageAugmentor(config.IMAGE_WIDTH, config.IMAGE_HEIGHT, config.SUN397_HDF5)

# predictions for accuracy measurement
y_true = np.full(len(images), True, dtype=bool)
y_pred = np.full(len(images), False, dtype=bool)

for i, (image, label) in enumerate(zip(images, labels)):
    image = augmentor.generate_plate_image(image)
    imput_image = np.expand_dims(image.T, -1)
    X_data = [imput_image]
    net_out_value = sess.run(net_out, feed_dict={net_inp: X_data})
    pred_text = LabelCodec.decode_number_from_output(net_out_value)

    show_image(image, label, pred_text)

    y_pred[i] = pred_text == label
    print('%6s - Predicted: %-10s - True: %-10s - %s' % (i + 1, pred_text, label, y_pred[i]))

print()
accuracy = accuracy_score(y_true, y_pred)
print("Accuracy: %s" % accuracy)
