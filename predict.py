import argparse
import random
import math

import cv2
import imutils
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from keras.optimizers import SGD

from config import alpr_config as config
from label_codec import LabelCodec

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default="../../datasets/alpr/test/TS-U9235.png", type=str, help="image to predict")
ap.add_argument("-l", "--label", default="TS-U9235", type=str, help="true label")
args = vars(ap.parse_args())


def load_image(filepath):
    stream = open(filepath, "rb")
    bytes = bytearray(stream.read())
    numpyarray = np.asarray(bytes, dtype=np.uint8)
    image = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def __euler_to_mat__(yaw, pitch, roll):
    # Rotate clockwise about the Y-axis
    c, s = math.cos(yaw), math.sin(yaw)
    M = np.matrix([[c, 0., s],
                   [0., 1., 0.],
                   [-s, 0., c]])

    # Rotate clockwise about the X-axis
    c, s = math.cos(pitch), math.sin(pitch)
    M = np.matrix([[1., 0., 0.],
                   [0., c, -s],
                   [0., s, c]]) * M

    # Rotate clockwise about the Z-axis
    c, s = math.cos(roll), math.sin(roll)
    M = np.matrix([[c, -s, 0.],
                   [s, c, 0.],
                   [0., 0., 1.]]) * M

    return M


def __make_affine_transform__(from_shape, to_shape,
                              min_scale, max_scale,
                              scale_variation=1.0,
                              rotation_variation=1.0,
                              translation_variation=1.0):
    from_size = np.array([[from_shape[1], from_shape[0]]]).T
    to_size = np.array([[to_shape[1], to_shape[0]]]).T

    scale = random.uniform((min_scale + max_scale) * 0.5 -
                           (max_scale - min_scale) * 0.5 * scale_variation,
                           (min_scale + max_scale) * 0.5 +
                           (max_scale - min_scale) * 0.5 * scale_variation)
    if scale > max_scale or scale < min_scale:
        raise Exception("out_of_bounds")

    roll = random.uniform(-0.3, 0.3) * rotation_variation
    pitch = random.uniform(-0.2, 0.2) * rotation_variation
    yaw = random.uniform(-1.2, 1.2) * rotation_variation

    # Compute a bounding box on the skewed input image (`from_shape`).
    M = __euler_to_mat__(yaw, pitch, roll)[:2, :2]
    h, w = from_shape
    corners = np.matrix([[-w, +w, -w, +w],
                         [-h, -h, +h, +h]]) * 0.5
    skewed_size = np.array(np.max(M * corners, axis=1) -
                           np.min(M * corners, axis=1))

    # Set the scale as large as possible such that the skewed and scaled shape
    # is less than or equal to the desired ratio in either dimension.
    scale *= np.min(to_size / skewed_size)

    # Set the translation such that the skewed and scaled image falls within
    # the output shape's bounds.
    trans = (np.random.random((2, 1)) - 0.5) * translation_variation
    trans = ((2.0 * trans) ** 5.0) / 2.0
    if np.any(trans < -0.5) or np.any(trans > 0.5):
        raise Exception("out_of_bounds")

    trans = (to_size - skewed_size * scale) * trans

    center_to = to_size / 2.
    center_from = from_size / 2.

    M = __euler_to_mat__(yaw, pitch, roll)[:2, :2]
    M *= scale
    M = np.hstack([M, trans + center_to - M * center_from])

    return M


def generate_plate_image(plate):
    bi = np.zeros((config.IMAGE_HEIGHT, config.IMAGE_WIDTH), np.uint8)
    bi[:] = 255

    M = __make_affine_transform__(
        from_shape=plate.shape,
        to_shape=bi.shape,
        min_scale=0.9,
        max_scale=1.0,
        rotation_variation=0.8,
        scale_variation=1.0,
        translation_variation=0.0)

    plate_mask = np.ones(plate.shape)
    plate = cv2.warpAffine(plate, M, (bi.shape[1], bi.shape[0]))
    plate_mask = cv2.warpAffine(plate_mask, M, (bi.shape[1], bi.shape[0]))

    out = plate * plate_mask + bi * (1.0 - plate_mask)
    out = out / 255.
    return out  # __gaussian_noise__(out, random.uniform(0.0, 0.0005))


def preprocess(image, width, height):
    if image.shape[0] > height:
        # resize image to target height keeping the aspect ratio
        image = imutils.resize(image, height=height)

    if image.shape[1] > width:
        # resize image to target width keeping the aspect ratio
        image = imutils.resize(image, width=width)

    image = generate_plate_image(image)

    # # determine the padding values for the height to obtain the target dimensions
    # padH = int((config.IMAGE_HEIGHT - image.shape[0]) / 2.0)
    # padW = int((config.IMAGE_WIDTH - image.shape[1]) / 2.0)
    #
    # # pad the image then apply one more resizing to handle any rounding issues
    # image = cv2.copyMakeBorder(image, padH, padH, padW, padW, cv2.BORDER_REPLICATE)
    # image = cv2.resize(image, (width, height))
    #
    # # image = cv2.fastNlMeansDenoising(image, None, 20, 20, 10)
    # # image = cv2.equalizeHist(image)
    # kernel = np.ones((2, 2), np.uint8)
    # # image = cv2.erode(image, kernel, iterations=4)
    # # image = cv2.dilate(image, kernel, iterations=3)
    # image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    # kernel = np.ones((1, 1), np.uint8)
    # image = cv2.dilate(image, kernel, iterations=3)

    # normalize data
    #image = image.astype(np.float32) / 255.
    # return the pre-processed image
    return image


def show_image(image):
    plt.axis("off")
    plt.title(label)
    plt.imshow(image, cmap='gray')
    plt.show()


img_filepath = args["image"]

if not args["label"] == "":
    label = img_filepath.split('/')[-1].split('.')[0]
else:
    label = args["label"]

img_filepath = "D:/development/datasets/alpr/val/SÜW-E1557.png"
label = "SÜW-E1557"

sess = tf.Session()
K.set_session(sess)

model = load_model(config.MODEL_PATH, compile=False)

optimizer = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)

net_inp = model.get_layer(name='input').input
net_out = model.get_layer(name='softmax').output

image = load_image(img_filepath)
image = preprocess(image, config.IMAGE_WIDTH, config.IMAGE_HEIGHT)
show_image(image)

image = np.expand_dims(image.T, -1)

# net_out_value = model.predict(
#     {"input": data, "labels": labels, "input_length": input_length, "label_length": label_length})

net_out_value = sess.run(net_out, feed_dict={net_inp: [image]})
pred_text = LabelCodec.decode_number_from_output(net_out_value)
fig = plt.figure(figsize=(10, 10))
outer = gridspec.GridSpec(2, 1, wspace=10, hspace=0.1)
ax1 = plt.Subplot(fig, outer[0])
fig.add_subplot(ax1)
ax2 = plt.Subplot(fig, outer[1])
fig.add_subplot(ax2)
print('Predicted: %9s\nTrue:      %9s\n=> %s' % (pred_text, label, pred_text == label))
image = image[:, :, 0].T
ax1.set_title('True: {}\nPred: {}'.format(label, pred_text), loc='left')
ax1.imshow(image, cmap='gray')
ax1.set_xticks([])
ax1.set_yticks([])
ax2.set_title('Activations')
ax2.imshow(net_out_value[0].T, cmap='binary', interpolation='nearest')
ax2.set_yticks(list(range(len(LabelCodec.ALPHABET) + 1)))
ax2.set_yticklabels(LabelCodec.ALPHABET)  # + ['blank'])
ax2.grid(False)
for h in np.arange(-0.5, len(LabelCodec.ALPHABET) + 1 + 0.5, 1):
    ax2.axhline(h, linestyle='-', color='k', alpha=0.5, linewidth=1)

plt.show()
