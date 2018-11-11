import argparse

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


def preprocess(image, width, height):
    # resize image to target width keeping the aspect ratio
    image = imutils.resize(image, width=width)

    # determine the padding values for the height to obtain the target dimensions
    padH = int((config.IMAGE_HEIGHT - image.shape[0]) / 2.0)

    # pad the image then apply one more resizing to handle any rounding issues
    image = cv2.copyMakeBorder(image, padH, padH, 0, 0, cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))

    # normalize data
    image = image.astype(np.float32) / 255.
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

img_filepath = "D:/development/datasets/alpr/val/test9.png"
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
