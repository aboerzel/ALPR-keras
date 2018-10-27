import itertools
import cv2
import h5py
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from keras.optimizers import SGD
from config import alpr_config as config


def decode(out):
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = ''
        for c in out_best:
            if c < len(config.ALPHABET):
                outstr += config.ALPHABET[c]
        return outstr


sess = tf.Session()
K.set_session(sess)

model = load_model(config.MODEL_PATH, compile=False)

optimizer = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)

net_inp = model.get_layer(name='input').input
net_out = model.get_layer(name='softmax').output

# load validation data from dataset
validationData = h5py.File(config.VAL_HDF5)
images = np.array(validationData["images"])
labels = np.array(validationData["labels"])
validationData.close()

# shuffle images and lables
randomize = np.arange(len(images))
np.random.shuffle(randomize)
images = images[randomize]
labels = labels[randomize]

for image, label in zip(images, labels):
    image = cv2.resize(image, (config.IMAGE_WIDTH, config.IMAGE_HEIGHT))
    image = image.astype("float") / 255.0
    image = np.expand_dims(image.T, -1)
    X_data = [image]
    net_out_value = sess.run(net_out, feed_dict={net_inp: X_data})
    pred_text = decode(net_out_value)
    print('Predicted: %s - True: %s - %s' % (pred_text, label, pred_text == label))
