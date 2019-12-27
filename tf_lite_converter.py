import argparse
import os

import tensorflow as tf

from tensorflow.keras.models import save_model
from tensorflow.python.keras.models import Model
from tensorflow_core.lite.python.lite import TFLiteConverter

from config import config
from label_codec import LabelCodec
from pyimagesearch.nn.conv import OCR

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--optimizer", default="rmsprop", help="optimizer method: sdg, rmsprop, adam, adagrad, adadelta")
args = vars(ap.parse_args())

OPTIMIZER = args["optimizer"]
MODEL_PATH = os.path.sep.join([config.OUTPUT_PATH, OPTIMIZER, config.MODEL_NAME]) + ".h5"
MODEL_WEIGHTS_PATH = os.path.sep.join([config.OUTPUT_PATH, OPTIMIZER, config.MODEL_NAME]) + '-weights.h5config'
TFLITE_MODEL_PATH = os.path.sep.join([config.OUTPUT_PATH, OPTIMIZER, config.MODEL_NAME]) + "tflite.h5"

print("Optimizer:    {}".format(OPTIMIZER))
print("Weights path: {}".format(MODEL_WEIGHTS_PATH))
print("Model path:   {}".format(MODEL_PATH))

tf.compat.v1.disable_eager_execution()

inputs, predictions = OCR.build((config.IMAGE_WIDTH, config.IMAGE_HEIGHT, 1), config.POOL_SIZE, len(LabelCodec.ALPHABET) + 1)
model = Model(inputs=inputs, outputs=predictions)

model.load_weights(MODEL_WEIGHTS_PATH)
save_model(model, filepath=MODEL_PATH, save_format="h5")

converter = TFLiteConverter.from_keras_model_file(MODEL_PATH)
tflite_model = converter.convert()

with open(TFLITE_MODEL_PATH, 'wb') as file:
    file.write(tflite_model)
