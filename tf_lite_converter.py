import argparse
import os

from tensorflow_core.lite.python.lite import TFLiteConverter
from config import config

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--optimizer", default=config.OPTIMIZER, help="supported optimizer methods: sdg, rmsprop, adam, adagrad, adadelta")
args = vars(ap.parse_args())

OPTIMIZER = args["optimizer"]
MODEL_PATH = os.path.join(config.OUTPUT_PATH, OPTIMIZER, config.MODEL_NAME) + ".h5"
TFLITE_MODEL_PATH = os.path.join(config.OUTPUT_PATH, OPTIMIZER, config.MODEL_NAME) + ".tflite"

print("Optimizer:    {}".format(OPTIMIZER))
print("Model path:   {}".format(MODEL_PATH))

converter = TFLiteConverter.from_keras_model_file(MODEL_PATH)
#converter = TFLiteConverter.from_saved_model("D:/development/tensorflow/saved-model/saved_model-1")
#converter = TFLiteConverter.from_keras_model(model)
converter.experimental_new_converter = True
tflite_model = converter.convert()

open("output/adagrad/clpr-model.tflite", "wb").write(tflite_model)

