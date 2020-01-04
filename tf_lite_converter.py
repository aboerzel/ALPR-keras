import argparse
import os
import numpy as np

from tensorflow_core.lite.python.interpreter import Interpreter
from tensorflow_core.lite.python.lite import TFLiteConverter
from config import config

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--optimizer", default=config.OPTIMIZER, help="supported optimizer methods: sdg, rmsprop, adam, adagrad, adadelta")
args = vars(ap.parse_args())

OPTIMIZER = args["optimizer"]
MODEL_PATH = os.path.join(config.OUTPUT_PATH, OPTIMIZER, config.MODEL_NAME) + ".h5"
SAVED_MODEL_PATH = os.path.join(config.OUTPUT_PATH, OPTIMIZER, "saved_model")
TFLITE_MODEL_PATH = os.path.join(config.OUTPUT_PATH, OPTIMIZER, config.MODEL_NAME) + ".tflite"

print("Optimizer:    {}".format(OPTIMIZER))
print("Model path:   {}".format(MODEL_PATH))

converter = TFLiteConverter.from_keras_model_file(MODEL_PATH)
# converter = TFLiteConverter.from_saved_model(SAVED_MODEL_PATH)
# converter = TFLiteConverter.from_keras_model(model)
converter.experimental_new_converter = True
# converter.post_training_quantize = True
tflite_model = converter.convert()

open(TFLITE_MODEL_PATH, "wb").write(tflite_model)


# Load TFLite model and allocate tensors.
interpreter = Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# Test model on random input data
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
