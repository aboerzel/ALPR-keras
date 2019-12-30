import argparse
import os

import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.models import save_model
from tensorflow.python.keras.models import Model
from tensorflow.python.tools import freeze_graph
from tensorflow_core.lite.python.lite import TFLiteConverter

from config import config
from label_codec import LabelCodec
from pyimagesearch.nn.conv import OCR

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--optimizer", default=config.OPTIMIZER, help="supported optimizer methods: sdg, rmsprop, adam, adagrad, adadelta")
args = vars(ap.parse_args())

OPTIMIZER = args["optimizer"]
MODEL_PATH = os.path.join(config.OUTPUT_PATH, OPTIMIZER, config.MODEL_NAME) + ".h5"
MODEL_WEIGHTS_PATH = os.path.join(config.OUTPUT_PATH, OPTIMIZER, config.MODEL_NAME) + '-weights.h5'
TFLITE_MODEL_PATH = os.path.join(config.OUTPUT_PATH, OPTIMIZER, config.MODEL_NAME) + ".tflite"

print("Optimizer:    {}".format(OPTIMIZER))
print("Weights path: {}".format(MODEL_WEIGHTS_PATH))
print("Model path:   {}".format(MODEL_PATH))

tf.compat.v1.disable_eager_execution()

inputs, predictions = OCR.conv_bgru((config.IMAGE_WIDTH, config.IMAGE_HEIGHT, 1), len(LabelCodec.ALPHABET) + 1)
model = Model(inputs=inputs, outputs=predictions)

model.load_weights(MODEL_WEIGHTS_PATH)

#save_model(model, "out", include_optimizer=False)
tf.saved_model.save(model, "out")
#model1 = tf.saved_model.load("out")

freeze_graph.freeze_graph(None,
                          None,
                          None,
                          None,
                          model.outputs[0].op.name,
                          None,
                          None,
                          os.path.join("out", "frozen_model.pb"),
                          False,
                          "",
                          input_saved_model_dir="out")

converter = TFLiteConverter.from_saved_model("out")
tflite_model = converter.convert()


with open(TFLITE_MODEL_PATH, 'wb') as file:
    file.write(tflite_model)

converter = TFLiteConverter.from_keras_model_file(MODEL_PATH)
# converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_model = converter.convert()

with open(TFLITE_MODEL_PATH+"-1", 'wb') as file:
    file.write(tflite_model)




# K.clear_session()
# K.set_learning_phase(0)


#with tf.keras.backend.get_session() as sess:
    #input_tensor = sess.graph.get_tensor_by_name('input:0')
    #output_tensor = sess.graph.get_tensor_by_name('time_distributed_1/Reshape_1:0')
#converter = TFLiteConverter.from_keras_model_file(MODEL_PATH) #.from_session(sess, [input_tensor], [output_tensor])

#tflite_model = converter.convert()
#print('Model converted successfully!')

#with open(TFLITE_MODEL_PATH, 'wb') as file:
#    file.write(tflite_model)

#tf.saved_model.save(K.get_session(),
#                           "out",
#                           inputs={"input": model.inputs[0]},
#                           outputs={"output": model.outputs[0]})

#model.save("out")

