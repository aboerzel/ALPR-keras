import argparse
import os

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import save_model
from tensorflow.keras.optimizers import SGD, Adam, Adagrad, Adadelta, RMSprop
from tensorflow.python.keras import Input
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.models import Model

from config import config
from label_codec import LabelCodec
from licence_plate_dataset_generator import LicensePlateDatasetGenerator
from license_plate_image_augmentator import LicensePlateImageAugmentator
from pyimagesearch.callbacks import CustomModelCheckpoint
from pyimagesearch.io.hdf5datasetloader import Hdf5DatasetLoader
from pyimagesearch.nn.conv import OCR

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--optimizer", default="rmsprop", help="optimizer method: sdg, rmsprop, adam, adagrad, adadelta")
args = vars(ap.parse_args())

OPTIMIZER = args["optimizer"]
MODEL_PATH = os.path.sep.join([config.OUTPUT_PATH, OPTIMIZER, config.MODEL_NAME]) + ".h5"
MODEL_WEIGHTS_PATH = os.path.sep.join([config.OUTPUT_PATH, OPTIMIZER, config.MODEL_NAME]) + '-weights.h5config'

print("Optimizer:    {}".format(OPTIMIZER))
print("Weights path: {}".format(MODEL_WEIGHTS_PATH))
print("Model path:   {}".format(MODEL_PATH))

tf.compat.v1.disable_eager_execution()


def get_optimizer(optimizer):
    if optimizer == "sdg":
        return SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    if optimizer == "rmsprop":
        return RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    if optimizer == "adam":
        return Adam(lr=0.001, decay=0.001 / config.NUM_EPOCHS)
        # Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    if optimizer == "adagrad":
        return Adagrad(0.01)
    if optimizer == "adadelta":
        return Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)


def get_callbacks(optimizer):
    logdir = os.path.join("logs", optimizer)

    callbacks = [
        EarlyStopping(monitor='loss', min_delta=0.01, patience=5, mode='min', verbose=1),
        CustomModelCheckpoint(model_to_save=train_model, filepath=MODEL_WEIGHTS_PATH, monitor='loss', verbose=1,
                              save_best_only=True, save_weights_only=True, mode='min'),
        ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2, verbose=1, mode='min', min_delta=0.01, cooldown=0,
                          min_lr=0),
        TensorBoard(log_dir=logdir)]
    return callbacks


print("[INFO] loading data...")
loader = Hdf5DatasetLoader()
X_train, y_train = loader.load(config.TRAIN_HDF5, shuffle=True)
X_test, y_test = loader.load(config.TEST_HDF5, shuffle=True)
# X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

loader = Hdf5DatasetLoader()
background_images = loader.load(config.SUN397_HDF5, shuffle=True, max_items=10000)

augmentator = LicensePlateImageAugmentator(config.IMAGE_WIDTH, config.IMAGE_HEIGHT, background_images)
train_generator = LicensePlateDatasetGenerator(X_train, y_train, config.IMAGE_WIDTH, config.IMAGE_HEIGHT,
                                               config.POOL_SIZE, config.MAX_TEXT_LEN, config.BATCH_SIZE, augmentator)

val_generator = LicensePlateDatasetGenerator(X_test, y_test, config.IMAGE_WIDTH, config.IMAGE_HEIGHT,
                                             config.POOL_SIZE, config.MAX_TEXT_LEN, config.BATCH_SIZE, augmentator)

print("Train dataset size: {}".format(X_train.shape[0]))
print("Test dataset size:  {}".format(X_test.shape[0]))


class CTCLoss(tf.keras.losses.Loss):

    def __init__(self, input_length, label_length, name='CTCLoss'):
        super().__init__(name=name)
        self.input_length = input_length
        self.label_length = label_length

    def call(self, labels, predictions):
        loss = tf.keras.backend.ctc_batch_cost(labels, predictions, self.input_length, self.label_length)
        loss = tf.reduce_mean(loss)
        return loss


print("[INFO] build model...")
labels = Input(name='labels', shape=(config.MAX_TEXT_LEN,), dtype='float32')
input_length = Input(name='input_length', shape=(1,), dtype='int64')
label_length = Input(name='label_length', shape=(1,), dtype='int64')

inputs, predictions = OCR.build((config.IMAGE_WIDTH, config.IMAGE_HEIGHT, 1), config.POOL_SIZE,
                                len(LabelCodec.ALPHABET) + 1)
train_model = Model(inputs=[inputs, labels, input_length, label_length], outputs=predictions)
train_model.add_loss(CTCLoss(input_length, label_length)(labels, predictions))
train_model.compile(loss=None, optimizer=get_optimizer(OPTIMIZER))

print("[INFO] model architecture...")
train_model.summary()

print("[INFO] train model...")
history = train_model.fit(
    train_generator.generator(),
    steps_per_epoch=train_generator.numImages // config.BATCH_SIZE,
    validation_data=val_generator.generator(),
    validation_steps=val_generator.numImages // config.BATCH_SIZE,
    epochs=config.NUM_EPOCHS,
    callbacks=get_callbacks(OPTIMIZER), verbose=1)

print("[INFO] save model...")
predict_model = Model(inputs=inputs, outputs=predictions)
predict_model.load_weights(MODEL_WEIGHTS_PATH)
save_model(predict_model, filepath=MODEL_PATH, save_format="h5")

# print("[INFO] evaluating model...")
# X_test, y_test = next(val_generator.generator())
# score = train_model.evaluate(X_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
