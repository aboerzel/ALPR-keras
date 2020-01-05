import argparse
import functools
import os

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Input
from tensorflow.python.keras.models import model_from_json, Model
from tensorflow.python.keras.saving.saved_model_experimental import export_saved_model, load_from_saved_model
from tensorflow.python.training.tracking.util import Checkpoint

from config import config
from label_codec import LabelCodec
from train_helper import TrainHelper
from licence_plate_dataset_generator import LicensePlateDatasetGenerator
from license_plate_image_augmentator import LicensePlateImageAugmentator
from pyimagesearch.io.hdf5datasetloader import Hdf5DatasetLoader
from pyimagesearch.nn.conv import OCR

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--optimizer", default=config.OPTIMIZER, help="supported optimizer methods: sdg, rmsprop, adam, adagrad, adadelta")
args = vars(ap.parse_args())

OPTIMIZER = args["optimizer"]
MODEL_PATH = os.path.join(config.OUTPUT_PATH, OPTIMIZER, config.MODEL_NAME) + ".h5"
MODEL_WEIGHTS_PATH = os.path.join(config.OUTPUT_PATH, OPTIMIZER, config.MODEL_NAME) + '-weights.h5'

print("Optimizer:    {}".format(OPTIMIZER))
print("Weights path: {}".format(MODEL_WEIGHTS_PATH))
print("Model path:   {}".format(MODEL_PATH))


print("[INFO] loading data...")
loader = Hdf5DatasetLoader()
images, labels = loader.load(config.TRAIN_HDF5, shuffle=True)
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

loader = Hdf5DatasetLoader()
background_images = loader.load(config.BACKGRND_HDF5, shuffle=True, max_items=10000)

augmentator = LicensePlateImageAugmentator(config.IMAGE_WIDTH, config.IMAGE_HEIGHT, background_images)

train_generator = LicensePlateDatasetGenerator(X_train, y_train, config.IMAGE_WIDTH, config.IMAGE_HEIGHT,
                                               config.DOWNSAMPLE_FACTOR, config.MAX_TEXT_LEN, config.BATCH_SIZE,
                                               augmentator)

val_generator = LicensePlateDatasetGenerator(X_test, y_test, config.IMAGE_WIDTH, config.IMAGE_HEIGHT,
                                             config.DOWNSAMPLE_FACTOR, config.MAX_TEXT_LEN, config.BATCH_SIZE,
                                             augmentator)

print("[INFO] build model...")
labels = Input(name='labels', shape=(config.MAX_TEXT_LEN,), dtype='float32')
input_length = Input(name='input_length', shape=(1,), dtype='int64')
label_length = Input(name='label_length', shape=(1,), dtype='int64')

inputs, predictions = OCR.conv_bgru((config.IMAGE_WIDTH, config.IMAGE_HEIGHT, 1), len(LabelCodec.ALPHABET) + 1)
model = Model(inputs=[inputs, labels, input_length, label_length], outputs=predictions)


def ctc_loss(labels, predictions, input_length, label_length):
    return tf.keras.backend.ctc_batch_cost(labels, predictions, input_length, label_length)


ctc_loss_prepare_fn = functools.partial(ctc_loss, input_length=input_length, label_length=label_length)

model.compile(loss=ctc_loss_prepare_fn,
              optimizer=TrainHelper.get_optimizer(OPTIMIZER),
              metrics=['accuracy'],
              experimental_run_tf_function=False)

# model.summary()

# Average the loss across the batch size within an epoch
train_loss = tf.keras.metrics.Mean(name='train_loss')
valid_loss = tf.keras.metrics.Mean(name='test_loss')
best_loss = None
epochs_without_improvement = 0
patience = 10

optimizer = TrainHelper.get_optimizer(OPTIMIZER)

ckpt = Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)

model_checkpoint_path = os.path.join(config.OUTPUT_PATH, OPTIMIZER, config.MODEL_NAME)


def save_best(ckpt, best_loss, epochs_without_improvement):
    is_best = best_loss is None or bool(valid_loss.result().numpy() < best_loss)
    if is_best:
        best_loss = valid_loss.result().numpy()
        file_path = ckpt.save(file_prefix=model_checkpoint_path)
        print("Checkpoint {} saved!".format(file_path))
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
        print("No improvement for {} epochs!".format(epochs_without_improvement))

    return best_loss, epochs_without_improvement, epochs_without_improvement > patience


def model_train(epoch, step, steps, features, labels, input_length, label_length):
    # Define the GradientTape context
    with tf.GradientTape() as tape:
        # Get the probabilities
        predictions = model(features)
        # the 2 is critical here since the first couple outputs of the RNN tend to be garbage:
        predictions = predictions[:, 2:, :]
        # Calculate the loss
        loss = ctc_loss(labels, predictions, input_length, label_length)
    # Get the gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    # Update the weights
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)

    template = 'Epoch {} {}/{}, train_loss: {}'
    print(template.format(epoch + 1,
                          step + 1,
                          steps,
                          train_loss.result()))


def model_validate(epoch, step, steps, features, labels, input_length, label_length):
    predictions = model(features)
    loss = ctc_loss(labels, predictions, input_length, label_length)

    valid_loss(loss)
    template = 'Epoch {} {}/{}, valid_loss: {}'
    print(template.format(epoch + 1,
                          step + 1,
                          steps,
                          valid_loss.result()))


for epoch in range(config.NUM_EPOCHS):
    train_steps = int(train_generator.numImages / config.NUM_EPOCHS)
    for train_step in range(train_steps):
        data, labels, input_length, label_length = next(train_generator.generator())
        model_train(epoch, train_step, train_steps, data, labels, input_length, label_length)

    test_steps = int(val_generator.numImages / config.NUM_EPOCHS)
    for test_step in range(test_steps):
        data, labels, input_length, label_length = next(val_generator.generator())
        model_validate(epoch, test_step, test_steps, data, labels, input_length, label_length)

    template = 'Epoch {}, train_loss: {}, valid_loss: {}'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          valid_loss.result()))

    ckpt.step.assign_add(1)
    best_loss, epochs_without_improvement, early_stop = save_best(ckpt, best_loss, epochs_without_improvement)

    if early_stop:
        print("Traoning stopped after {} epochs!".format(epoch))
        break

# Export the model to a SavedModel
export_saved_model(model, 'path_to_saved_model')

# Recreate the exact same model
new_model = load_from_saved_model('path_to_saved_model')

# new_predictions = new_model.predict(x_test)


# model.save(config.MODEL_PATH, save_format="tf")
json_config = model.to_json()
with open('model_config.json', 'w') as json_file:
    json_file.write(json_config)
# Save weights to disk
model.save_weights('path_to_my_weights.h5')

# Reload the model from the 2 files we saved
with open('model_config.json') as json_file:
    json_config = json_file.read()
new_model = model_from_json(json_config)
new_model.load_weights('path_to_my_weights.h5')

# Check that the state is preserved
# new_predictions = new_model.predict(x_test)
# np.testing.assert_allclose(predictions, new_predictions, atol=1e-6)
