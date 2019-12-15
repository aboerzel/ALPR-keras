import argparse
import functools
import os

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import SGD, Adam, Adagrad, Adadelta, RMSprop
from tensorflow.python.training.tracking.util import Checkpoint
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.saving.saved_model_experimental import export_saved_model, load_from_saved_model
from tensorflow.python.keras import Input

from config import alpr_config as config
from label_codec import LabelCodec
from licence_plate_dataset_generator import LicensePlateDatasetGenerator
from license_plate_image_augmentor import LicensePlateImageAugmentor
from pyimagesearch.callbacks import CustomModelCheckpoint
from pyimagesearch.io.hdf5datasetloader import Hdf5DatasetLoader
from pyimagesearch.nn.conv import OCR

from sklearn.model_selection import train_test_split

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", default=config.MODEL_PATH, help="model file")
ap.add_argument("-o", "--optimizer", default="sdg", help="optimizer method: sdg, rmsprop, adam, adagrad, adadelta")
args = vars(ap.parse_args())


def get_optimizer(optimizer_method):
    if optimizer_method == "sdg":
        return SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    if optimizer_method == "rmsprop":
        return RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    if optimizer_method == "adam":
        return Adam(lr=0.001, decay=0.001 / config.NUM_EPOCHS)
        # Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    if optimizer_method == "adagrad":
        return Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
    if optimizer_method == "adadelta":
        return Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)


def get_callbacks(output_path, optimizer_method, model_name):
    model_checkpoint_path = os.path.sep.join(
        [output_path, optimizer_method, model_name]) + '.h5'

    callbacks = [
        EarlyStopping(monitor='loss', min_delta=0.01, patience=5, mode='min', verbose=1),
        CustomModelCheckpoint(model_to_save=model, filepath=model_checkpoint_path,
                              monitor='loss', verbose=1, save_best_only=True, mode='min', period=1),
        ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2, verbose=1, mode='min', min_delta=0.01,
                          cooldown=0, min_lr=0)]
    return callbacks


print("[INFO] loading data...")
loader = Hdf5DatasetLoader()
images, labels = loader.load(config.TRAIN_HDF5, shuffle=True)

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

loader = Hdf5DatasetLoader()
background_images = loader.load(config.DATASET_ROOT_PATH + '/hdf5/background.h5', shuffle=True, max_items=10000)

augmentor = LicensePlateImageAugmentor(config.IMAGE_WIDTH, config.IMAGE_HEIGHT, background_images)
train_generator = LicensePlateDatasetGenerator(X_train, y_train, config.IMAGE_WIDTH, config.IMAGE_HEIGHT,
                                               config.POOL_SIZE, config.MAX_TEXT_LEN, config.BATCH_SIZE, augmentor)

val_generator = LicensePlateDatasetGenerator(X_test, y_test, config.IMAGE_WIDTH, config.IMAGE_HEIGHT,
                                             config.POOL_SIZE, config.MAX_TEXT_LEN, config.BATCH_SIZE, augmentor)

# Average the loss across the batch size within an epoch
train_loss = tf.keras.metrics.Mean(name='train_loss')
valid_loss = tf.keras.metrics.Mean(name='test_loss')
best_loss = None
epochs_without_improvement = 0
patience = 10

# model = CLPR(config.IMAGE_WIDTH, config.IMAGE_HEIGHT, config.POOL_SIZE, len(LabelCodec.ALPHABET) + 1)

model, input_length, label_length = OCR.build_train_model(config.IMAGE_WIDTH, config.IMAGE_HEIGHT, config.POOL_SIZE, len(LabelCodec.ALPHABET) + 1,
                              config.MAX_TEXT_LEN)


# model.add_loss(ctc_loss(y_pred=y_pred, input_length=input_length, label_length=label_length, labels=labels))
# model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001))

def ctc_loss(labels, predictions, input_length, label_length):
    # the 2 is critical here since the first couple outputs of the RNN tend to be garbage:
    predictions = predictions[:, 2:, :]
    return tf.keras.backend.ctc_batch_cost(labels, predictions, input_length, label_length)


ctc_loss_prepare_fn = functools.partial(ctc_loss, input_length=input_length, label_length=label_length)

OPTIMIZER = "adagrad"

model.compile(loss=ctc_loss_prepare_fn,
              optimizer=get_optimizer(OPTIMIZER),
              metrics=['accuracy'],
              experimental_run_tf_function=False)

# model.summary()

history = model.fit(
    train_generator.generator(),
    steps_per_epoch=train_generator.numImages // config.BATCH_SIZE,
    validation_data=val_generator.generator(),
    validation_steps=val_generator.numImages // config.BATCH_SIZE,
    epochs=config.NUM_EPOCHS,
    callbacks=get_callbacks(config.OUTPUT_PATH, OPTIMIZER, config.MODEL_NAME), verbose=1)

OPTIMIZER_METHOD = "rmsprop"

optimizer = get_optimizer(OPTIMIZER_METHOD)

ckpt = Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)

model_checkpoint_path = os.path.sep.join([config.OUTPUT_PATH, OPTIMIZER_METHOD, config.MODEL_NAME])


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
