import argparse
import os

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

from config import anpr_config as config
from pyimagesearch.callbacks import CustomTensorBoard, CustomModelCheckpoint
from pyimagesearch.io import HDF5DatasetGenerator
from pyimagesearch.nn.conv import OCR
from pyimagesearch.preprocessing import SimplePreprocessor


ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", default=config.MODEL_PATH, help="model file")
args = vars(ap.parse_args())

print("[INFO] loading data...")

# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=18, zoom_range=0.15,
                         width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
                         horizontal_flip=True, fill_mode="nearest")

# initialize the image preprocessors
sp = SimplePreprocessor(config.IMAGE_WIDTH, config.IMAGE_HEIGHT)

# initialize the training and validation images generators
trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, config.IMAGE_WIDTH, config.IMAGE_HEIGHT, config.MAX_TEXT_LEN,
                                config.BATCH_SIZE, preprocessors=[sp], aug=aug)

valGen = HDF5DatasetGenerator(config.VAL_HDF5, config.IMAGE_WIDTH, config.IMAGE_HEIGHT, config.MAX_TEXT_LEN,
                              config.BATCH_SIZE, preprocessors=[sp])

# clipnorm seems to speeds up convergence
#optimizer = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
# optimizer = Adam(lr=0.001, decay=0.001 / config.NUM_EPOCHS)

model = OCR.build(config.IMAGE_WIDTH, config.IMAGE_HEIGHT, config.POOL_SIZE, trainGen.get_output_size(),
                  config.MAX_TEXT_LEN)

# the loss calc occurs elsewhere, so use a dummy lambda func for the loss
#model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer, metrics=['accuracy'])


# construct the set of callbacks
# callbacks = [
#     EpochCheckpoint(config.CHECKPOINTS_PATH, every=5, startAt=config.START_EPOCH),
#     TrainingMonitor(config.FIG_PATH, jsonPath=config.JSON_PATH, startAt=config.START_EPOCH)]

def create_callbacks(saved_weights_name, tensorboard_logs, model_to_save):
    if not os.path.exists(tensorboard_logs):
        os.makedirs(tensorboard_logs)

    early_stop = EarlyStopping(
        monitor='loss',
        min_delta=0.01,
        patience=5,
        mode='min',
        verbose=1
    )
    checkpoint = CustomModelCheckpoint(
        model_to_save=model_to_save,
        filepath=saved_weights_name,  # + '{epoch:02d}.h5',
        monitor='loss',
        verbose=1,
        save_best_only=True,
        mode='min',
        period=1
    )
    reduce_on_plateau = ReduceLROnPlateau(
        monitor='loss',
        factor=0.1,
        patience=2,
        verbose=1,
        mode='min',
        min_delta=0.01,
        cooldown=0,
        min_lr=0
    )
    tensorboard = CustomTensorBoard(
        log_dir=tensorboard_logs,
        write_graph=True,
        write_images=True,
    )
    return [early_stop, checkpoint, reduce_on_plateau, tensorboard]


# callbacks = create_callbacks(config.MODEL_FILENAME, config.TENSORBOARD_PATH, model)
callbacks = []

print("[INFO] training...")
model.fit_generator(
    trainGen.generator(),
    steps_per_epoch=trainGen.numImages // config.BATCH_SIZE,
    validation_data=valGen.generator(),
    validation_steps=valGen.numImages // config.BATCH_SIZE,
    epochs=config.NUM_EPOCHS,
    max_queue_size=10,
    callbacks=callbacks, verbose=0)

print("[INFO] saving model...")
model.save(config.MODEL_PATH)
print("[INFO] model saved to: %s" % config.MODEL_PATH)

# close the databases
trainGen.close()
valGen.close()
