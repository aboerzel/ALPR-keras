import argparse

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

from config import alpr_config as config
from licence_plate_dataset_generator import LicensePlateDatasetGenerator
from pyimagesearch.callbacks import CustomModelCheckpoint
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
trainGen = LicensePlateDatasetGenerator(config.TRAIN_HDF5, config.IMAGE_WIDTH, config.IMAGE_HEIGHT, config.POOL_SIZE,
                                        config.MAX_TEXT_LEN, config.BATCH_SIZE, preprocessors=[sp], aug=aug)

valGen = LicensePlateDatasetGenerator(config.VAL_HDF5, config.IMAGE_WIDTH, config.IMAGE_HEIGHT, config.POOL_SIZE,
                                      config.MAX_TEXT_LEN, config.BATCH_SIZE, preprocessors=[sp])

# define optimizer
optimizer = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
# optimizer = Adam(lr=0.001, decay=0.001 / config.NUM_EPOCHS)

model = OCR.build(config.IMAGE_WIDTH, config.IMAGE_HEIGHT, config.POOL_SIZE, trainGen.get_output_size(),
                  config.MAX_TEXT_LEN)

model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)

# construct the set of callbacks
# callbacks = [
#     EpochCheckpoint(config.CHECKPOINTS_PATH, every=5, startAt=config.START_EPOCH),
#     TrainingMonitor(config.FIG_PATH, jsonPath=config.JSON_PATH, startAt=config.START_EPOCH)]

callbacks = [
    EarlyStopping(monitor='loss', min_delta=0.01, patience=5, mode='min', verbose=1),
    CustomModelCheckpoint(model_to_save=model, filepath=config.MODEL_CHECKPOINT_PATH,
                          monitor='loss', verbose=1, save_best_only=True, mode='min', period=1),
    ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2, verbose=1, mode='min', min_delta=0.01,
                      cooldown=0, min_lr=0)]

print("[INFO] training...")
model.fit_generator(
    trainGen.generator(),
    steps_per_epoch=trainGen.numImages // config.BATCH_SIZE,
    validation_data=valGen.generator(),
    validation_steps=valGen.numImages // config.BATCH_SIZE,
    epochs=config.NUM_EPOCHS,
    max_queue_size=10,
    callbacks=callbacks, verbose=1)

print("[INFO] saving model...")
model.save(config.MODEL_PATH)
print("[INFO] model saved to: %s" % config.MODEL_PATH)

# close the databases
trainGen.close()
valGen.close()
