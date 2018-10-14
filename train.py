import argparse
import json

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

from config import anpr_config as config
from pyimagesearch.callbacks import EpochCheckpoint, TrainingMonitor
from pyimagesearch.io import HDF5DatasetGenerator
from pyimagesearch.nn.conv import OCR
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import MeanPreprocessor
from pyimagesearch.preprocessing import SimplePreprocessor

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", default=config.MODEL_PATH, help="model file")
args = vars(ap.parse_args())


print("[INFO] loading data...")

# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=18, zoom_range=0.15,
                         width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
                         horizontal_flip=True, fill_mode="nearest")

# load the RGB means for the training set
means = json.loads(open(config.DATASET_MEAN).read())

# initialize the image preprocessors
sp = SimplePreprocessor(config.IMAGE_WIDTH, config.IMAGE_HEIGHT)
# mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()

# initialize the training and validation images generators
trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, config.BATCH_SIZE, preprocessors=[sp, iap], aug=aug)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, config.BATCH_SIZE, preprocessors=[sp, iap])

# clipnorm seems to speeds up convergence
optimizer = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
#optimizer = Adam(lr=0.001, decay=0.001 / config.NUM_EPOCHS)

model = OCR.build(config.IMAGE_WIDTH, config.IMAGE_HEIGHT, config.POOL_SIZE, len(config.LETTERS)+1, config.MAX_TEXT_LEN)

# the loss calc occurs elsewhere, so use a dummy lambda func for the loss
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)

# construct the set of callbacks
callbacks = [
    EpochCheckpoint(config.CHECKPOINTS, every=5, startAt=config.START_EPOCH),
    TrainingMonitor(config.FIG_PATH, jsonPath=config.JSON_PATH, startAt=config.START_EPOCH)]

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
