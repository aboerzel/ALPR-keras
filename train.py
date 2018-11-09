import argparse
import cv2

import h5py
import numpy as np
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import SGD
from sklearn.model_selection import StratifiedKFold

from config import alpr_config as config
from licence_plate_dataset_generator import LicensePlateDatasetGenerator
from pyimagesearch.callbacks import CustomModelCheckpoint
from pyimagesearch.nn.conv import OCR
from label_codec import LabelCodec

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", default=config.MODEL_PATH, help="model file")
args = vars(ap.parse_args())

print("[INFO] loading data...")

trainData = h5py.File(config.TRAIN_HDF5)
images = np.array(trainData["images"])
labels = np.array(trainData["labels"])
trainData.close()


def create_model():
    optimizer = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    # optimizer = Adam(lr=0.001, decay=0.001 / config.NUM_EPOCHS)

    model = OCR.build(config.IMAGE_WIDTH, config.IMAGE_HEIGHT, config.POOL_SIZE,
                      LabelCodec.get_alphabet_len() + 1, config.MAX_TEXT_LEN)

    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)

    return model


def get_callbacks():
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
    return callbacks


k = 10
cvscores = []
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=1)

k = 10
l = int(len(images) / k)

for i in range(k):
    xval = images[i * l:(i + 1) * l]
    yval = labels[i * l:(i + 1) * l]

    xtrain = np.concatenate([images[:i * l], images[(i + 1) * l:]]);
    ytrain = np.concatenate([labels[:i * l], labels[(i + 1) * l:]]);

    print("Training on fold " + str(i + 1) + "/{0}...".format(k))

    model = create_model()

    trainGen = LicensePlateDatasetGenerator(xtrain, ytrain, config.IMAGE_WIDTH, config.IMAGE_HEIGHT,
                                            config.POOL_SIZE, config.MAX_TEXT_LEN, config.BATCH_SIZE)

    valGen = LicensePlateDatasetGenerator(xval, yval, config.IMAGE_WIDTH, config.IMAGE_HEIGHT,
                                            config.POOL_SIZE, config.MAX_TEXT_LEN, config.BATCH_SIZE)
    history = model.fit_generator(
        trainGen.generator(),
        steps_per_epoch=trainGen.numImages // config.BATCH_SIZE,
        validation_data=valGen.generator(),
        validation_steps=valGen.numImages // config.BATCH_SIZE,
        epochs=config.NUM_EPOCHS,
        max_queue_size=10,
        callbacks=get_callbacks(),
        verbose=1)

    scores = model.evaluate(xval, yval, verbose=1)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    cvscores.append(scores[1] * 100)

    accuracy_history = history.history['acc']
    val_accuracy_history = history.history['val_acc']
    print("Last training accuracy: " + str(accuracy_history[-1]) + ", last validation accuracy: " + str(
        val_accuracy_history[-1]))

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
