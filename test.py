from keras.optimizers import SGD
import argparse
from config import anpr_config as config
from DatasetGenerator import DatasetGenerator
from pyimagesearch.nn.conv import OCR

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", default="license_number_model.h5", help="model file")
args = vars(ap.parse_args())

trainGen = DatasetGenerator(config.TRAIN_HDF5, config.IMAGE_WIDTH, config.IMAGE_HEIGHT, config.POOL_SIZE,
                            config.BATCH_SIZE)

valGen = DatasetGenerator(config.VAL_HDF5, config.IMAGE_WIDTH, config.IMAGE_HEIGHT, config.POOL_SIZE,
                          config.BATCH_SIZE)

# clipnorm seems to speeds up convergence
sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

model = OCR.build(config.IMAGE_WIDTH, config.IMAGE_HEIGHT, config.POOL_SIZE, trainGen.get_output_size(),
                  config.MAX_TEXT_LEN)

# the loss calc occurs elsewhere, so use a dummy lambda func for the loss
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)

model.fit_generator(generator=trainGen.next_batch(),
                    steps_per_epoch=trainGen.n,
                    epochs=config.BATCH_SIZE,
                    validation_data=valGen.next_batch(),
                    validation_steps=valGen.n)

print("[INFO] saving model...")
model.save(config.MODEL_PATH)
print("[INFO] model saved to: %s" % config.MODEL_PATH)

# close the databases
trainGen.close()
valGen.close()
