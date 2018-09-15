from keras.optimizers import SGD
import argparse
from nn import OCR
from dataset.LicensePlateDatasetLoader import LicensePlateDatasetLoader
from preprocessing import RandomRotatePreprocessor
from preprocessing import GaussianNoisePreprocessor

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", default="license_number_model.h5", help="model file")
ap.add_argument("-d", "--data", default="data", help="data directory")
ap.add_argument("-e", "--epochs", type=float, default=3, help="# of epochs")
ap.add_argument("-b", "--batch-size", type=int, default=32, help="size of batches")
args = vars(ap.parse_args())

# train parameters
img_w = 128
img_h = 64
pool_size = 2
batch_size = args["batch_size"]
epochs = args["epochs"]

print("[INFO] loading data...")
preprocessors = [
    RandomRotatePreprocessor(-20, 20, img_w, img_h),
    GaussianNoisePreprocessor(35)]

train_dataset = LicensePlateDatasetLoader(img_w, img_h, pool_size, batch_size, preprocessors)
train_dataset.load(args["data"] + '/train')

test_dataset = LicensePlateDatasetLoader(img_w, img_h, pool_size, batch_size, preprocessors)
test_dataset.load(args["data"] + '/test')

# clipnorm seems to speeds up convergence
sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

model = OCR.build(img_w, img_h, pool_size, train_dataset.get_output_size(), train_dataset.get_max_text_len())

# the loss calc occurs elsewhere, so use a dummy lambda func for the loss
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)

print("[INFO] training...")
model.fit_generator(generator=train_dataset.next_batch(),
                    steps_per_epoch=train_dataset.get_data_size(),
                    epochs=epochs,
                    validation_data=test_dataset.next_batch(),
                    validation_steps=test_dataset.get_data_size())

print("[INFO] saving model...")
model.save(args["model"])
print("[INFO] model saved to: %s" % args["model"])
