from keras.optimizers import SGD
import argparse
from nn import OCR
from dataset.LicensePlateDatasetLoader import LicensePlateDatasetLoader
from preprocessing import RandomRotatePreprocessor
from preprocessing import RandomGaussianNoisePreprocessor
from utils.utils import makedirs
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from callbacks import CustomModelCheckpoint, CustomTensorBoard

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", default="license_number_model.h5", help="model file")
ap.add_argument("-d", "--data", default="data", help="data directory")
ap.add_argument("-e", "--epochs", type=float, default=3, help="# of epochs")
ap.add_argument("-b", "--batch-size", type=int, default=32, help="size of batches")
ap.add_argument("-t", "--tensorboard-dir", default="logs", help="tensorboard log directory")
args = vars(ap.parse_args())

# train parameters
img_w = 128
img_h = 64
pool_size = 2
batch_size = args["batch_size"]
epochs = args["epochs"]

print("[INFO] loading data...")
preprocessors = [
    RandomRotatePreprocessor(-10, 10, img_w, img_h),
    RandomGaussianNoisePreprocessor(25)]

train_dataset = LicensePlateDatasetLoader(img_w, img_h, pool_size, batch_size, preprocessors)
train_dataset.load(args["data"] + '/train')

test_dataset = LicensePlateDatasetLoader(img_w, img_h, pool_size, batch_size, preprocessors)
test_dataset.load(args["data"] + '/test')


def create_callbacks(saved_weights_name, tensorboard_logs, model_to_save):
    makedirs(tensorboard_logs)

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


# clipnorm seems to speeds up convergence
sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

model = OCR.build(img_w, img_h, pool_size, train_dataset.get_output_size(), train_dataset.get_max_text_len())

# the loss calc occurs elsewhere, so use a dummy lambda func for the loss
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)

callbacks = create_callbacks(args["model"], args['tensorboard_dir'], model)

print("[INFO] training...")
model.fit_generator(generator=train_dataset.next_batch(),
                    steps_per_epoch=train_dataset.get_data_size(),
                    epochs=epochs,
                    validation_data=test_dataset.next_batch(),
                    validation_steps=test_dataset.get_data_size(),
                    callbacks=callbacks,
                    workers=4,
                    max_queue_size=8)

print("[INFO] saving model...")
model.save(args["model"])
print("[INFO] model saved to: %s" % args["model"])
