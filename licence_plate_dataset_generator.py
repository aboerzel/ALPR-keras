# import the necessary packages
import random
import h5py
import numpy as np
from config import alpr_config as config
from licens_plate_augmentor import LicensePlateAugmentor


class LicensePlateDatasetGenerator:
    def __init__(self, db_path, img_w, img_h, pool_size, max_text_len, batch_size):

        self.img_w = img_w
        self.img_h = img_h
        self.max_text_len = max_text_len
        self.batch_size = batch_size
        self.downsample_factor = pool_size ** 2

        # open the HDF5 database for reading and determine the total number of entries in the database
        self.db = h5py.File(db_path)
        self.numImages = self.db["labels"].shape[0]

        self.indexes = list(range(self.numImages))
        random.shuffle(self.indexes)
        self.cur_index = 0
        self.batch_index = 0

        self.augmentor = LicensePlateAugmentor(img_w, img_h)

    def next_sample(self):
        self.cur_index += 1

        if self.cur_index >= self.numImages:
            self.cur_index = 0
            random.shuffle(self.indexes)

        return self.db["images"][self.indexes[self.cur_index]], self.db["labels"][self.indexes[self.cur_index]]

    def next_batch(self):

        if self.batch_index >= (self.numImages // self.batch_size):
            self.batch_index = 0
            random.shuffle(self.indexes)

        index = self.indexes[self.batch_index]
        self.batch_index += 1
        return self.db["images"][index:index + self.batch_size], self.db["labels"][index:index + self.batch_size]

    def generator(self, passes=np.inf):
        # initialize the epoch count
        epochs = 0

        # keep looping infinitely -- the model will stop once we have
        # reach the desired number of epochs
        while epochs < passes:

            data = np.ones([self.batch_size, self.img_w, self.img_h, 1])
            labels = np.ones([self.batch_size, self.max_text_len])
            input_length = np.ones((self.batch_size, 1)) * self.img_w // self.downsample_factor - 2
            label_length = np.zeros((self.batch_size, 1))

            x_data, y_data = self.next_batch()
            x_data, y_data = self.augmentor.generator(x_data, y_data)

            for i, (image, number) in enumerate(zip(x_data, y_data)):
                image = image.T
                image = np.expand_dims(image, -1)
                data[i] = image
                text_length = len(number)
                labels[i, 0:text_length] = self.number_to_labels(number)
                label_length[i] = text_length

            data = data.astype("float") / 255.0

            inputs = {
                'input': data,
                'labels': labels,
                'input_length': input_length,
                'label_length': label_length
            }
            outputs = {'ctc': np.zeros([self.batch_size])}
            yield (inputs, outputs)

            # increment the total number of epochs
            epochs += 1

    def close(self):
        # close the database
        self.db.close()

    @staticmethod
    def get_output_size():
        return len(config.ALPHABET) + 1

    # Translation of characters to unique numerical classes
    @staticmethod
    def number_to_labels(number):
        return list(map(lambda c: config.ALPHABET.index(c), number))

    # Reverse translation of numerical classes back to characters
    @staticmethod
    def labels_to_number(labels):
        ret = []
        for c in labels:
            if c == len(config.ALPHABET):  # CTC Blank
                ret.append("")
            else:
                ret.append(config.ALPHABET[c])
        return "".join(ret)
