import random

import numpy as np

from label_codec import LabelCodec
from license_plate_image_augmentor import LicensePlateImageAugmentor


class LicensePlateDatasetGenerator:
    def __init__(self, images, labels, img_w, img_h, pool_size, max_text_len, batch_size, background_image_db_path):

        self.img_w = img_w
        self.img_h = img_h
        self.max_text_len = max_text_len
        self.batch_size = batch_size
        self.downsample_factor = pool_size ** 2

        self.images = images
        self.labels = labels
        self.numImages = self.labels.shape[0]

        self.indexes = np.asarray(range(self.numImages))
        random.shuffle(self.indexes)
        self.batch_index = 0

        self.augmentor = LicensePlateImageAugmentor(img_w, img_h, background_image_db_path)

    def next_batch(self):

        if self.batch_index >= (self.numImages // self.batch_size):
            self.batch_index = 0
            random.shuffle(self.indexes)

        current_index = self.batch_index * self.batch_size
        batch_indexes = self.indexes[current_index:current_index + self.batch_size]
        self.batch_index += 1
        return self.images[batch_indexes], self.labels[batch_indexes]

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

            for i, (image, number) in enumerate(zip(x_data, y_data)):
                image = self.augmentor.generate_plate_image(image)
                image = image.T
                image = np.expand_dims(image, -1)
                data[i] = image
                text_length = len(number)
                labels[i, 0:text_length] = LabelCodec.encode_number(number)
                label_length[i] = text_length

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

    @staticmethod
    def get_output_size():
        return LabelCodec.get_alphabet_len() + 1
