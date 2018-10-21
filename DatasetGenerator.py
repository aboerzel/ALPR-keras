import random

import cv2
import h5py
import numpy as np

# letters = sorted(list("ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜ0123456789- "))
letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜ0123456789- "


class DatasetGenerator:
    def __init__(self, dbPath, img_w, img_h, pool_size, max_text_len, batch_size):

        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size
        self.max_text_len = max_text_len
        self.downsample_factor = pool_size ** 2

        self.db = h5py.File(dbPath)
        self.n = self.db["labels"].shape[0]
        self.indexes = list(range(self.n))
        self.cur_index = 0

    @staticmethod
    def get_output_size():
        return len(letters) + 1

    def next_sample(self):
        self.cur_index += 1
        if self.cur_index >= self.n:
            self.cur_index = 0
            random.shuffle(self.indexes)
        return self.db["images"][self.indexes[self.cur_index]], self.db["labels"][self.indexes[self.cur_index]]

    def next_batch(self):
        while True:
            data = np.ones([self.batch_size, self.img_w, self.img_h, 1])
            labels = np.ones([self.batch_size, self.max_text_len])
            input_length = np.ones((self.batch_size, 1)) * (self.img_w // self.downsample_factor - 2)
            label_length = np.zeros((self.batch_size, 1))

            for i in range(self.batch_size):
                image, number = self.next_sample()
                image = cv2.resize(image, (self.img_w, self.img_h))
                image = image.reshape(self.img_w, self.img_h, 1)

                data[i] = image
                text_length = len(number)
                labels[i, 0:text_length] = self.text_to_labels(number)
                label_length[i] = len(number)

            inputs = {
                'data': data,
                'labels': labels,
                'data_length': input_length,
                'label_length': label_length
            }
            outputs = {'ctc': np.zeros([self.batch_size])}
            yield (inputs, outputs)

    @staticmethod
    def text_to_labels(text):
        return list(map(lambda x: letters.index(x), text))
