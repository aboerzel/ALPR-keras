import os
import random
import cv2
import numpy as np
from keras import backend as K


class LicensePlateDatasetLoader:
    #letters = sorted(list("ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜ0123456789- "))
    samples = []
    max_text_len = 0
    indexes = []
    cur_index = 0

    def __init__(self,
                 img_w, img_h,
                 pool_size,
                 batch_size,
                 preprocessors=None):

        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size
        self.preprocessors = preprocessors
        self.downsample_factor = pool_size ** 2

        if self.preprocessors is None:
            self.preprocessors = []

    # def _is_valid(self, text):
    #     for c in text:
    #         if c not in self.letters:
    #             return False
    #     return True

    def load(self, imagedir):
        filepaths = list(os.listdir(imagedir))
        self.max_text_len = max(len(os.path.splitext(f)[0]) for f in filepaths)

        for i, filename in enumerate(filepaths):
            label, ext = os.path.splitext(filename)

            # skip not valid labeltexts
            # if not self._is_valid(label):
            #     continue

            filepath = os.path.join(imagedir, filename)

            try:
                stream = open(filepath, "rb")
                bytes = bytearray(stream.read())
                numpyarray = np.asarray(bytes, dtype=np.uint8)
                image = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image, (self.img_w, self.img_h))
                image = image.astype(np.float32)
                image /= 255

                while len(label) < self.max_text_len:
                    label += " "

                # add original image
                self.samples.append([image, label])

                # data augmentation
                for p in self.preprocessors:
                    for n in range(10):
                        image = p.preprocess(image)
                        self.samples.append([image, label])
            except:
                continue  # skip corrupt files

        self.indexes = list(range(len(self.samples)))

    def _text_to_labels(self, text):
        return list(map(lambda x: self.letters.index(x), text))

    def get_classes(self):
        return len(self.letters) + 1

    def get_data_size(self):
        return len(self.samples)

    def get_max_text_len(self):
        return self.max_text_len

    def _next_sample(self):
        if self.cur_index >= len(self.indexes):
            self.cur_index = 0
            random.shuffle(self.indexes)

        sample = self.samples[self.indexes[self.cur_index]]
        self.cur_index += 1
        return sample

    def next_batch(self):
        while True:
            # width and height are backwards from typical Keras convention
            # because width is the time dimension when it gets fed into the RNN
            if K.image_data_format() == 'channels_first':
                x_data = np.ones([self.batch_size, 1, self.img_w, self.img_h])
            else:
                x_data = np.ones([self.batch_size, self.img_w, self.img_h, 1])

            y_data = np.ones([self.batch_size, self.max_text_len])
            data_length = np.ones((self.batch_size, 1)) * (self.img_w // self.downsample_factor - 2)
            label_length = np.zeros((self.batch_size, 1))

            for i in range(self.batch_size):
                img, text = self._next_sample()
                img = img.T
                if K.image_data_format() == 'channels_first':
                    img = np.expand_dims(img, 0)
                else:
                    img = np.expand_dims(img, -1)

                x_data[i] = img
                y_data[i] = self._text_to_labels(text)
                label_length[i] = len(text)

            inputs = {
                'data': x_data,
                'labels': y_data,
                'data_length': data_length,
                'label_length': label_length
            }
            outputs = {'ctc': np.zeros([self.batch_size])}
            yield (inputs, outputs)
