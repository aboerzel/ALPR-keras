import os
import random
import cv2
import numpy as np
from keras import backend as K

letters = sorted(list("ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜ0123456789 "))


class DatasetGenerator:
    def __init__(self,
                 img_dirpath,
                 img_w, img_h,
                 pool_size,
                 batch_size,
                 max_text_len=8):

        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size
        self.max_text_len = max_text_len
        self.downsample_factor = pool_size ** 2
        self.samples = []
        for filename in os.listdir(img_dirpath):
            name, ext = os.path.splitext(filename)
            img_filepath = img_dirpath + "/" + filename
            self.samples.append([img_filepath, name.replace("-", "")])

        self.n = len(self.samples)
        self.indexes = list(range(self.n))
        self.cur_index = 0

    def build_data(self):
        self.imgs = np.zeros((self.n, self.img_h, self.img_w))
        self.texts = []
        for i, (img_filepath, text) in enumerate(self.samples):

            # stream = open(img_filepath, "rb")
            # bytes = bytearray(stream.read())
            # numpyarray = np.asarray(bytes, dtype=np.uint8)
            # img = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
            try:
                stream = open(img_filepath, "rb")
                bytes = bytearray(stream.read())
                numpyarray = np.asarray(bytes, dtype=np.uint8)
                img = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
                # img = cv2.imread(os.getcwd() + "/" + img_filepath)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (self.img_w, self.img_h))
                img = img.astype(np.float32)
                img /= 255
                # width and height are backwards from typical Keras convention
                # because width is the time dimension when it gets fed into the RNN
                self.imgs[i, :, :] = img

                while len(text) < 8:
                    text += " "

                self.texts.append(text)
            except:
                continue

    @staticmethod
    def get_output_size():
        return len(letters) + 1

    def next_sample(self):
        self.cur_index += 1
        if self.cur_index >= self.n:
            self.cur_index = 0
            random.shuffle(self.indexes)
        return self.imgs[self.indexes[self.cur_index]], self.texts[self.indexes[self.cur_index]]

    def next_batch(self):
        while True:
            # width and height are backwards from typical Keras convention
            # because width is the time dimension when it gets fed into the RNN
            if K.image_data_format() == 'channels_first':
                X_data = np.ones([self.batch_size, 1, self.img_w, self.img_h])
            else:
                X_data = np.ones([self.batch_size, self.img_w, self.img_h, 1])
            Y_data = np.ones([self.batch_size, self.max_text_len])
            input_length = np.ones((self.batch_size, 1)) * (self.img_w // self.downsample_factor - 2)
            label_length = np.zeros((self.batch_size, 1))
            source_str = []

            for i in range(self.batch_size):
                img, text = self.next_sample()
                img = img.T
                if K.image_data_format() == 'channels_first':
                    img = np.expand_dims(img, 0)
                else:
                    img = np.expand_dims(img, -1)
                X_data[i] = img
                Y_data[i] = self.text_to_labels(text)
                source_str.append(text)
                label_length[i] = len(text)

            inputs = {
                'the_input': X_data,
                'the_labels': Y_data,
                'input_length': input_length,
                'label_length': label_length,
                # 'source_str': source_str
            }
            outputs = {'ctc': np.zeros([self.batch_size])}
            yield (inputs, outputs)

    def text_to_labels(self, text):
        return list(map(lambda x: letters.index(x), text))
