# import the necessary packages

import h5py
import numpy as np
from config import anpr_config as config


class HDF5DatasetGenerator:
    def __init__(self, dbPath, batch_size, preprocessors=None, aug=None):
        # store the batch size, preprocessors, and data augmentor,
        # whether or not the labels should be binarized, along with
        # the total number of classes
        self.batch_size = batch_size
        self.preprocessors = preprocessors
        self.aug = aug

        # open the HDF5 database for reading and determine the total
        # number of entries in the database
        self.db = h5py.File(dbPath)
        self.numImages = self.db["labels"].shape[0]

        self.img_w = 160
        self.img_h = 32
        self.downsample_factor = 2
        self.max_text_len = 9

    def generator(self, passes=np.inf):
        # initialize the epoch count
        epochs = 0

        # keep looping infinitely -- the model will stop once we have
        # reach the desired number of epochs
        while epochs < passes:

            data_length = np.ones((self.batch_size, 1)) * (self.img_w // self.downsample_factor - 2)
            label_length = np.ones((self.batch_size, 1))
            labels = np.ones([self.batch_size, self.max_text_len])

            # loop over the HDF5 images
            for i in np.arange(0, self.numImages, self.batch_size):
                # extract the images and labels from the HDF images
                images = self.db["images"][i: i + self.batch_size]
                numbers = self.db["labels"][i: i + self.batch_size]

                # check to see if our preprocessors are not None
                if self.preprocessors is not None:
                    # initialize the list of processed images
                    procImages = []

                    # loop over the images
                    for j, image in enumerate(images):
                        # loop over the preprocessors and apply each
                        # to the image
                        for p in self.preprocessors:
                            image = p.preprocess(image)

                        # update the list of processed images
                        procImages.append(image)
                        number = numbers[j]
                        labels[j] = self.text_to_labels(number) + self.fill_blanks(len(numbers[j]))
                        label_length[j] = len(number)

                    # update the images array to be the processed
                    # images
                    images = np.array(procImages)

                # if the data augmenator exists, apply it
                # if self.aug is not None:
                #     (images, labels) = next(self.aug.flow(images,
                #                                           labels, batch_size=self.batch_size))

                inputs = {
                    'data': images,
                    'labels': labels,
                    'data_length': data_length,
                    'label_length': label_length
                }
                outputs = {'ctc': np.zeros([self.batch_size])}
                # yield a tuple of images and labels
                yield (inputs, outputs)

            # increment the total number of epochs
            epochs += 1

    def close(self):
        # close the database
        self.db.close()

    @staticmethod
    def text_to_labels(text):
        return list(map(lambda x: config.LETTERS.index(x), text))

    def fill_blanks(self, text_len):
        text = []
        for n in range(self.max_text_len - text_len):
            text.append(config.LETTERS.index(" "))
        return text
