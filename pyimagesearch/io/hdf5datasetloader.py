import h5py
import numpy as np


class Hdf5DatasetLoader:
    def __init__(self, shuffle=False, max_items=np.inf, preprocessors=None):
        self.shuffle = shuffle
        self.max_items = max_items
        self.preprocessors = preprocessors

        # if the preprocessors are None, initialize them as an empty list
        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, db_path):

        db = h5py.File(db_path)
        images = np.array(db["images"])
        labels = np.array(db["labels"])
        db.close()

        if self.shuffle:
            randomized_indexex = np.arange(len(images))
            np.random.shuffle(randomized_indexex)
            images = images[randomized_indexex]
            labels = labels[randomized_indexex]

        if self.max_items != np.inf and self.max_items <= len(images):
            self.max_items = len(images)
            images = images[0:self.max_items]
            labels = labels[0:self.max_items]

        # preprocess images
        for i, (image, label) in enumerate(zip(images, labels)):

            for p in self.preprocessors:
                image = p.preprocess(image)
                images[i] = image

        return images, labels
