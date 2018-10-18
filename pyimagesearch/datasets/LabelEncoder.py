from config import anpr_config as config
import numpy as np


class LabelEncoder:
    def __init__(self, max_text_len):
        self.max_text_len = max_text_len

    def fit_transform(self, texts):
        labels = []
        for text in texts:
            labels.append(np.asarray(self.text_to_labels(text) + self.fill_blanks(len(text))))
        return np.array(labels)

    def fill_blanks(self, text_len):
        text = []
        for n in range(self.max_text_len - text_len):
            text.append(config.ALPHABET.index(" "))
        return text

    @staticmethod
    def text_to_labels(text):
        return list(map(lambda x: config.ALPHABET.index(x), text))

    @staticmethod
    def labels_to_text(labels):
        outstr = ''
        for c in labels:
            outstr += config.ALPHABET[c]
        return outstr

# letters = sorted(list("ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜ0123456789- "))
