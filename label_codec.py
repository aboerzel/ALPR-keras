import itertools
import numpy as np


class LabelCodec:
    ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜ0123456789- "

    # Translation of characters to unique numerical classes
    @staticmethod
    def encode_number(number):
        return list(map(lambda c: LabelCodec.ALPHABET.index(c), number))

    # Reverse translation of numerical classes back to characters
 #   @staticmethod
 #   def decode_number(labels):
 #       ret = []
 #       for c in labels:
 #           if c == len(LabelCodec.ALPHABET):  # CTC Blank
 #               ret.append("")
 #           else:
 #               ret.append(LabelCodec.ALPHABET[c])
 #       return "".join(ret)

    @staticmethod
    def decode_number(out):
        # out : (1, 32, 42)
        # the 2 is critical here since the first couple outputs of the RNN tend to be garbage:
        out_best = list(np.argmax(out[0, 2:], axis=1))  # get max index -> len = 32
        out_best = [k for k, g in itertools.groupby(out_best)]  # remove overlap value
        outstr = ''
        for c in out_best:
            if c < len(LabelCodec.ALPHABET):
                outstr += LabelCodec.ALPHABET[c]
        return outstr
