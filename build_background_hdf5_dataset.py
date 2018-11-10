import tarfile

import cv2
import numpy
import progressbar

from config import alpr_config as config
from pyimagesearch.io import HDF5DatasetWriter

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256


def im_from_file(f):
    a = numpy.asarray(bytearray(f.read()), dtype=numpy.uint8)
    return cv2.imdecode(a, cv2.IMREAD_GRAYSCALE)


def extract_backgrounds(archive_name, output_path):
    print("[INFO] reading content of {}...".format(archive_name))
    tar = tarfile.open(name=archive_name)
    files = tar.getnames()

    print("[INFO] building {}...".format(output_path))
    writer = HDF5DatasetWriter((len(files), IMAGE_HEIGHT, IMAGE_WIDTH), output_path)

    widgets = ["Building Dataset: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(files), widgets=widgets).start()
    index = 0

    for i, file in enumerate(files):
        f = tar.extractfile(file)
        if f is None:
            continue
        try:
            image = im_from_file(f)
        finally:
            f.close()
        if image is None:
            continue

        if image.shape[0] > image.shape[1]:
            image = image[:image.shape[1], :]
        else:
            image = image[:, :image.shape[0]]

        if image.shape[0] != 256:
            image = cv2.resize(image, (256, 256))

        name = "{:08}".format(index)

        # check image size
        if not image.shape == (IMAGE_HEIGHT, IMAGE_WIDTH):
            print("image with wrong size: %s" % name)
            continue

        # add the image and name to the HDF5 db
        writer.add([image], [name])
        pbar.update(i)

        index += 1

    # close the HDF5 writer
    pbar.finish()
    writer.close()


if __name__ == "__main__":
    extract_backgrounds(config.SUN397_TAR_FILE, config.SUN397_HDF5)
