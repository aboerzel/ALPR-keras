import sys
import tarfile
import progressbar
import cv2
import numpy

from pyimagesearch.io import HDF5DatasetWriter

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256


def im_from_file(f):
    a = numpy.asarray(bytearray(f.read()), dtype=numpy.uint8)
    return cv2.imdecode(a, cv2.IMREAD_GRAYSCALE)


def extract_backgrounds(archive_name, outputPath):
    t = tarfile.open(name=archive_name)

    def members():
        m = t.next()
        while m:
            yield m
            m = t.next()

    index = 0

    members = members()

    # create HDF5 writer
    print("[INFO] building {}...".format(outputPath))
    writer = HDF5DatasetWriter((len(members), IMAGE_HEIGHT, IMAGE_WIDTH), outputPath)

    # initialize the progress bar
    widgets = ["Building Dataset: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(members), widgets=widgets).start()

    for i, m in enumerate(members):
        # if not m.name.endswith(".jpg"):
        #     continue
        f = t.extractfile(m)
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

        # if image.shape[0] > 256:
        #     image = cv2.resize(image, (256, 256))

        image = cv2.resize(image, (256, 256))

        name = "{:08}".format(index)

        # add the image and name # to the HDF5 images
        writer.add([image], [name])
        pbar.update(i)

        index += 1

    # close the HDF5 writer
    pbar.finish()
    writer.close()


if __name__ == "__main__":
    extract_backgrounds(sys.argv[1])
