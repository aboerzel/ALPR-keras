import sys
import tarfile
import progressbar
import cv2
import numpy

from pyimagesearch.io import HDF5DatasetWriter

IMAGE_WIDTH = 151
IMAGE_HEIGHT = 32


def im_from_file(f):
    a = numpy.asarray(bytearray(f.read()), dtype=numpy.uint8)
    # cv2.imdecode(a, cv2.CV_CV_LOAD_IMAGE_GRAYSCALE)
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
            im = im_from_file(f)
        finally:
            f.close()
        if im is None:
            continue

        if im.shape[0] > im.shape[1]:
            im = im[:im.shape[1], :]
        else:
            im = im[:, :im.shape[0]]
        if im.shape[0] > 256:
            im = cv2.resize(im, (256, 256))

        fname = "{:08}.jpg".format(index)
        # print(fname)

        # add the image and label # to the HDF5 images
        writer.add([im], [fname])
        pbar.update(i)

        index += 1

    # close the HDF5 writer
    pbar.finish()
    writer.close()


if __name__ == "__main__":
    extract_backgrounds(sys.argv[1])
