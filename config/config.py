# import the necessary packages
from os import path

# root paths
DATASET_ROOT_PATH = "D:/development/tensorflow/datasets/clpr"
SUN397_TAR_FILE = "D:/development/tensorflow/datasets/SUN397.tar.gz"

# network image size
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 64

POOL_SIZE = 2

# training parameter
BATCH_SIZE = 100
NUM_EPOCHS = 1000

# license number construction
MAX_TEXT_LEN = 10

# model name
MODEL_NAME = "clpr-model"

# define the path to the output directory used for storing plots, classification reports, etc.
OUTPUT_PATH = "output"

# json file with the list of german county marks
GERMAN_COUNTY_MARKS = "./config/german_county_marks.json"

# define the paths to the training and validation directories
TRAIN_IMAGES = path.sep.join([DATASET_ROOT_PATH, "images"])

# define the path to the output training, validation, and testing HDF5 files
TRAIN_HDF5 = path.sep.join([DATASET_ROOT_PATH, "hdf5/train.h5"])
TEST_HDF5 = path.sep.join([DATASET_ROOT_PATH, "hdf5/test.h5"])
SUN397_HDF5 = path.sep.join([DATASET_ROOT_PATH, "hdf5/background.h5"])
