# import the necessary packages
from os import path

IMAGE_WIDTH = 160
IMAGE_HEIGHT = 32
IMAGE_DEPTH = 1

# initialize the batch size and number of epochs for training
BATCH_SIZE = 128
POOL_SIZE = 2
NUM_EPOCHS = 10
START_EPOCH = 0

MAX_TEXT_LEN = 9
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜ0123456789- "

# define the paths to the training and validation directories
TRAIN_IMAGES = "../datasets/anpr/train"
VAL_IMAGES = "../datasets/anpr/val/images"

# json file with the list of german county marks
GERMAN_COUNTY_MARKS = "./config/german_county_marks.json"

# define the path to the output training, validation, and testing
# HDF5 files
TRAIN_HDF5 = "../datasets/anpr/hdf5/train.hdf5"
VAL_HDF5 = "../datasets/anpr/hdf5/val.hdf5"
TEST_HDF5 = "../datasets/anpr/hdf5/test.hdf5"

# define the path to the output directory used for storing plots,
# classification reports, etc.
OUTPUT_PATH = "output"
# path to the output model file
MODEL_PATH = path.sep.join([OUTPUT_PATH, "anpr.model.hdf5"])
# define the path to the images mean
DATASET_MEAN = path.sep.join([OUTPUT_PATH, "anpr-mean.json"])
FIG_PATH = path.sep.join([OUTPUT_PATH, "anpr.png"])
JSON_PATH = path.sep.join([OUTPUT_PATH, "anpr.json"])
CHECKPOINTS =  path.sep.join([OUTPUT_PATH, "checkpoints"])

