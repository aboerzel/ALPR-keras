# import the necessary packages
from os import path

# root paths
PROJECT_ROOT_PATH = "D:/development/cv/ANPR-keras"
DATASET_ROOT_PATH = "D:/development/datasets/alpr"

# network image size
IMAGE_WIDTH = 160
IMAGE_HEIGHT = 32

# training parameters
BATCH_SIZE = 128
POOL_SIZE = 2
NUM_EPOCHS = 50
START_EPOCH = 0

# license number construction
MAX_TEXT_LEN = 9
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜ0123456789- "

# model name
MODEL_NAME = "alpr.model"

# json file with the list of german county marks
GERMAN_COUNTY_MARKS = "./config/german_county_marks.json"

# define the paths to the training and validation directories
TRAIN_IMAGES = path.sep.join([DATASET_ROOT_PATH, "images"])
VAL_IMAGES = path.sep.join([DATASET_ROOT_PATH, "val"])

# define the path to the output training, validation, and testing HDF5 files
TRAIN_HDF5 = path.sep.join([DATASET_ROOT_PATH, "hdf5/train.h5"])
VAL_HDF5 = path.sep.join([DATASET_ROOT_PATH, "hdf5/val.h5"])
TEST_HDF5 = path.sep.join([DATASET_ROOT_PATH, "hdf5/test.h5"])

# define the path to the output directory used for storing plots, classification reports, etc.
OUTPUT_PATH = "output"

# path to the output model file
MODEL_PATH = path.sep.join([OUTPUT_PATH, MODEL_NAME]) + ".h5"
MODEL_CHECKPOINT_PATH = path.sep.join([OUTPUT_PATH, MODEL_NAME]) + '.{epoch:02d}.hdf5'
FIG_PATH = path.sep.join([OUTPUT_PATH, "alpr.png"])
JSON_PATH = path.sep.join([OUTPUT_PATH, "alpr.json"])
CHECKPOINTS_PATH = path.sep.join([OUTPUT_PATH, "checkpoints"])
TENSORBOARD_PATH = path.sep.join([OUTPUT_PATH, "tensorboard"])