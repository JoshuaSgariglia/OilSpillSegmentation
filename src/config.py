from dataclasses import dataclass
from enum import Enum
import os

# Input-Output parameters
INPUT_WIDTH = 256
INPUT_HEIGHT = 256
INP_CHANNELS = 1
OUT_MASKS = 1

# Default parameters
TRAIN_BATCH_SIZE: int = 8
VAL_BATCH_SIZE: int = 4
DECAYING_FACTOR: float = 0.5
PATIENCE: int = 3
EPOCHS: int = 1
DROPOUT_RATE: float = 0.2
LR: float = 1e-4
MIN_LR: float = 1e-6
MOMENTUM: float = 0.98

# Default parameter sets for automatic training
TRAIN_BATCH_SIZE_VALUES: tuple = (4, 8)
VAL_BATCH_SIZE_VALUES: tuple = (4, 8)
DECAYING_FACTOR_VALUES: tuple = (0.5, 0.4)
PATIENCE_VALUES: tuple = (3, 4)

# Datetime format
DATETIME_FORMAT: str = "%Y-%m-%d_%H-%M-%S"

# Logging
LOG_FILENAME: str = "log"

# Names of files created during training and validation
class SaveFilename(Enum):
    EVALUATION = "evaluation.json"
    MODEL = "model.hdf5"
    PARAMETERS = "parameters.json"
    TIME = "time.json"
    TRAINING = "training.csv"

# Main project paths
class Paths:
    DATASETS = os.path.join(os.getcwd(), "sos-dataset/dataset")
    LOGS = os.path.join(os.getcwd(), "logs")
    PREDICTIONS = os.path.join(os.getcwd(), "prediction_test")
    SAVES = os.path.join(os.getcwd(), "saves")
    
# Interface for DatasetRegistry attributes
@dataclass
class DatasetPaths:
    TRAIN_IMAGES_PATH: str
    TRAIN_LABELS_PATH: str
    TEST_IMAGES_PATH: str
    TEST_LABELS_PATH: str

# Dataset registry for different datasets
class DatasetRegistry:
    
    # Dataset predefined paths
    PALSAR = DatasetPaths(
        os.path.join(os.getcwd(), Paths.DATASETS, "train/palsar/image/"),
        os.path.join(os.getcwd(), Paths.DATASETS, "train/palsar/label/"),
        os.path.join(os.getcwd(), Paths.DATASETS, "test/palsar/image/"),
        os.path.join(os.getcwd(), Paths.DATASETS, "test/palsar/label/")
    )
    SENTINEL = DatasetPaths(
        os.path.join(os.getcwd(), Paths.DATASETS, "train/sentinel/image/"),
        os.path.join(os.getcwd(), Paths.DATASETS, "train/sentinel/label/"),
        os.path.join(os.getcwd(), Paths.DATASETS, "test/sentinel/image/"),
        os.path.join(os.getcwd(), Paths.DATASETS, "test/sentinel/label/")
    )
    