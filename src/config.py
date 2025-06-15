from dataclasses import dataclass
import os

# Config parameters
INPUT_WIDTH = 256
INPUT_HEIGHT = 256
INP_CHANNELS = 1
OUT_MASKS = 1

# Training parameters
TRAIN_BATCH_SIZE = 8
VAL_BATCH_SIZE = 4
LR = 1e-4
MIN_LR = 1e-6
DECAYING_FACTOR = 0.5
MOMENTUM = 0.98
PATIENCE = 3
EPOCHS = 60
DROPOUT_RATE = 0.2


# Interface for DatasetRegistry attributes
@dataclass(frozen=True)
class DatasetPaths:
    TRAIN_IMAGES_PATH: str
    TRAIN_LABELS_PATH: str
    TEST_IMAGES_PATH: str
    TEST_LABELS_PATH: str
    TEST_PREDICTS_PATH: str

# Dataset registry for different datasets
class DatasetRegistry:
    PALSAR = DatasetPaths(
        os.path.join(os.getcwd(), "sos-dataset/dataset/train/palsar/image/"),
        os.path.join(os.getcwd(), "sos-dataset/dataset/train/palsar/label/"),
        os.path.join(os.getcwd(), "sos-dataset/dataset/test/palsar/image/"),
        os.path.join(os.getcwd(), "sos-dataset/dataset/test/palsar/label/"),
        os.path.join(os.getcwd(), "sos-dataset/dataset/test/palsar/predict/")
    )
    SENTINEL = DatasetPaths(
        os.path.join(os.getcwd(), "sos-dataset/dataset/train/sentinel/image/"),
        os.path.join(os.getcwd(), "sos-dataset/dataset/train/sentinel/label/"),
        os.path.join(os.getcwd(), "sos-dataset/dataset/test/sentinel/image/"),
        os.path.join(os.getcwd(), "sos-dataset/dataset/test/sentinel/label/"),
        os.path.join(os.getcwd(), "sos-dataset/dataset/test/sentinel/predict/")
    )
    