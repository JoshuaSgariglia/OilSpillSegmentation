from enum import Enum
import os
from dataclass import DatasetPaths, Parameters, ParametersValues

# Datetime format
DATETIME_FORMAT: str = "%Y-%m-%d_%H-%M-%S"

# Logging
LOG_FILENAME: str = "log"

# Input-Output parameters
INPUT_WIDTH = 256
INPUT_HEIGHT = 256
INP_CHANNELS = 1
OUT_MASKS = 1

    
# Parameters registry for different models
class ParametersRegistry:
    
    # Take parameters from model
    AUTOMATIC = None
    
    # Default parameters
    DEFAULT_PARAMETERS = Parameters(
        BATCH_SIZE = 8,
        DECAYING_FACTOR = 0.5,
        PATIENCE = 3,
        EPOCHS = 50,
        LR = 1e-4,
        MIN_LR = 1e-6
    )
    
    # Session test parameters
    SESSION_TEST_PARAMETERS = Parameters(
        BATCH_SIZE = 4,
        DECAYING_FACTOR = 0.5,
        PATIENCE = 3,
        EPOCHS = 1,
        LR = 1e-4,
        MIN_LR = 1e-6
    )
    
    # Default parameter values for automatic training
    DEFAULT_PARAMETERS_VALUES = ParametersValues(
        BATCH_SIZE_VALUES = (8, 16, 32),
        DECAYING_FACTOR_VALUES = (0.5, 0.4),
        PATIENCE_VALUES = (3, 4),
        EPOCHS = DEFAULT_PARAMETERS.EPOCHS,
        LR = DEFAULT_PARAMETERS.LR,
        MIN_LR = DEFAULT_PARAMETERS.MIN_LR
    )
    
    # Dataset predefined paths
    UNET = ParametersValues(
        BATCH_SIZE_VALUES = (8, 16, 32),
        DECAYING_FACTOR_VALUES = (0.5, 0.4),
        PATIENCE_VALUES = (3, 4),
        EPOCHS = 50,
        LR = 5e-5,
        MIN_LR = 1e-6
    )
    
    UNETL = ParametersValues(
        BATCH_SIZE_VALUES = (8, 16, 32),
        DECAYING_FACTOR_VALUES = (0.5, 0.4),
        PATIENCE_VALUES = (3, 4),
        EPOCHS = 50,
        LR = 1e-4,
        MIN_LR = 1e-6
    )
    
    UNETPP = ParametersValues(
        BATCH_SIZE_VALUES = (8, 16),
        DECAYING_FACTOR_VALUES = (0.5, 0.4),
        PATIENCE_VALUES = (3, 2),
        EPOCHS = 40,
        LR = 2.5e-5,
        MIN_LR = 1e-7
    )
    
    UNETPPL = ParametersValues(
        BATCH_SIZE_VALUES = (8, 16),
        DECAYING_FACTOR_VALUES = (0.5, 0.4),
        PATIENCE_VALUES = (3, 2),
        EPOCHS = 40,
        LR = 5e-5,
        MIN_LR = 5e-7
    )
    
    TRANSUNET = ParametersValues(
        BATCH_SIZE_VALUES = (8, 12),
        DECAYING_FACTOR_VALUES = (0.5, 0.3),
        PATIENCE_VALUES = (3, 2),
        EPOCHS = 40,
        LR = 1e-4,
        MIN_LR = 5e-7
    )
    
    LIGHTMUNET = ParametersValues(
        BATCH_SIZE_VALUES = (4, ),
        DECAYING_FACTOR_VALUES = (0.5, 0.3),
        PATIENCE_VALUES = (3, 2),
        EPOCHS = 40,
        LR = 1e-5,
        MIN_LR = 1e-7
    )

# Names of files created during training and validation
class SaveFilename(Enum):
    EMISSIONS = "emissions.json"
    EVALUATION = "evaluation.json"
    MODEL = "model.hdf5"
    MODEL_TF = "model.tf"
    PARAMETERS = "parameters.json"
    TIME = "time.json"
    TRAINING = "training.csv"

# Main project paths
class Paths:
    DATASETS = os.path.join(os.getcwd(), "sos-dataset/dataset")
    DATASET_DENOISED = os.path.join(os.getcwd(), "sos-dataset/denoised")
    DENOISING = os.path.join(os.getcwd(), "module_test/denoising")
    EMISSIONS = os.path.join(os.getcwd(), "emissions")
    LOGS = os.path.join(os.getcwd(), "logs")
    PREDICTIONS = os.path.join(os.getcwd(), "module_test/prediction")
    SAVES = os.path.join(os.getcwd(), "saves")

# Dataset registry for different datasets
class DatasetRegistry:

    # Dataset predefined paths
    PALSAR = DatasetPaths( 
        "Palsar",
        os.path.join(Paths.SAVES, "palsar"),
        os.path.join(Paths.DATASETS, "train/palsar/image"),
        os.path.join(Paths.DATASETS, "train/palsar/label"),
        os.path.join(Paths.DATASETS, "test/palsar/image"),
        os.path.join(Paths.DATASETS, "test/palsar/label"),
        os.path.join(Paths.DATASET_DENOISED, "train/palsar/image"),
        os.path.join(Paths.DATASET_DENOISED, "test/palsar/image")
    )
    
    SENTINEL = DatasetPaths(
        "Sentinel",
        os.path.join(Paths.SAVES, "sentinel"),
        os.path.join(Paths.DATASETS, "train/sentinel/image"),
        os.path.join(Paths.DATASETS, "train/sentinel/label"),
        os.path.join(Paths.DATASETS, "test/sentinel/image"),
        os.path.join(Paths.DATASETS, "test/sentinel/label"),
        os.path.join(Paths.DATASET_DENOISED, "train/sentinel/image"),
        os.path.join(Paths.DATASET_DENOISED, "test/sentinel/image")
    )
    