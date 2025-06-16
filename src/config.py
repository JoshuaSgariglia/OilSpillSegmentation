from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
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

class Parameters:
    
    def __init__(self, 
        TRAIN_BATCH_SIZE: int = 8,
        VAL_BATCH_SIZE: int = 4,
        DECAYING_FACTOR: float = 0.5,
        PATIENCE: int = 3,
        EPOCHS: int = 1,
        DROPOUT_RATE: float = 0.2,
        MOMENTUM: float = 0.98,
        LR: float = 1e-4,
        MIN_LR: float = 1e-6,        
    ):
        self.TRAIN_BATCH_SIZE = TRAIN_BATCH_SIZE
        self.VAL_BATCH_SIZE = VAL_BATCH_SIZE
        self.LR = LR
        self.MIN_LR = MIN_LR
        self.DECAYING_FACTOR = DECAYING_FACTOR
        self.MOMENTUM = MOMENTUM
        self.PATIENCE = PATIENCE
        self.EPOCHS = EPOCHS
        self.DROPOUT_RATE = DROPOUT_RATE
        
    def Generate() -> list[Parameters]:
        parameters_list: list[Parameters] = []
        for train_batch_size in (4, 8):
            for val_batch_size in (4, 8):
                for decaying_factor in (0.5, 0.4):
                    for patience in (3, 4):
                        parameters_list.append(Parameters(
                            train_batch_size, 
                            val_batch_size, 
                            decaying_factor, 
                            patience
                            ))
        
        return parameters_list

class Paths:
    # Common paths
    SAVES_PATH = os.path.join(os.getcwd(), "saves")
    
    # Get current datetime
    def current_datetime() -> str:
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Generate directory name for specific trained model
    @classmethod
    def generate_model_dir_name(cls, model_name: str) -> str:
        return f"{model_name}_{cls.current_datetime()}"
    
    # Get path for saves of a type of model
    @classmethod
    def get_model_type_saves_path(cls, model_name: str) -> str:
        return os.path.join(os.getcwd(), cls.SAVES_PATH, model_name)
    
    # Get path for a specific save of model
    @classmethod
    def get_model_saves_path(cls, model_name: str) -> str:
        return os.path.join(os.getcwd(), cls.SAVES_PATH, model_name, cls.generate_model_dir_name(model_name))
    

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
    