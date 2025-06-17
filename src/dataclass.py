from dataclasses import dataclass

# Parameters model classes
@dataclass
class Parameters:
    BATCH_SIZE: int
    DECAYING_FACTOR: float
    PATIENCE: int
    EPOCHS: int
    LR: float
    MIN_LR: float

@dataclass
class ParametersValues:
    BATCH_SIZE_VALUES: tuple[int]
    DECAYING_FACTOR_VALUES: tuple[float]
    PATIENCE_VALUES: tuple[int]
    EPOCHS: int
    LR: float
    MIN_LR: float
    
    
# DatasetPaths model class
@dataclass
class DatasetPaths:
    MODEL_SAVES_PATH: str
    TRAIN_IMAGES_PATH: str
    TRAIN_LABELS_PATH: str
    TEST_IMAGES_PATH: str
    TEST_LABELS_PATH: str
    TRAIN_IMAGES_DENOISED_PATH: str
    TEST_IMAGES_DENOISED_PATH: str