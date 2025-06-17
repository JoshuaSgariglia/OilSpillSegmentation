from __future__ import annotations
from datetime import datetime
import os
import logging
import cv2
from tensorflow.keras.losses import Loss, BinaryCrossentropy # type: ignore
import numpy as np
import tensorflow as tf
from config import DATETIME_FORMAT, DECAYING_FACTOR, DECAYING_FACTOR_VALUES, DROPOUT_RATE, EPOCHS, LOG_FILENAME, LR, MIN_LR, PATIENCE, PATIENCE_VALUES, TRAIN_BATCH_SIZE, TRAIN_BATCH_SIZE_VALUES, VAL_BATCH_SIZE, VAL_BATCH_SIZE_VALUES, Paths
from numpy.typing import NDArray
from numpy import float32



    

# Losses
class DiceLoss(Loss):
  def __init__(self):
    super().__init__()
    self.redsum = tf.math.reduce_sum

  def call(self, y_true, y_pred, smooth=0.01):
    intersec = self.redsum(y_true * y_pred)
    dice = (2 * intersec + smooth) / (self.redsum(y_true) + self.redsum(y_pred) + smooth)

    return 1 - dice
  
class BCEDiceLoss(Loss):
    def __init__(self):
        super().__init__()
        self.bce = BinaryCrossentropy(from_logits=False)
        self.dice = DiceLoss()

    def call(self, y_true, y_pred):
        bce_loss = self.bce(y_true, y_pred)
        dice_loss = self.dice(y_true, y_pred)
        return (bce_loss + dice_loss) / 2.0
    
    
# Logger
def setup_logger(level: int = logging.INFO):
    """Set up a logger that logs to both console and file."""
    
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.propagate = False  # Prevent log duplication if multiple handlers
    
    # Formatter for log messages
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Create directory for logs if it doesn't exist
    os.makedirs(Paths.LOGS, exist_ok=True)
    
    # Log file path
    log_file_path = generate_logging_path()
    
    # File handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    
    # Add handlers if not already added
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    
    # First log
    logger.info(f"Start of log - file saved at \"{log_file_path}\"")
    
    return logger

def generate_logging_path():
    return os.path.join(Paths.LOGS, f"{LOG_FILENAME}_{current_datetime()}.txt")

# GPU \ System
def config_gpu():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" # GPU with ID 0

    gpus = tf.config.list_physical_devices(device_type='GPU')

    for gpu in gpus:
        # GPU memory allocation expands as needed
        tf.config.experimental.set_memory_growth(device=gpu, enable=True)


# Get current datetime
def current_datetime() -> str:
    return datetime.now().strftime(DATETIME_FORMAT)

# Parameter model class
class Parameters:
    def __init__(self, 
        TRAIN_BATCH_SIZE: int = TRAIN_BATCH_SIZE,
        VAL_BATCH_SIZE: int = VAL_BATCH_SIZE,
        DECAYING_FACTOR: float = DECAYING_FACTOR,
        PATIENCE: int = PATIENCE,
        EPOCHS: int = EPOCHS,
        DROPOUT_RATE: float = DROPOUT_RATE,   
        LR: float = LR,
        MIN_LR: float = MIN_LR,     
    ):
        self.TRAIN_BATCH_SIZE = TRAIN_BATCH_SIZE
        self.VAL_BATCH_SIZE = VAL_BATCH_SIZE
        self.DECAYING_FACTOR = DECAYING_FACTOR
        self.PATIENCE = PATIENCE
        self.EPOCHS = EPOCHS
        self.DROPOUT_RATE = DROPOUT_RATE
        self.LR = LR
        self.MIN_LR = MIN_LR
    
    @classmethod
    def generate(cls) -> list[Parameters]:
        parameters_list: list[Parameters] = []
        for train_batch_size in TRAIN_BATCH_SIZE_VALUES:
            for val_batch_size in VAL_BATCH_SIZE_VALUES:
                for decaying_factor in DECAYING_FACTOR_VALUES:
                    for patience in PATIENCE_VALUES:
                        parameters_list.append(cls(
                            train_batch_size, 
                            val_batch_size, 
                            decaying_factor, 
                            patience
                            ))
        
        return parameters_list
    
# Model classes

# Evaluation model class
class Evaluation:
    def __init__(self, 
        confusion_matrix: NDArray,
        accuracy: float,
        precision_values: NDArray,
        recall_values: NDArray,
        f1_score_values: NDArray,
         
    ):
        self.CONFUSION_MATRIX = confusion_matrix.tolist()
        self.ACCURACY = accuracy
    
    @classmethod
    def generate(cls) -> list[Parameters]:
        parameters_list: list[Parameters] = []
        for train_batch_size in TRAIN_BATCH_SIZE_VALUES:
            for val_batch_size in VAL_BATCH_SIZE_VALUES:
                for decaying_factor in DECAYING_FACTOR_VALUES:
                    for patience in PATIENCE_VALUES:
                        parameters_list.append(cls(
                            train_batch_size, 
                            val_batch_size, 
                            decaying_factor, 
                            patience
                            ))
        
        return parameters_list