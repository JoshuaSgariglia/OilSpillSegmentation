from __future__ import annotations
from datetime import datetime
import os
import logging
from keras.models import Model
from tensorflow.keras.losses import Loss, BinaryCrossentropy # type: ignore
import tensorflow as tf
from config import DATETIME_FORMAT, LOG_FILENAME, Paths
from dataclass import ParametersValues, Parameters

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
    
# Custom Parameter Loader class for models
class ParametersLoaderModel(Model):
    NAME: str
    NEEDS_BUILDING: bool = False
    
    @property
    def INTERNAL_NAME(self) -> str:
        return self.name if self.name is not None else self.NAME
    
    @classmethod
    def get_parameters_values(cls) -> ParametersValues:
        raise NotImplementedError("Method not implemented")

    @classmethod
    def show_model_summary(cls):
        print(cls().summary())
    
    @property
    def inp_channels(self) -> int:
        return self.input_shape[-1]  
      
    # Generate a list of Parameters from a ParametersValues object
    @staticmethod
    def generate_parameters_list(parameters_values: ParametersValues) -> list[Parameters]:
        # Initialize empty Parameters list
        parameters_list: list[Parameters] = []
        
        # Get all possible combinations
        for batch_size in parameters_values.BATCH_SIZE_VALUES:
            for decaying_factor in parameters_values.DECAYING_FACTOR_VALUES:
                for patience in parameters_values.PATIENCE_VALUES:
                    parameters_list.append(Parameters(
                        batch_size,
                        decaying_factor, 
                        patience,
                        parameters_values.EPOCHS,
                        parameters_values.LR,
                        parameters_values.MIN_LR
                        ))
        
        return parameters_list
    
    
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

