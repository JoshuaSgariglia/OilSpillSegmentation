import glob
import os
import cv2
from tensorflow.keras.losses import Loss, BinaryCrossentropy # type: ignore
import numpy as np
import tensorflow as tf
from config import INPUT_HEIGHT, INPUT_WIDTH, DatasetPaths, DatasetRegistry
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


# GPU \ System
def config_gpu():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" # GPU with ID 0

    gpus = tf.config.list_physical_devices(device_type='GPU')

    for gpu in gpus:
        # GPU memory allocation expands as needed
        tf.config.experimental.set_memory_growth(device=gpu, enable=True)

# Get image file paths
def get_image_filepaths(directory: str, extensions: list[str] = ["jpg", "tif", "png", "jpeg"]):
    filepaths = []
    for ext in extensions:
        filepaths += glob.glob(os.path.join(directory, f"*.{ext}"))
    return sorted(filepaths)

# Get dataset file paths
def get_dataset_filepaths(dataset: type[DatasetPaths]):
    train_img_paths = get_image_filepaths(dataset.TRAIN_IMAGES_PATH)
    train_mask_paths = get_image_filepaths(dataset.TRAIN_LABELS_PATH)
    test_img_paths = get_image_filepaths(dataset.TEST_IMAGES_PATH)
    test_mask_paths = get_image_filepaths(dataset.TEST_LABELS_PATH)

    return train_img_paths, train_mask_paths, test_img_paths, test_mask_paths

# Loading images or masks
def load_data(filepath: str, preprocessing: callable):
    image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    image = image[:, :, 0]
    image = cv2.resize(image, (INPUT_WIDTH, INPUT_HEIGHT))
    image = preprocessing(image)
    image = np.expand_dims(image, axis=-1)  # (H, W, 1)

    return image

def preprocess_image(image: NDArray[float32]) -> NDArray[float32]:
    #image = image.astype(np.float32) / 255.0
    image = (image - np.mean(image))/np.std(image)

    return image

def preprocess_mask(mask: NDArray[float32]) -> NDArray[float32]:
    mask = mask.astype(np.float32) / 255.0
    mask = (mask > 0.5).astype(np.float32)  # Binarize
    return mask

# Loading image file
def load_image(filepath: str):
    return load_data(filepath, preprocess_image)

# Loading image file
def load_mask(filepath: str):
    return load_data(filepath, preprocess_mask)
