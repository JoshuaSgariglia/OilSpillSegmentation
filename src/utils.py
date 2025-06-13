import glob
import os
import cv2
from keras.losses import binary_crossentropy
from keras.backend import flatten, sum
import numpy as np
import tensorflow as tf
from Config import INP_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH, OUT_MASKS, DatasetPaths
from numpy.typing import NDArray
from numpy import float64

# Losses
def generalized_dice_coefficient(y_true, y_pred):
        smooth = 1.
        y_true_f = flatten(y_true)
        y_pred_f = flatten(y_pred)
        intersection = sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (
                    sum(y_true_f) + sum(y_pred_f) + smooth)
        return score
    
def dice_loss(y_true, y_pred):
        loss = 1 - generalized_dice_coefficient(y_true, y_pred)
        return loss

def bce_dice_loss(y_true, y_pred):
        loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
        return loss / 2.0

class DiceLoss(tf.keras.losses.Loss):
  def __init__(self):
    super().__init__()
    self.redsum = tf.math.reduce_sum

  def call(self, y_true, y_pred, smooth=0.01):
    intersec = self.redsum(y_true * y_pred)
    dice = (2 * intersec + smooth) / (self.redsum(y_true) + self.redsum(y_pred) + smooth)

    return 1 - dice


# GPU \ System
def config_gpu():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" # GPU with ID 0

    gpus = tf.config.list_physical_devices(device_type='GPU')

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(device=gpu, enable=True)

# Get image file paths
def get_image_filepaths(directory: str, extensions: list[str] = ["jpg", "tif", "png", "jpeg"]):
    filepaths = []
    for ext in extensions:
        filepaths += glob.glob(os.path.join(directory, f"*.{ext}"))
    return sorted(filepaths)

# Loading iamges or masks
def load_data(directory: str, channels: int, preprocessing: callable, extensions: list[str] = ["jpg", "tif", "png", "jpeg"]):
    # Get all image file paths in the directory
    images_paths: list[str] = get_image_filepaths(directory, extensions)

    # Initialize an empty array to hold the images
    images: NDArray[float64] = np.empty((len(images_paths), INPUT_HEIGHT, INPUT_WIDTH, channels))

    # Load and preprocess each image
    for index, image_path in enumerate(images_paths):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = image[:, :, 0]
        image = cv2.resize(image, (INPUT_WIDTH, INPUT_HEIGHT))
        image = preprocessing(image)
        images[index, :, :, 0] = image

    return images

def preprocess_image(image: NDArray[float64]) -> NDArray[float64]:
    image = (image - np.mean(image))/np.std(image)

    return image

def preprocess_mask(mask: NDArray[float64]) -> NDArray[float64]:
    mask = mask.astype(np.float64) / 255.0

    return mask

# Loading image files
def load_images(directory: str , extensions:list[str] = ["jpg", "tif", "png", "jpeg"]):
    return load_data(directory, INP_CHANNELS, preprocess_image, extensions)

# Loading image files
def load_masks(directory: str , extensions:list[str] = ["jpg", "tif", "png", "jpeg"]):
    return load_data(directory, OUT_MASKS, preprocess_mask, extensions)

# Load whole dataset
def load_dataset(dataset: type[DatasetPaths]):
    train_set = load_images(dataset.TRAIN_IMAGES_PATH)
    train_labels = load_masks(dataset.TRAIN_LABELS_PATH)
    val_test_set = load_images(dataset.TEST_IMAGES_PATH)
    val_test_labels = load_masks(dataset.TEST_LABELS_PATH)

    return train_set, train_labels, val_test_set, val_test_labels


