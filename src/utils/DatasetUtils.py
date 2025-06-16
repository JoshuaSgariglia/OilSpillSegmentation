import glob
import os
import cv2
import numpy as np
from numpy.typing import NDArray
from numpy import float32
from config import INPUT_HEIGHT, INPUT_WIDTH, DatasetPaths

class DatasetUtils:
    # Get image file paths
    @staticmethod
    def get_image_filepaths(directory: str, extensions: list[str] = ["jpg", "tif", "png", "jpeg"]):
        filepaths = []
        for ext in extensions:
            filepaths += glob.glob(os.path.join(directory, f"*.{ext}"))
        return sorted(filepaths)

    # Get dataset file paths
    @classmethod
    def get_dataset_filepaths(cls, dataset: type[DatasetPaths]):
        train_img_paths = cls.get_image_filepaths(dataset.TRAIN_IMAGES_PATH)
        train_mask_paths = cls.get_image_filepaths(dataset.TRAIN_LABELS_PATH)
        test_img_paths = cls.get_image_filepaths(dataset.TEST_IMAGES_PATH)
        test_mask_paths = cls.get_image_filepaths(dataset.TEST_LABELS_PATH)

        return train_img_paths, train_mask_paths, test_img_paths, test_mask_paths

    # Loading and preprocessing images or masks
    @staticmethod
    def load_data(filepath: str, preprocessing: callable):
        image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        image = image[:, :, 0]
        image = cv2.resize(image, (INPUT_WIDTH, INPUT_HEIGHT))
        image = preprocessing(image)
        image = np.expand_dims(image, axis=-1)  # (H, W, 1)

        return image

    # Preprocessing logic for image
    @staticmethod
    def preprocess_image(image: NDArray[float32]) -> NDArray[float32]:
        #image = image.astype(np.float32) / 255.0
        image = (image - np.mean(image))/np.std(image)

        return image

    # Preprocessing logic for mask
    @staticmethod
    def preprocess_mask(mask: NDArray[float32]) -> NDArray[float32]:
        mask = mask.astype(np.float32) / 255.0
        mask = (mask >= 0.5).astype(np.float32)  # Binarize
        return mask

    # Loading and proprocessing image file
    @classmethod
    def load_image(cls, filepath: str) -> NDArray[float32]:
        return cls.load_data(filepath, cls.preprocess_image)

    # Loading and preprocessing image file
    @classmethod
    def load_mask(cls, filepath: str) -> NDArray[float32]:
        return cls.load_data(filepath, cls.preprocess_mask)