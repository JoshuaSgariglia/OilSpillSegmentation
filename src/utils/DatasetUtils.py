import glob
import os
from PIL import Image
import cv2
import numpy as np
from numpy.typing import NDArray
from numpy import float32
from config import INPUT_HEIGHT, INPUT_WIDTH, DatasetPaths, DatasetRegistry, Paths
from utils.Denoiser import Denoiser




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
    def get_dataset_filepaths(cls, dataset: DatasetPaths, fastNlMeans_denoised: bool = False):
        train_img_paths = cls.get_image_filepaths(dataset.TRAIN_IMAGES_PATH if not
                                                  fastNlMeans_denoised else dataset.TRAIN_IMAGES_DENOISED_PATH)
        train_mask_paths = cls.get_image_filepaths(dataset.TRAIN_LABELS_PATH)
        test_img_paths = cls.get_image_filepaths(dataset.TEST_IMAGES_PATH if not
                                                 fastNlMeans_denoised else dataset.TEST_IMAGES_DENOISED_PATH)
        test_mask_paths = cls.get_image_filepaths(dataset.TEST_LABELS_PATH)

        return train_img_paths, train_mask_paths, test_img_paths, test_mask_paths

    # Loading and preprocessing images or masks
    @staticmethod
    def load_data(filepath: str, preprocessing: callable, extra_filter: callable = None):
        image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        image = image[:, :, 0]
        image = cv2.resize(image, (INPUT_WIDTH, INPUT_HEIGHT))
        image = preprocessing(image, extra_filter)
        image = np.expand_dims(image, axis=-1)  # (H, W, 1)

        return image

    # Preprocessing logic for image
    @staticmethod
    def preprocess_image(image: NDArray[float32], extra_filter: callable = None) -> NDArray[float32]:
        # IoU UNetL without filter: 0.792  
        # fastNlMeans: 0.791
        #image = Denoiser.median_blur(image) # 0.790 : 0.792 with fnlm
        #image = Denoiser.box_filter(image) # 0.792 : 0.793 with fnlm
        #image = Denoiser.gaussian_blur(image) # 0.793 ; 0.794 with fnlm
        #image = Denoiser.bilateral_filter(image) # 0.786
        image = extra_filter(image) if extra_filter is not None else image

        #image = image.astype(np.float32) / 255.0
        image = (image - np.mean(image))/np.std(image)

        return image

    # Preprocessing logic for mask
    @staticmethod
    def preprocess_mask(mask: NDArray[float32], extra_filter: callable = None) -> NDArray[float32]:
        mask = mask.astype(np.float32) / 255.0
        mask = (mask >= 0.5).astype(np.float32)  # Binarize
        return mask

    # Loading and proprocessing image file
    @classmethod
    def load_image(cls, filepath: str, extra_filter: callable = None) -> NDArray[float32]:
        return cls.load_data(filepath, cls.preprocess_image, extra_filter)

    # Loading and preprocessing image file
    @classmethod
    def load_mask(cls, filepath: str) -> NDArray[float32]:
        return cls.load_data(filepath, cls.preprocess_mask)
    
    @classmethod
    def denoise_dataset(cls, dataset: DatasetPaths):
        
        def denoise_images(source_path, save_path):
            image_paths = cls.get_image_filepaths(source_path)
            for image_path in image_paths:
                image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                image = Denoiser.fastNlMeans(image)
                
                image_filename = image_path.split('/')[-1]
                os.makedirs(save_path, exist_ok=True)
                Image.fromarray((image).astype(np.uint8)).save(os.path.join(save_path, image_filename))
                
        denoise_images(dataset.TRAIN_IMAGES_PATH, dataset.TRAIN_IMAGES_DENOISED_PATH)
        denoise_images(dataset.TEST_IMAGES_PATH, dataset.TEST_IMAGES_DENOISED_PATH)

    
    @staticmethod
    def test_denoising(dataset: DatasetPaths = DatasetRegistry.PALSAR, image_number: int = 123):
        image_path = os.path.join(dataset.TEST_IMAGES_PATH, f'{image_number}.png')
        mask_path = os.path.join(dataset.TEST_LABELS_PATH, f'{image_number}.png')    
        
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)  

        image_gaussian = Denoiser.gaussian_blur(image)
        image_median = Denoiser.median_blur(image)
        image_bilateral = Denoiser.bilateral_filter(image)
        image_box = Denoiser.box_filter(image)
        image_fastNlMeans = Denoiser.fastNlMeans(image)
        image_gaussian_fnlm = Denoiser.gaussian_blur(image_fastNlMeans)
        image_median_fnlm = Denoiser.median_blur(image_fastNlMeans)
        image_bilateral_fnlm = Denoiser.bilateral_filter(image_fastNlMeans)
        image_box_fnlm = Denoiser.box_filter(image_fastNlMeans)
        
        def save_to_denoising_test_directory(image: NDArray, filename: str):
            Image.fromarray((image).astype(np.uint8)).save(os.path.join(Paths.DENOISING, filename))


        os.makedirs(Paths.DENOISING, exist_ok=True)
        save_to_denoising_test_directory(image, "image.png")
        save_to_denoising_test_directory(mask, "mask.png")
        save_to_denoising_test_directory(image_gaussian, "gaussian.png")
        save_to_denoising_test_directory(image_median, "median.png")
        save_to_denoising_test_directory(image_bilateral, "bilateral.png")
        save_to_denoising_test_directory(image_box, "box.png")
        save_to_denoising_test_directory(image_fastNlMeans, "fastNlMeans.png")
        save_to_denoising_test_directory(image_gaussian_fnlm, "gaussian_fastNlMeans.png")
        save_to_denoising_test_directory(image_median_fnlm, "median_fastNlMeans.png")
        save_to_denoising_test_directory(image_bilateral_fnlm, "bilateral_fastNlMeans.png")
        save_to_denoising_test_directory(image_box_fnlm, "box_fastNlMeans.png")

    
   


   