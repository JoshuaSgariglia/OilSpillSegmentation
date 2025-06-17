import glob
import os
from PIL import Image
import cv2
import numpy as np
from numpy.typing import NDArray
from numpy import float32
from config import INPUT_HEIGHT, INPUT_WIDTH, DatasetPaths, DatasetRegistry, Paths


# Denoising
class Denoiser:
    # Gaussian Blur
    @staticmethod
    def gaussian_blur(image: NDArray[float32]) -> NDArray[float32]:
        return cv2.GaussianBlur(image, (5, 5), 1)

    # Median Filter
    @staticmethod
    def median_blur(image: NDArray[float32]) -> NDArray[float32]:
        return cv2.medianBlur((image).astype(np.uint8), 7) 

    # Bilateral Filter
    @staticmethod
    def bilateral_filter(image: NDArray[float32], d=11, sigma_color=100, sigma_space=75) -> NDArray[float32]:
        return cv2.bilateralFilter((image).astype(np.uint8), d, sigma_color, sigma_space) 

    # Box Filter
    @staticmethod
    def box_filter(image: NDArray[float32]) -> NDArray[float32]:
        return cv2.boxFilter((image).astype(np.uint8), 3, (5, 5))
    
    @staticmethod
    def fastNlMeans(image: NDArray[float32], h=40, templateWindowSize=7, searchWindowSize=21) -> NDArray[float32]:
        """
        Apply Non-local Means Denoising to a grayscale image.
        - h: Filter strength (higher removes noise but may remove details)
        - templateWindowSize: Size in pixels of the template patch
        - searchWindowSize: Size in pixels of the window used to compute weighted average
        """
        # Ensure image is uint8
        return cv2.fastNlMeansDenoising((image).astype(np.uint8), None, h, templateWindowSize, searchWindowSize)
         

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
        # IoU UNetL without filter: 0.792  
        # fastNlMeans: 0.791
        #image = Denoiser.median_blur(image) # 0.790
        #image = Denoiser.box_filter(image) # 0.792
        #image = Denoiser.gaussian_blur(image) # 0.793
        #image = Denoiser.bilateral_filter(image) # 0.786

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
                
        #denoise_images(dataset.TRAIN_IMAGES_PATH, dataset.TRAIN_IMAGES_DENOISED_PATH)
        denoise_images(dataset.TEST_IMAGES_PATH, dataset.TEST_IMAGES_DENOISED_PATH)

    
    @staticmethod
    def test_denoising(image_number: int = 123):
        image_path = os.path.join(DatasetRegistry.PALSAR.TEST_IMAGES_PATH, f'{image_number}.png')
        mask_path = os.path.join(DatasetRegistry.PALSAR.TEST_LABELS_PATH, f'{image_number}.png')    
        
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)  

        image_gaussian = Denoiser.gaussian_blur(image)
        image_median = Denoiser.median_blur(image)
        image_bilateral = Denoiser.bilateral_filter(image)
        image_box = Denoiser.box_filter(image)
        iamge_fastNlMeans = Denoiser.fastNlMeans(image)


        os.makedirs(Paths.DENOISING, exist_ok=True)
        Image.fromarray((image_gaussian).astype(np.uint8)).save(os.path.join(Paths.DENOISING, "gaussian.png"))
        Image.fromarray((image_median).astype(np.uint8)).save(os.path.join(Paths.DENOISING, "median.png"))
        Image.fromarray((image_bilateral).astype(np.uint8)).save(os.path.join(Paths.DENOISING, "bilateral.png"))
        Image.fromarray((image_box).astype(np.uint8)).save(os.path.join(Paths.DENOISING, "box.png"))
        Image.fromarray((iamge_fastNlMeans).astype(np.uint8)).save(os.path.join(Paths.DENOISING, "fastNlMeans.png"))
        Image.fromarray((image).astype(np.uint8)).save(os.path.join(Paths.DENOISING, "image.png"))
        Image.fromarray((mask).astype(np.uint8)).save(os.path.join(Paths.DENOISING, "mask.png"))

    
   


   