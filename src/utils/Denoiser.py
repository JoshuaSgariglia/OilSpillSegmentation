# Denoising
import cv2
from numpy.typing import NDArray
from numpy import float32, uint8


class Denoiser:
    # Gaussian Blur
    @staticmethod
    def gaussian_blur(image: NDArray[float32]) -> NDArray[float32]:
        return cv2.GaussianBlur(image, (5, 5), 1)

    # Median Filter
    @staticmethod
    def median_blur(image: NDArray[float32]) -> NDArray[float32]:
        return cv2.medianBlur((image).astype(uint8), 7) 

    # Bilateral Filter
    @staticmethod
    def bilateral_filter(image: NDArray[float32], d=11, sigma_color=100, sigma_space=75) -> NDArray[float32]:
        return cv2.bilateralFilter((image).astype(uint8), d, sigma_color, sigma_space) 

    # Box Filter
    @staticmethod
    def box_filter(image: NDArray[float32]) -> NDArray[float32]:
        return cv2.boxFilter((image).astype(uint8), 3, (5, 5))
    
    @staticmethod
    def fastNlMeans(image: NDArray[float32], h=40, templateWindowSize=7, searchWindowSize=21) -> NDArray[float32]:
        """
        Apply Non-local Means Denoising to a grayscale image.
        - h: Filter strength (higher removes noise but may remove details)
        - templateWindowSize: Size in pixels of the template patch
        - searchWindowSize: Size in pixels of the window used to compute weighted average
        """
        # Ensure image is uint8
        return cv2.fastNlMeansDenoising((image).astype(uint8), None, h, templateWindowSize, searchWindowSize)
         