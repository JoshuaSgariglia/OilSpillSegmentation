import random
from typing import Callable, Sequence

import cv2
import numpy as np
from Config import INP_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH, OUT_MASKS, TRAIN_BATCH_SIZE, VAL_BATCH_SIZE
from utils import load_image, load_mask, preprocess_image, preprocess_mask


class BatchLoader(Sequence):
    @classmethod
    def LoadTrainingBatches(cls, input_paths, mask_paths, augment=None, shuffle=True):
        'Create training batches'
        return cls(input_paths, mask_paths, TRAIN_BATCH_SIZE, augment=augment, shuffle=shuffle)
    
    @classmethod
    def LoadValidationBatches(cls, input_paths, mask_paths, augment=None, shuffle=True):
        'Create validation batches'
        return cls(input_paths, mask_paths, VAL_BATCH_SIZE, augment=augment, shuffle=shuffle)
    
    def __init__(self, input_paths, mask_paths, batch_size: int, augment=None, shuffle=True):
        self.input_paths = input_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment_class = augment

        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.input_paths) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X, Y = self.__data_generation(indexes)
        return X, Y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.input_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_idx):
        X = np.empty((self.batch_size, INPUT_HEIGHT, INPUT_WIDTH, INP_CHANNELS), dtype=np.float32)
        Y = np.empty((self.batch_size, INPUT_HEIGHT, INPUT_WIDTH, OUT_MASKS), dtype=np.float32)
        
        for i_batch, idx in enumerate(list_idx):
            image_path = self.input_paths[idx]
            mask_path = self.mask_paths[idx]

            # Load image and mask from disk
            image = load_image(image_path)
            mask = load_mask(mask_path)

            # Augment
            if self.augment_class is not None:
                augment = self.augment_class(seed=random.randint(0,100))
                image, mask = augment(image, mask)

            X[i_batch, :, :, :] = image
            Y[i_batch, :, :, :] = mask

        return X, Y