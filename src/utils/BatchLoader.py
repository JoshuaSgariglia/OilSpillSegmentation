import random
from tensorflow.keras.utils import Sequence # type: ignore
import numpy as np
from config import INP_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH, OUT_MASKS
from utils.DatasetUtils import DatasetUtils


class BatchLoader(Sequence):
    
    def __init__(self, input_paths, mask_paths, batch_size: int, input_channels: int = INP_CHANNELS, shuffle=True):
        self.input_paths = input_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.input_channels = input_channels
        self.augment_class = augment
        self.shuffle = shuffle

        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.input_paths) / self.batch_size))

    def __getitem__(self, batch_index):
        element_indexes = self.indexes[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]

        X, Y = self.__data_generation(element_indexes)
        return X, Y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.input_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, element_indexes):
        X = np.empty((self.batch_size, INPUT_HEIGHT, INPUT_WIDTH, self.input_channels), dtype=np.float32)
        Y = np.empty((self.batch_size, INPUT_HEIGHT, INPUT_WIDTH, OUT_MASKS), dtype=np.float32)
        
        for index, element_index in enumerate(element_indexes):
            image_path = self.input_paths[element_index]
            mask_path = self.mask_paths[element_index]

            # Load image and mask from disk
            image = DatasetUtils.load_image(image_path)
            mask = DatasetUtils.load_mask(mask_path)
                
            # Convert image from 3-channels to 1-channel (needed for TransUNet)
            image = image = np.repeat(image, 3, axis=2) if self.input_channels == 3 else image

            X[index, :, :, :] = image
            Y[index, :, :, :] = mask

        return X, Y