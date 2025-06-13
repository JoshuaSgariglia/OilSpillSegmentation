import random
from typing import Sequence

import numpy as np
from Config import INP_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH, OUT_MASKS, TRAIN_BATCH_SIZE, VAL_BATCH_SIZE 


class BatchLoader(Sequence):
    @classmethod
    def LoadTrainingBatches(cls, inputs, masks, augment=None, shuffle=True):
        'Create training batches'
        return cls(inputs, masks, TRAIN_BATCH_SIZE, augment=augment, shuffle=shuffle)
    
    @classmethod
    def LoadValidationBatches(cls, inputs, masks, augment=None, shuffle=True):
        'Create validation batches'
        return cls(inputs, masks, VAL_BATCH_SIZE, augment=augment, shuffle=shuffle)
    

    def __init__(self, inputs, masks, batch_size: int, augment=None, shuffle=True):
        'The constructor can be expanded with as many attributes as needed'
        self.inputs = inputs
        self.masks = masks
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment_class = augment
        self.img_width = INPUT_WIDTH
        self.img_height = INPUT_HEIGHT
        self.inp_chan = INP_CHANNELS
        self.out_chan = OUT_MASKS

        self.on_epoch_end()

    def __len__(self):
        'Take all batches in each iteration'
        return int(np.floor(len(self.inputs) / self.batch_size))

    def __getitem__(self, index):
        'Get next batch'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Generate data
        X, Y = self.__data_generation(indexes)
        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.inputs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_idx):
        'Generates data containing batch_size samples'
        X = np.empty((self.batch_size, self.img_height, self.img_width, self.inp_chan), dtype=np.float32)
        Y = np.empty((self.batch_size, self.img_height, self.img_width, self.out_chan), dtype=np.float32)
        for i_batch, id in enumerate(list_idx):
          # Store samples and masks
          img = self.inputs[id]
          mask = self.masks[id]

          # Augment
          if self.augment_class is not None:
            augment = self.augment_class(seed=random.randint(0,100))
            img, mask = augment(img, mask)

          X[i_batch, :, :, :] = img
          Y[i_batch, :, :, :] = mask

        return X, Y