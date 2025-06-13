
import os
from keras.callbacks import ModelCheckpoint, CSVLogger
from sklearn.model_selection import train_test_split
import tensorflow as tf
from BatchLoader import BatchLoader
from Config import EPOCHS, LR, MOMENTUM, DatasetRegistry
from models.UNetL import UNetL
from utils import DiceLoss, bce_dice_loss, config_gpu, load_dataset
from transunet import TransUNet

# Prepare GPU
config_gpu()

# Select model
#model = TransUNet(image_size=256, grid=(16,16), num_classes=2, pretrain=True) 
model = UNetL()

# Decaying learning rate
steps_per_epoch = 1
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    LR,
    decay_steps=steps_per_epoch * 2,
    decay_rate=0.7,
    staircase=True)

# Select optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=MOMENTUM)

# IoU metric
iou = tf.keras.metrics.IoU(num_classes=2, target_class_ids=[0])

# Compile model
# Da rivedere perché la binary crossentropy si basa su due classi
model.compile(
    loss=DiceLoss(), # should be loss='binary_crossentropy', # or loss=loss
    optimizer=optimizer,
    metrics=[iou],  # IoU metric
    #metrics=[tf.keras.metrics.BinaryAccuracy()]  #binary_accuracy
)

# Load dataset
train_set, train_labels, val_test_set, val_test_labels = load_dataset(DatasetRegistry.PALSAR)

print("Dataset caricato con successo!")

# Split val and test 50-50
val_set, val_labels, test_set, test_labels = train_test_split(val_test_set, val_test_labels, test_size=0.5)

print("Dataset diviso in train, val e test con successo!")

# Load batches
train_batches = BatchLoader.LoadTrainingBatches(train_set, train_labels)
val_batches = BatchLoader.LoadValidationBatches(val_set, val_labels)

print("Batches caricati con successo!")

checkpoint = ModelCheckpoint(
    filepath=f"{os.getcwd()}/checkpoints/{model.name}.hdf5",
    verbose=1,
    monitor='val_loss',
    mode='min',
    save_best_only=False,
    save_weights_only=False
    )

print("Checkpoint creato con successo!")

csv_logger = CSVLogger(
    f"{os.getcwd()}/logs/{model.name}.csv",
    append=True,
    separator=';'
    )

print("CSV Logger creato con successo!")

with tf.device('/GPU:0'):
    history = model.fit(
        x=train_batches,
        validation_data=val_batches,
        epochs=EPOCHS,
        callbacks=[checkpoint, csv_logger],
        verbose=1
    )

print("Training completato con successo!")
