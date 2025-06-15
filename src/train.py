from datetime import datetime
import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau # type: ignore
from sklearn.model_selection import train_test_split
from BatchLoader import BatchLoader
from config import DECAYING_FACTOR, EPOCHS, LR, MIN_LR, PATIENCE, DatasetRegistry
from models.UNetL import UNetL
from utils import DiceLoss, BCEDiceLoss, config_gpu, get_image_filepaths
from transunet import TransUNet

# Prepare GPU
config_gpu()

dataset = DatasetRegistry.PALSAR

# Get file paths for images and masks
train_img_paths = get_image_filepaths(dataset.TRAIN_IMAGES_PATH)
train_mask_paths = get_image_filepaths(dataset.TRAIN_LABELS_PATH)
test_img_paths = get_image_filepaths(dataset.TEST_IMAGES_PATH)
test_mask_paths = get_image_filepaths(dataset.TEST_LABELS_PATH)

# Split test set into validation and test sets (50/50 split, stratify if possible)
val_img_paths, test_img_paths, val_mask_paths, test_mask_paths = train_test_split(
    test_img_paths, test_mask_paths, test_size=0.5, random_state=42
)

print("File paths caricati con successo!")

# Create batch loaders (on-the-fly loading)
train_batches = BatchLoader.LoadTrainingBatches(train_img_paths, train_mask_paths)
val_batches = BatchLoader.LoadValidationBatches(val_img_paths, val_mask_paths)


print("Batches caricati con successo!")

# Select model
#model = TransUNet(image_size=256, grid=(16,16), num_classes=2, pretrain=True) 
model = UNetL()

print("Modello creato con successo!")

# Decaying learning rate
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=DECAYING_FACTOR,
    patience=PATIENCE, 
    min_lr=MIN_LR,      # Don't reduce LR below this
    verbose=1
)

# Select optimizer
# Without scheduler set LR = 1e-4
optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

# Losses
bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
dice = DiceLoss()
bce_dice_loss = BCEDiceLoss()

# Metrics
binary_accuracy = tf.keras.metrics.BinaryAccuracy(name='binary_accuracy')
binary_iou = tf.keras.metrics.BinaryIoU(name='binary_iou', target_class_ids=[0, 1])

# Compile model
model.compile(
    loss="binary_crossentropy",
    optimizer=optimizer,
    metrics=[binary_accuracy, binary_iou]
)

print("Modello compilato con successo!")

# Create directories for checkpoints and logs if they don't exist
os.makedirs(f"{os.getcwd()}/checkpoints", exist_ok=True)
os.makedirs(f"{os.getcwd()}/logs", exist_ok=True)

current_datetime: str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

# Create callbacks for model checkpointing and logging
checkpoint = ModelCheckpoint(
    filepath=f"{os.getcwd()}/checkpoints/{model.name}_{current_datetime}.hdf5",
    verbose=1,
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    save_weights_only=False
    )

print("Checkpoint creato con successo!")

csv_logger = CSVLogger(
    f"{os.getcwd()}/logs/{model.name}_{current_datetime}.csv",
    append=True,
    separator=';'
    )

print("CSV Logger creato con successo!")

with tf.device('/GPU:0'):
    history = model.fit(
        x=train_batches,
        validation_data=val_batches,
        epochs=EPOCHS,
        callbacks=[lr_scheduler, checkpoint, csv_logger],
        verbose=1
    )

print("Training completato con successo!")
