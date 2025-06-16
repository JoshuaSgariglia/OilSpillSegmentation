from logging import Logger
import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau # type: ignore
from sklearn.model_selection import train_test_split
from BatchLoader import BatchLoader
from config import DatasetRegistry, Parameters, Paths
from keras.models import Model
from utils import DiceLoss, BCEDiceLoss, get_image_filepaths, save_parameters

class TrainingSession:
    def __init__(self, dataset: DatasetRegistry, models: list[Model], parameters_list: list[Parameters], logger: Logger):
        self.dataset = dataset
        self.models = models
        self.parameters_list = parameters_list
        
        logger.info("Training session initialization started")
        
        # Get file paths for images and masks
        self.train_img_paths = get_image_filepaths(dataset.TRAIN_IMAGES_PATH)
        self.train_mask_paths = get_image_filepaths(dataset.TRAIN_LABELS_PATH)
        test_img_paths = get_image_filepaths(dataset.TEST_IMAGES_PATH)
        test_mask_paths = get_image_filepaths(dataset.TEST_LABELS_PATH)

        # Split test set into validation and test sets (50/50 split, stratify if possible)
        self.val_img_paths, self.test_img_paths, self.val_mask_paths, self.test_mask_paths = train_test_split(
            test_img_paths, test_mask_paths, test_size=0.5, random_state=42
        )
        
        logger.info("File paths loaded successfully")

        # Create batch loaders (on-the-fly loading)
        train_batch_loader = BatchLoader.LoadTrainingBatches(self.train_img_paths, self.train_mask_paths)
        val_batch_loader = BatchLoader.LoadValidationBatches(self.val_img_paths, self.val_mask_paths)

        logger.info("Batch loaders initialized successfully")
        
        parameters_list_len = len(parameters_list)
        
        # Train all models with all parameters combinations
        for model in self.models:
            logger.info(f"Started training model {model.name}")
            for index, parameters in enumerate(self.parameters_list):
                logger.info(f"Started training model {model.name} on parameter set n°{index + 1}/{parameters_list_len}")
                
                # Scheduler for decaying learning rate
                lr_scheduler = ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=parameters.DECAYING_FACTOR,
                    patience=parameters.PATIENCE, 
                    min_lr=parameters.MIN_LR,      # Don't reduce LR below this
                    verbose=1
                )
                    
                # Select optimizer
                # Without scheduler set LR = 1e-4
                optimizer = tf.keras.optimizers.Adam(learning_rate=parameters.LR)

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

                logger.info(f"Model compiled successfully")

                # Determine save path
                save_path: str = Paths.get_model_saves_path(model.name)

                # Create directories for checkpoints and logs if they don't exist
                os.makedirs(save_path, exist_ok=True)

                # Create callbacks for model checkpointing and logging
                checkpoint = ModelCheckpoint(
                    filepath=os.path.join(save_path, "model.hdf5"),
                    verbose=1,
                    monitor='val_loss',
                    mode='min',
                    save_best_only=True,
                    save_weights_only=False
                    )

                # Create CSV Logger
                csv_logger = CSVLogger(
                    os.path.join(save_path, "training.csv"),
                    append=True,
                    separator=';'
                    )

                # Training
                with tf.device('/GPU:0'):
                    model.fit(
                        x=train_batch_loader,
                        validation_data=val_batch_loader,
                        epochs=parameters.EPOCHS,
                        callbacks=[lr_scheduler, checkpoint, csv_logger],
                        verbose=1
                    )

                logger.info(f"Completed training of model {model.name} on parameter set N°{index + 1}")
                
                # Save parameters
                save_parameters(save_path, parameters)
            
            logger.info(f"Completed training of model {model.name}")
        logger.info(f"Completed training of all models")




