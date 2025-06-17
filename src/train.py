from logging import Logger
import timeit
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau # type: ignore
from sklearn.model_selection import train_test_split
from predict import EvaluationSession
from utils.SavesManager import SavesManager
from utils.DatasetUtils import DatasetUtils
from utils.BatchLoader import BatchLoader
from config import DatasetPaths, ParametersRegistry
from keras.models import Model
from utils.misc import DiceLoss, BCEDiceLoss, Parameters, ParametersLoaderModel

class TrainingSession:
    def __init__(self, 
                 dataset: DatasetPaths, 
                 models: list[ParametersLoaderModel], 
                 parameters_list: list[Parameters] | None, 
                 logger: Logger):
        self.dataset = dataset
        self.models = models
        self.parameters_list = parameters_list
        self.logger = logger
        
        logger.info("Training session initialization started")
        
        # Get file paths for images and masks
        self.train_img_paths, self.train_mask_paths, test_img_paths, test_mask_paths = DatasetUtils.get_dataset_filepaths(dataset, True)

        # Split test set into validation and test sets (50/50 split, stratify if possible)
        self.val_img_paths, self.test_img_paths, self.val_mask_paths, self.test_mask_paths = train_test_split(
            test_img_paths, test_mask_paths, test_size=0.5, random_state=42
        )
        
        logger.info("File paths loaded successfully")
    
    # Determine set of parameters to use
    def get_parameters_list(self, model: ParametersLoaderModel):
            if self.parameters_list is None or self.parameters_list == [] or self.parameters_list[0] == ParametersRegistry.AUTOMATIC:
                self.logger.info("Parameters list taken from model")
                return model.get_parameters_values()
            
            self.logger.info("Parameters list taken from external source")
            return self.parameters_list
    
    # Train all models
    def train_all(self):
        
        # Train all models with all parameters combinations
        for model in self.models:
            self.logger.info(f"Started training {model.name} models")
            
            # Get parameters list
            parameters_list = self.get_parameters_list(model)
            
            # Get parameters list length
            parameters_list_len = len(parameters_list)
            
            # Train the model on each parameters set
            for index, parameters in enumerate(parameters_list):
                
                # Train the model
                self.train(model, parameters, index, parameters_list_len)
            
            self.logger.info(f"Completed training of {model.name} models")
        self.logger.info(f"Completed training of all models")
        
        # Reset save paths
        SavesManager.reset_save_paths()
        
        self.logger.info(f"Training session ended")
        
    # Train a model according to a set of parameters
    def train(self, model: Model, parameters: Parameters, index: int = 0, parameters_list_len: int = 1):
        self.logger.info(f"Started training model {model.name} on parameter set {index + 1} out of {parameters_list_len}")
        
        # Create batch loaders (on-the-fly loading)
        self.train_batch_loader = BatchLoader(self.train_img_paths, self.train_mask_paths, parameters.BATCH_SIZE)
        self.val_batch_loader = BatchLoader(self.val_img_paths, self.val_mask_paths, parameters.BATCH_SIZE)

        self.logger.info("Batch loaders initialized successfully")
            
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

        self.logger.info(f"Model compiled successfully")

        # Determine save paths and dir name for the model
        model_save_paths: SavesManager.SavePaths = SavesManager.set_generated_save_paths(self.dataset.MODEL_SAVES_PATH, model.name)

        # Scheduler for decaying learning rate
        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss',
            factor=parameters.DECAYING_FACTOR,
            patience=parameters.PATIENCE, 
            min_lr=parameters.MIN_LR,      # Don't reduce LR below this
            verbose=1
        )

        # Create callbacks for model checkpointing and logging
        checkpoint = ModelCheckpoint(
            filepath=model_save_paths.MODEL,
            verbose=1,
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            save_weights_only=False
            )

        # Create CSV Logger
        csv_logger = CSVLogger(
            model_save_paths.TRAINING,
            append=True,
            separator=';'
            )
        
        self.logger.info("Training model...")

        # Training
        with tf.device('/GPU:0'):
            model.fit(
                x=self.train_batch_loader,
                validation_data=self.val_batch_loader,
                epochs=parameters.EPOCHS,
                callbacks=[lr_scheduler, checkpoint, csv_logger],
                verbose=1
            )

        self.logger.info(f"Completed training of model {model.name} on parameter set 1 out of {index + 1}")
        
        # Save parameters
        SavesManager.save_parameters(parameters)
        
        self.logger.info(f"Training results, model and parameters have been saved under \"{model_save_paths.DIRECTORY_NAME}\"")

class TrainingAndEvaluationSession:
    def __init__(self, dataset: DatasetPaths, models: list[ParametersLoaderModel], parameters_list: list[Parameters], logger: Logger):
        self.dataset = dataset
        self.models = models
        self.parameters_list = parameters_list
        self.logger = logger
    
    def start_session_wise(self):
        self.logger.info("Started session-wise training and evaluation session")
        
        # Create training session
        training_session = TrainingSession(self.dataset, self.models, self.parameters_list, self.logger)
        
        # Initialize objects needed for evaluation
        model_names = [model.name for model in self.models]
        image_paths = training_session.test_img_paths
        mask_paths = training_session.test_mask_paths
        
        # Create evaluation session
        eval_session = EvaluationSession(self.dataset.MODEL_SAVES_PATH, image_paths, mask_paths, model_names, self.logger)
        
        # Start training and evaluation
        training_session.train_all()
        eval_session.evaluate_all()
        
        # Reset save paths
        SavesManager.reset_save_paths()
        
        self.logger.info("Ended session-wise training and evaluation session")

    def start_model_wise(self):
        self.logger.info("Started model-wise training and evaluation session")
        
        # Create sessions
        training_session = TrainingSession(self.dataset, self.models, self.parameters_list, self.logger)
            
        # Train all models with all parameters combinations
        for model in self.models:
            training_session.logger.info(f"Started training and evaluating {model.name} models")
            
            # Get parameters list
            parameters_list = training_session.get_parameters_list(model)
            
            # Get parameters list length
            parameters_list_len = len(parameters_list)
            
            for index, parameters in enumerate(parameters_list):
                self.logger.info(f"Started training and evaluating model {model.name} on parameter set {index + 1} out of {parameters_list_len}")
                
                # Train the model
                training_time = timeit.timeit(lambda: training_session.train(model, parameters, index, parameters_list_len), number = 1)
                
                # Evaluate the model
                evaluation_time = timeit.timeit(lambda: EvaluationSession.evaluate(
                    training_session.test_img_paths, 
                    training_session.test_mask_paths, 
                    model, 
                    SavesManager.CURRENT_SAVE_PATHS.DIRECTORY_NAME, 
                    self.logger
                ), number = 1)

                time_dict = {
                    "training_time": training_time,
                    "evaluation_time": evaluation_time,
                    "training_time_per_epoch": training_time / parameters.EPOCHS,
                    "inference_time": evaluation_time / len(training_session.test_img_paths)
                }

                # Save time metrics
                SavesManager.save_time_metrics(time_dict)
                
                self.logger.info(f"Completed training and evaluation of model {model.name} on parameter set {index + 1} out of {parameters_list_len}")
            
            self.logger.info(f"Completed training and evaluation of {model.name} models")
        self.logger.info(f"Completed training and evaluation of all models")
        
        # Reset save paths
        SavesManager.reset_save_paths()
        
        self.logger.info("Ended model-wise training and evaluation session")
        
        
        
        


