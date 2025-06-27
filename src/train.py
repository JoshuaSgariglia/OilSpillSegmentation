from logging import Logger
import timeit
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau # type: ignore
from sklearn.model_selection import train_test_split
from predict import EvaluationSession
from utils.Denoiser import Denoiser
from utils.SavesManager import SavesManager
from utils.DatasetUtils import DatasetUtils
from utils.BatchLoader import BatchLoader
from config import DatasetPaths, ParametersRegistry
from keras.models import Model
from utils.misc import DiceLoss, BCEDiceLoss, Parameters, ParametersLoaderModel

class TrainingSession:
    def __init__(self,  
                 logger: Logger,
                 dataset: DatasetPaths, 
                 model_classes: list[type[ParametersLoaderModel]], 
                 parameters_list: list[Parameters] | None = None,
                 filter_parameters: bool = True,
                 denoised_dataset: bool = True,
                 extra_filter: callable = Denoiser.gaussian_blur
                 ):
        self.logger = logger
        self.dataset = dataset
        self.model_classes = model_classes
        self.parameters_list = parameters_list
        self.filter_parameters = filter_parameters
        self.denoised_dataset = denoised_dataset
        self.extra_filter = extra_filter
        
        
        logger.info("Training session initialization started")
        logger.info(f"Training on {'denoised' if denoised_dataset else 'undenoised'} dataset {self.dataset.DATASET_NAME}")
        logger.info(f"{'Using extra filter ' + extra_filter.__name__ if extra_filter is not None else 'No extra filter used'}")
        
        # Get file paths for images and masks
        self.train_img_paths, self.train_mask_paths, test_img_paths, test_mask_paths = DatasetUtils.get_dataset_filepaths(dataset, denoised_dataset)

        # Split test set into validation and test sets (50/50 split, stratify if possible)
        self.val_img_paths, self.test_img_paths, self.val_mask_paths, self.test_mask_paths = train_test_split(
            test_img_paths, test_mask_paths, test_size=0.5, random_state=42
        )
        
        logger.info("File paths loaded successfully")
    
    # Determine set of parameters to use
    def get_parameters_list(self, model_class: type[ParametersLoaderModel]):
            if self.parameters_list is None or self.parameters_list == [] or self.parameters_list[0] == ParametersRegistry.AUTOMATIC:
                self.logger.info("Parameters list taken from model")
                return model_class.get_parameters_values()
            
            self.logger.info("Parameters list taken from external source")
            return self.parameters_list
        
    # Filter the list of parameters
    def filter_parameters_list(self, model_name: str, parameters_list: list[Parameters]):
        # List all single subdirectories of model saves
        model_saves_paths = SavesManager.get_all_saves_paths_by_model_name(model_name, self.dataset.MODEL_SAVES_PATH)
        
        # Load all saved parameters into a list
        saved_parameters_list = [SavesManager.load_parameters(path) for path in model_saves_paths]
        
        # Filter parameters
        filtered_parameters_list = [p for p in parameters_list if p not in saved_parameters_list]
        
        self.logger.info(f"Skipped {len(parameters_list) - len(filtered_parameters_list)} parameter sets")
        
        return filtered_parameters_list
    
    # Train all models
    def train_all(self):
        
        # Train all models with all parameters combinations
        for model_class in self.model_classes:
            self.logger.info(f"Started training {model_class.NAME} models")
            
            # Get parameters list
            parameters_list = self.get_parameters_list(model_class)
            
            # Filter parameters list
            if self.filter_parameters:
                parameters_list = self.filter_parameters_list(model_class.NAME, parameters_list)
            
            # Get parameters list length
            parameters_list_len = len(parameters_list)
            
            # Train the model on each parameters set
            for index, parameters in enumerate(parameters_list):
                
                # Instantiate model
                model = model_class()
                
                # Train the model
                self.train(model, parameters, index, parameters_list_len)
            
            self.logger.info(f"Completed training of {model.NAME} models")
        self.logger.info(f"Completed training of all models")
        
        # Reset save paths
        SavesManager.reset_save_paths()
        
        self.logger.info(f"Training session ended")
        
    # Train a model according to a set of parameters
    def train(self, model: ParametersLoaderModel, parameters: Parameters, index: int = 0, parameters_list_len: int = 1, save: bool = True):
        self.logger.info(f"Started training model {model.NAME} on parameter set {index + 1} out of {parameters_list_len}")
        
        self.logger.info(f"{'Loading images using ' + self.extra_filter.__name__ if self.extra_filter is not None else 'Loading images with no extra filter'}")
        
        # Building model
        if model.NEEDS_BUILDING:
            model.build(model.build_input_shape)
            self.logger.info("Model built successfully")
        
        # Create batch loaders (on-the-fly loading)
        train_batch_loader = BatchLoader(self.train_img_paths, self.train_mask_paths, parameters.BATCH_SIZE, model.inp_channels, self.extra_filter)
        val_batch_loader = BatchLoader(self.val_img_paths, self.val_mask_paths, parameters.BATCH_SIZE, model.inp_channels, self.extra_filter)

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
        model_save_paths: SavesManager.SavePaths = SavesManager.set_generated_save_paths(self.dataset.MODEL_SAVES_PATH, model.NAME)

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
            filepath=model_save_paths.MODEL_TF if model.NEEDS_BUILDING else model_save_paths.MODEL,
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
                x=train_batch_loader,
                validation_data=val_batch_loader,
                epochs=parameters.EPOCHS,
                callbacks=[lr_scheduler, checkpoint, csv_logger] if save else [lr_scheduler],
                verbose=1
            )

        self.logger.info(f"Completed training of model {model.NAME} on parameter set {index + 1} out of {parameters_list_len}")
        
        # Save parameters
        if save:
            SavesManager.save_parameters(parameters)
        
        self.logger.info(f"Training results, model and parameters have been saved under \"{model_save_paths.DIRECTORY_NAME}\"")

class TrainingAndEvaluationSession:
    def __init__(self,  
                 logger: Logger,
                 dataset: DatasetPaths, 
                 model_classes: list[type[ParametersLoaderModel]], 
                 parameters_list: list[Parameters] | None = None,
                 filter_parameters: bool = True,
                 denoised_dataset: bool = True,
                 extra_filter: callable = Denoiser.gaussian_blur
                 ):
        self.logger = logger
        self.dataset = dataset
        self.model_classes = model_classes
        self.parameters_list = parameters_list
        self.filter_parameters = filter_parameters
        self.denoised_dataset = denoised_dataset
        self.extra_filter = extra_filter
    
    def start_session_wise(self):
        self.logger.info("Started session-wise training and evaluation session")
        
        # Create training session
        training_session = TrainingSession(self.logger, self.dataset, self.model_classes, self.parameters_list, self.filter_parameters, self.denoised_dataset, self.extra_filter)
        
        # Initialize objects needed for evaluation
        model_names = [model_class.NAME for model_class in self.model_classes]
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
        training_session = TrainingSession(self.logger, self.dataset, self.model_classes, self.parameters_list, self.filter_parameters, self.denoised_dataset, self.extra_filter)
            
        # Train all models with all parameters combinations
        for model_class in self.model_classes:
            training_session.logger.info(f"Started training and evaluating {model_class.NAME} models")
            
            # Get parameters list
            parameters_list = training_session.get_parameters_list(model_class)
            
            # Filter parameters list
            if self.filter_parameters:
                parameters_list = training_session.filter_parameters_list(model_class.NAME, parameters_list)
            
            # Get parameters list length
            parameters_list_len = len(parameters_list)
            
            for index, parameters in enumerate(parameters_list):
                
                # Instantiate the model
                model = model_class()
        
                self.logger.info(f"Started training and evaluating model {model.NAME} on parameter set {index + 1} out of {parameters_list_len}")
                
                # Train the model
                training_time = timeit.timeit(lambda: training_session.train(model, parameters, index, parameters_list_len), number = 1)
                
                # Evaluate the model
                evaluation_time = timeit.timeit(lambda: EvaluationSession.evaluate(
                    training_session.test_img_paths, 
                    training_session.test_mask_paths, 
                    model, 
                    SavesManager.CURRENT_SAVE_PATHS.DIRECTORY_NAME, 
                    self.logger,
                    True,
                    self.extra_filter
                ), number = 1)

                time_dict = {
                    "training_time": training_time,
                    "evaluation_time": evaluation_time,
                    "training_time_per_epoch": training_time / parameters.EPOCHS,
                    "inference_time": evaluation_time / len(training_session.test_img_paths)
                }

                # Save time metrics
                SavesManager.save_time_metrics(time_dict)
                
                self.logger.info(f"Completed training and evaluation of model {model.NAME} on parameter set {index + 1} out of {parameters_list_len}")
            
            self.logger.info(f"Completed training and evaluation of {model_class.NAME} models")
        self.logger.info(f"Completed training and evaluation of all models")
        
        # Reset save paths
        SavesManager.reset_save_paths()
        
        self.logger.info("Ended model-wise training and evaluation session")
        
        
        
        


