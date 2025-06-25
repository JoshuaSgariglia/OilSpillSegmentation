from logging import Logger
import os
from codecarbon import EmissionsTracker
from dataclass import DatasetPaths, Parameters
from predict import EvaluationSession
from config import Paths, SaveFilename
from utils.SavesManager import SavesManager
from train import TrainingSession
from utils.misc import ParametersLoaderModel

class CO2Tracker:
    
    # Track training and evaluation emissions
    @staticmethod
    def track_emissions(  
                logger: Logger,
                dataset: DatasetPaths, 
                model_classes: list[type[ParametersLoaderModel]]
                ): 
        
        # Instantiate training session
        training_session = TrainingSession(logger, dataset, model_classes)
        
        for model_class in model_classes:
            # Get some parameters for the model and themultiplier
            parameters_list = model_class.get_parameters_values()
            parameters: Parameters = parameters_list[0]
            multiplier = len(parameters_list)
            
            # Instantiate the model
            model = model_class()
            
            # Define tracker
            tracker: EmissionsTracker

            # Track emission for training
            with EmissionsTracker() as tracker:
                training_session.train(model, parameters)
           
            model_emission_value = tracker.final_emissions * 1000
            training_emissions = {
                "model_emissions": "{:.2f}".format(model_emission_value),  # Convert to grams
                "epoch_emissions": "{:.2f}".format(model_emission_value / parameters.EPOCHS),  # Convert to grams per image
                "total_emissions": "{:.2f}".format(model_emission_value * multiplier)
                }
            
            # Track emission for evaluation
            with EmissionsTracker() as tracker:
                EvaluationSession.evaluate(
                    training_session.test_img_paths, 
                    training_session.test_mask_paths, 
                    model, 
                    SavesManager.CURRENT_SAVE_PATHS.DIRECTORY_NAME, 
                    logger
                    )
                
            model_emission_value = tracker.final_emissions * 1000
            evaluation_emissions = {
                "model_emissions": "{:.2f}".format(model_emission_value),  # Convert to grams
                "inference_emissions": "{:.2f}".format(model_emission_value / len(training_session.test_img_paths)),  # Convert to grams per image
                "total_emissions": "{:.2f}".format(model_emission_value * multiplier)
                }

            # Prepare emissions data
            emissions = {
                "dataset": dataset.DATASET_NAME,
                "model": model_class.NAME,
                "epochs": parameters.EPOCHS,
                "training": training_emissions,
                "evaluation": evaluation_emissions
            }
            
            # Create save path if it does not exist
            save_path = os.path.join(Paths.EMISSIONS, SavesManager.generate_model_dir_name(model.name))    
            os.makedirs(save_path)
            
            # Save emissions data
            SavesManager.save_json(os.path.join(save_path, SaveFilename.EMISSIONS.value), emissions) 


     
