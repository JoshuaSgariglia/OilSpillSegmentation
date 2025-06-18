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
                model_classes: list[type[ParametersLoaderModel]],
                multiplier: int = 1
                ): 
        
        # Instantiate training session
        training_session = TrainingSession(logger, dataset, model_classes)
        
        for model_class in model_classes:
            # Get some parameters for the model and the model
            parameters: Parameters = model_class.generate_parameters_list()[0]
            model = model_class()

            # Track emission for training
            with EmissionsTracker() as tracker:
                training_session.train(model,parameters)
           
            training_emissions = {
                "model_emissions": "{:.2f}".format(tracker.final_emissions * 1000),  # Convert to grams
                "epochs_emissions": ("{:.4f}".format(tracker.final_inference_emissions * 1000)/len(parameters.EPOCHS)),  # Convert to grams per image
                "emissions_data": tracker.final_emissions_data,
                }
            
            if multiplier > 1:
                training_emissions = training_emissions.update(
                    {"total_emissions": training_emissions.get("model_emissions")*multiplier}
                )
            
            # Track emission for evaluation
            with EmissionsTracker()as tracker:
                EvaluationSession.evaluate(
                    training_session.test_img_paths, 
                    training_session.test_mask_paths, 
                    model, 
                    SavesManager.CURRENT_SAVE_PATHS.DIRECTORY_NAME, 
                    logger
                    )
                
            evaluation_emissions = {
                "model_emissions": "{:.2f}".format(tracker.final_emissions * 1000),  # Convert to grams
                "inference_emissions": "{:.2f}".format((tracker.final_inference_emissions * 1000)/len(training_session.test_img_paths)),  # Convert to grams per image
                "emissions_data": tracker.final_emissions_data,
                }
            
            if multiplier > 1:
                evaluation_emissions = evaluation_emissions.update(
                    {"total_emissions": evaluation_emissions.get("model_emissions")*multiplier}
                )

            # Prepare emissions data
            dict = {
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
            SavesManager.save_json(os.path.join(save_path, SaveFilename.EMISSIONS.value), dict) 


     
