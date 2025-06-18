from logging import Logger
import os
from codecarbon import EmissionTracker
from dataclass import DatasetPaths
from predict import EvaluationSession
from config import Paths, SaveFilename
from utils import SavesManager
from train import TrainingSession
from utils.misc import ParametersLoaderModel

class CO2Tracker:
    @staticmethod
    def track_emissions(  
                logger: Logger,
                dataset: DatasetPaths, 
                model_classes: list[type[ParametersLoaderModel]] ) : 
        
        training_session = TrainingSession(logger, dataset, model_classes)
        for  model_class in model_classes:
        
            parameters=model_class.generate_parameters_list()[0]
            model=model_class()


            with EmissionTracker()as tracker:
                
                training_session.train(model,parameters)
           
            dict_training= {
                "emissions": tracker.final_emissions * 1000,  # Convert to grams
                "epochs_emissions": (tracker.final_inference_emissions * 1000)/len(parameters.EPOCHS),  # Convert to grams per image
                "emissions_data": tracker.final_emissions_data,
                }
             
            
            with EmissionTracker()as tracker:
                EvaluationSession.evaluate(
                    training_session.test_img_paths, 
                    training_session.test_mask_paths, 
                    model, 
                    SavesManager.CURRENT_SAVE_PATHS.DIRECTORY_NAME, 
                    self.logger
                    )
            dict_evaluation = {
                
                "emissions": tracker.final_emissions * 1000,  # Convert to grams
                "inference_emissions": (tracker.final_inference_emissions * 1000)/len(training_session.test_img_paths),  # Convert to grams per image
                "emissions_data": tracker.final_emissions_data,
                }

            dict= {
                "dataset": dataset.DATASET_NAME,
                "model": model_class.NAME,
                "epochs": parameters.EPOCHS,
                "training_emissions": dict_training,
                "evaluation_emissions": dict_evaluation
            }
            SavesManager.save_json(os.path.join(Paths.EMISSIONS,SavesManager.generate_model_dir_name(model.name), SaveFilename.EMISSIONS.value), dict) 


     
