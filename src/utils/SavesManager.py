from __future__ import annotations
import json
import os
from config import Paths, SaveFilename
from utils.misc import current_datetime, Parameters
from keras.models import Model
from tensorflow.keras.models import load_model # type: ignore     

        
        
class SavesManager:
    CURRENT_SAVE_PATHS: SavePaths
    
    # Get path for saves of a type of model (e.g. UNet)
    @staticmethod
    def get_model_type_saves_path(model_name: str) -> str:
        return os.path.join(Paths.SAVES, model_name)
    
    # Generate directory name for specific trained model
    @staticmethod
    def generate_model_dir_name(model_name: str) -> str:
        return f"{model_name}_{current_datetime()}"
    
    # Generate saves path for a specific model
    @classmethod
    def generate_model_saves_path(cls, model_name: str) -> str:
        return os.path.join(Paths.SAVES, model_name, cls.generate_model_dir_name(model_name))
    
    # Generate and set new save paths for a specific model
    @classmethod
    def set_save_paths(cls, model_name: str) -> SavePaths:
        cls.CURRENT_SAVE_PATHS = cls.SavePaths(model_name)
        return cls.CURRENT_SAVE_PATHS
    
    # Save data in JSON file
    @staticmethod
    def save_json(save_path: str, data: dict):
        with open(save_path, "w") as outfile:
            json.dump(data, outfile)
    
    # Load data in JSON file
    @staticmethod    
    def load_json(save_path: str):
        with open(save_path, "r") as infile:
            return json.load(infile)
    
    # Save parameters in JSON file
    @classmethod
    def save_parameters(cls, parameters: Parameters) -> None:
        cls.save_json(cls.CURRENT_SAVE_PATHS.PARAMETERS, parameters.__dict__)
            
    # Loading parameters from JSON file
    @classmethod
    def load_parameters(cls) -> Parameters:
        params_dict = cls.load_json(cls.CURRENT_SAVE_PATHS.PARAMETERS)
        return Parameters(**params_dict)
    
    # Save evaluation in JSON file
    @classmethod
    def save_evaluation(cls, evaluation: dict) -> None:
        cls.save_json(cls.CURRENT_SAVE_PATHS.EVALUATION, evaluation.__dict__)
            
    # Loading evaluation from JSON file
    @classmethod
    def load_evaluation(cls) -> dict:
        evaluation_dict = cls.load_json(cls.CURRENT_SAVE_PATHS.EVALUATION)
        return evaluation_dict
    
    # Loading evaluation from JSON file
    @classmethod
    def load_model(cls) -> Model:
        model = load_model(cls.CURRENT_SAVE_PATHS.MODEL)
        return model
    
    # Class that represents all the save paths for a specific model
    class SavePaths:
        def __init__(self, model_name: str):
            # Get save directory path
            saves_path: str = SavesManager.generate_model_saves_path(model_name)
            
            # Create the save directory if it doesn't exist
            os.makedirs(saves_path, exist_ok=True)
            
            # Inner function
            def join_with_save_path(save_filename: SaveFilename):
                return os.path.join(saves_path, save_filename.value)
            
            # Properties
            self.DIRECTORY = saves_path
            self.EVALUATION = join_with_save_path(SaveFilename.EVALUATION)
            self.MODEL = join_with_save_path(SaveFilename.MODEL)
            self.PARAMETERS = join_with_save_path(SaveFilename.PARAMETERS)
            self.TIME = join_with_save_path(SaveFilename.TIME)
            self.TRAINING = join_with_save_path(SaveFilename.TRAINING)
            
            
           
        
    