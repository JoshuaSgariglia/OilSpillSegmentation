from __future__ import annotations
import json
import os
from config import SaveFilename
from utils.misc import current_datetime, Parameters
from keras.models import Model
from tensorflow.keras.models import load_model # type: ignore     

        
        
class SavesManager:
    CURRENT_SAVE_PATHS: SavePaths
    
    # Generate directory name for specific trained model
    @staticmethod
    def generate_model_dir_name(model_name: str) -> str:
        return f"{model_name}_{current_datetime()}"
    
    # Get path for saves of a type of model (e.g. UNet)
    @staticmethod
    def get_model_type_saves_path(model_name: str, saves_dir: str) -> str:
        return os.path.join(saves_dir, model_name)
    
    @classmethod
    def get_model_saves_path(cls, model_dir_name: str, saves_dir: str) -> str:
        model_name = model_dir_name.split('_')[0]
        return os.path.join(cls.get_model_type_saves_path(model_name, saves_dir), model_dir_name)
    
    @classmethod
    def generate_model_saves_path(cls, model_name: str, saves_dir: str) -> str:
        return os.path.join(cls.get_model_type_saves_path(model_name, saves_dir), cls.generate_model_dir_name(model_name))
    
    # Generate and set new save paths for a specific model
    @classmethod
    def set_generated_save_paths(cls, saves_dir: str, model_name: str) -> SavePaths:
        cls.CURRENT_SAVE_PATHS = cls.SavePaths.GenerateFromModelName(saves_dir, model_name)
        return cls.CURRENT_SAVE_PATHS
    
    # Set new save paths manually
    @classmethod
    def set_save_paths(cls, saves_dir: str, model_dir_name: str) -> SavePaths:
        cls.CURRENT_SAVE_PATHS = cls.SavePaths(saves_dir, model_dir_name)
        return cls.CURRENT_SAVE_PATHS
    
    # Reset current_save_path
    @classmethod
    def reset_save_paths(cls) -> None:
        cls.CURRENT_SAVE_PATHS = None
    
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
        cls.save_json(cls.CURRENT_SAVE_PATHS.EVALUATION, evaluation)
            
    # Loading evaluation from JSON file
    @classmethod
    def load_evaluation(cls) -> dict:
        evaluation_dict = cls.load_json(cls.CURRENT_SAVE_PATHS.EVALUATION)
        return evaluation_dict
    
    # Save time in JSON file
    @classmethod
    def save_time_metrics(cls, time_metrics: dict) -> None:
        cls.save_json(cls.CURRENT_SAVE_PATHS.TIME, time_metrics)
            
    # Loading time from JSON file
    @classmethod
    def load_time_metrics(cls) -> dict:
        time_metrics_dict = cls.load_json(cls.CURRENT_SAVE_PATHS.TIME)
        return time_metrics_dict
    
    # Loading evaluation from JSON file
    @classmethod
    def load_model(cls) -> Model:
        model = load_model(cls.CURRENT_SAVE_PATHS.MODEL)
        return model
    
    # Class that represents all the save paths for a specific model
    class SavePaths:
        def __init__(self, saves_dir: str, model_dir_name: str):
            saves_path = SavesManager.get_model_saves_path(model_dir_name, saves_dir)
            
            # Create the save directory if it doesn't exist
            os.makedirs(saves_path, exist_ok=True)
            
            # Inner function
            def join_with_save_path(save_filename: SaveFilename):
                return os.path.join(saves_path, save_filename.value)
            
            # Properties
            self.DIRECTORY_NAME = model_dir_name
            self.DIRECTORY = saves_path
            self.EVALUATION = join_with_save_path(SaveFilename.EVALUATION)
            self.MODEL = join_with_save_path(SaveFilename.MODEL)
            self.PARAMETERS = join_with_save_path(SaveFilename.PARAMETERS)
            self.TIME = join_with_save_path(SaveFilename.TIME)
            self.TRAINING = join_with_save_path(SaveFilename.TRAINING)
        
        @classmethod
        def GenerateFromModelName(cls, saves_dir: str, model_name: str):
            # Generate directory name for the model
            model_dir_name = SavesManager.generate_model_dir_name(model_name)
            
            # Generate and return corresponding save paths
            return cls(saves_dir, model_dir_name)
            
