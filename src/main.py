from logging import Logger
import os
import shutil

import numpy as np
from config import DatasetRegistry, ParametersRegistry, SaveFilename
from models.UNetL import UNetL
from models.UNet import UNet
from models.UNetPPL import UNetPPL
from models.UNetPP import UNetPP
from models.TransUNet import TransUNet, PretrainedTransUNet
from keras.models import Model
from tensorflow.keras.models import load_model # type: ignore
from predict import EvaluationSession
from models.LightMUNet import LightMUNet
from utils.Denoiser import Denoiser
from utils.SavesManager import SavesManager
from utils.CO2Tracker import CO2Tracker
from utils.DatasetUtils import DatasetUtils
from train import TrainingAndEvaluationSession, TrainingSession
from utils.misc import config_gpu, setup_logger

# Functions that can be executed in main
test_prediction = EvaluationSession.test_prediction
test_denoising = DatasetUtils.test_denoising
denoise_dataset = DatasetUtils.denoise_dataset

# Training and evaluation session
def train_eval_session(logger: Logger):
    datasets = [DatasetRegistry.PALSAR, DatasetRegistry.SENTINEL]
    models = [UNetL, UNet, UNetPPL, UNetPP, TransUNet, PretrainedTransUNet, LightMUNet]
    params = [ParametersRegistry.AUTOMATIC]
    
    for dataset in datasets:
        TrainingAndEvaluationSession(logger, dataset, models, params, True, True, Denoiser.gaussian_blur).start_model_wise()
    
# Find best combination of denoising filters
def determine_best_filters(logger: Logger):
    # Instantiate objects needed for training and evaluation
    datasets = [DatasetRegistry.PALSAR, DatasetRegistry.SENTINEL]
    models = [UNetL]
    params = [ParametersRegistry.AUTOMATIC]
    
    # Create session
    for dataset in datasets:
        TrainingAndEvaluationSession(logger, dataset, models, params, False, False, Denoiser.gaussian_blur).start_model_wise()
        TrainingAndEvaluationSession(logger, dataset, models, params, False, False, Denoiser.median_blur).start_model_wise()
        TrainingAndEvaluationSession(logger, dataset, models, params, False, False, Denoiser.box_filter).start_model_wise()
        TrainingAndEvaluationSession(logger, dataset, models, params, False, False, Denoiser.bilateral_filter).start_model_wise()
        TrainingAndEvaluationSession(logger, dataset, models, params, False, False, None).start_model_wise()
        
        TrainingAndEvaluationSession(logger, dataset, models, params, False, True, Denoiser.gaussian_blur).start_model_wise()
        TrainingAndEvaluationSession(logger, dataset, models, params, False, True, Denoiser.median_blur).start_model_wise()
        TrainingAndEvaluationSession(logger, dataset, models, params, False, True, Denoiser.box_filter).start_model_wise()
        TrainingAndEvaluationSession(logger, dataset, models, params, False, True, Denoiser.bilateral_filter).start_model_wise()
        TrainingAndEvaluationSession(logger, dataset, models, params, False, True, None).start_model_wise()

# Track training and evaluation emissions
def track_emissions_session(logger: Logger): 
    # Instantiate objects needed for emission tracking
    datasets = [DatasetRegistry.PALSAR, DatasetRegistry.SENTINEL]
    models = [UNetL, UNet, UNetPPL, UNetPP, TransUNet, PretrainedTransUNet, LightMUNet]
    
    # Create session
    for dataset in datasets:
        CO2Tracker.track_emissions(logger, dataset, models)

# Determine best save models
def determine_best_models(logger: Logger):
    # Instantiate objects needed for training and evaluation
    datasets = [DatasetRegistry.PALSAR, DatasetRegistry.SENTINEL]
    models = [UNetL, UNet, UNetPPL, UNetPP, TransUNet, PretrainedTransUNet, LightMUNet]
    
    for dataset in datasets:
        for model in models:
            logger.info(f"Start reading evaluations on {dataset.DATASET_NAME} dataset")
            model_saves_paths = SavesManager.get_all_saves_paths_by_model_name(model.NAME, dataset.MODEL_SAVES_PATH)
            
            # Best model is determined based on mIoU
            best_mean_iou = 0.0
            best_model_save_path = ""
            for model_save_path in model_saves_paths:
                logger.info(f"Start reading evaluations of {model.NAME} models")
                evaluation = SavesManager.load_evaluation(model_save_path)
                mean_iou = evaluation.get("iou")
                
                if mean_iou > best_mean_iou:
                    best_mean_iou = mean_iou
                    best_model_save_path = model_save_path
                    logger.info(f"New best network with {str(mean_iou)} mIoU at: {model_save_path}")
            
            # Determine destination path and copy best model files to destination    
            destination_path = os.path.join(dataset.MODEL_SAVES_PATH, "best_models", model.NAME)
            os.makedirs(destination_path, exist_ok=True)
            shutil.copytree(best_model_save_path, destination_path, dirs_exist_ok=True)  

# Load and evaluate LightMUNet model    
def load_and_evaluate_LightMUNet_model(logger: Logger, dir_name: str = "LightMUNet_2025-06-27_07-01-03"):
    a = TrainingSession(logger, DatasetRegistry.PALSAR, [LightMUNet])
    model: LightMUNet = load_model(
        os.path.join(os.getcwd(), f"saves/palsar/LightMUNet/{dir_name}/model.tf"),
        custom_objects={'light_mu_net_2': LightMUNet}
        )
    
    model.inp_channels = 1
    
    SavesManager.set_save_paths(os.path.join(os.getcwd(), "saves/palsar"), dir_name)
    EvaluationSession.evaluate(a.test_img_paths, 
                               a.test_mask_paths, 
                               model,
                               dir_name, 
                               logger)

# Load and evaluate TransUNet model    
def load_and_evaluate_TransUNet_model(logger: Logger, dir_name: str = "TransUNet_2025-06-18_22-47-13"):
    # Choose a function to execute
    a = TrainingSession(logger, DatasetRegistry.PALSAR, [TransUNet])
    SavesManager.set_save_paths(os.path.join(os.getcwd(), "saves/palsar"), dir_name)
    EvaluationSession.evaluate(a.test_img_paths, 
                               a.test_mask_paths, 
                               load_model(
                                   os.path.join(os.getcwd(), f"saves/palsar/TransUNet/{dir_name}/model.hdf5"),
                                   custom_objects={'TransUNet': TransUNet}
                                   ), 
                               dir_name, 
                               logger)

# Load and evaluate UNetPP model    
def load_and_evaluate_UNetPP_model(logger: Logger, dir_name: str = "UNetPP_2025-06-19_22-30-51"):
    # Choose a function to execute
    a = TrainingSession(logger, DatasetRegistry.PALSAR, [UNetPP])
    SavesManager.set_save_paths(os.path.join(os.getcwd(), "saves/palsar"), dir_name)
    EvaluationSession.evaluate(a.test_img_paths, 
                               a.test_mask_paths, 
                               load_model(
                                   os.path.join(os.getcwd(), f"saves/palsar/UNetPP/{dir_name}/model.hdf5"),
                                   custom_objects={'UNetPP': UNetPP}
                                   ), 
                               dir_name, 
                               logger)
    
                        

# Main
def main(logger: Logger):
    
    ''' Choose a function to execute '''
         
    # Main functions             
    #denoise_dataset(DatasetRegistry.SENTINEL)
    #determine_best_filters(logger)    
    #train_eval_session(logger)
    #track_emissions_session(logger)
    #determine_best_models(logger)
    
    # Test denoising and prediction of single image
    #test_denoising()
    #test_prediction()
    
    # Model summaries
    #UNetL.show_model_summary()         #   1.9 million parameters
    #UNetPPL.show_model_summary()       #   2.2 million parameters
    #UNet.show_model_summary()          #   7.8 million parameters
    #UNetPP.show_model_summary()        #   9.0 million parameters
    #TransUNet.show_model_summary()     # 100.9 million parameters
    #LightMUNet.show_model_summary()    #   8.5 million parameters
    
    # Custom loaders for models that had trained but crashed when evaluating
    #load_and_evaluate_LightMUNet_model(logger)
    #load_and_evaluate_TransUNet_model(logger)
    #load_and_evaluate_UNetPP_model(logger)

    return

# Entry point
if __name__ == "__main__":
    # Prepare GPU
    config_gpu()
    
    # Setup logger
    logger = setup_logger()
    
    # Execute main and catch exceptions
    try:
        main(logger)
        logger.info("Execution completed without exceptions")
    except Exception as e:
        logger.info("Execution stopped because an exception was raised")
        logger.info(e)
        raise e
        