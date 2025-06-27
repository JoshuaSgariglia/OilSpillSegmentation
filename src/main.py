from logging import Logger
import os
from config import DatasetRegistry, ParametersRegistry
from models.UNetL import UNetL
from models.UNet import UNet
from models.UNetPPL import UNetPPL
from models.UNetPP import UNetPP
from models.TransUNet import TransUNet
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
    # Instantiate objects needed for training and evaluation
    datasets = [DatasetRegistry.PALSAR, DatasetRegistry.SENTINEL]
    models = [LightMUNet]
    params = [ParametersRegistry.AUTOMATIC]
    
    for dataset in datasets:
        TrainingAndEvaluationSession(logger, dataset, models, params, False, True, Denoiser.gaussian_blur).start_model_wise()
    
    '''
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
    '''

# Track training and evaluation emissions
def track_emissions_session(logger: Logger): 
    # Instantiate objects needed for emission tracking
    datasets = [DatasetRegistry.SENTINEL]
    models = [UNetPP, TransUNet]
    
    # Create session
    for dataset in datasets:
        CO2Tracker.track_emissions(logger, dataset, models)

def main(logger: Logger):
    
    '''
    logger = setup_logger()
    # Choose a function to execute
    a = TrainingSession(logger, DatasetRegistry.PALSAR, [TransUNet])
    SavesManager.set_save_paths(os.path.join(os.getcwd(), "saves/palsar"), "TransUNet_2025-06-18_22-47-13")
    EvaluationSession.evaluate(a.test_img_paths, 
                               a.test_mask_paths, 
                               load_model(
                                   os.path.join(os.getcwd(), "saves/palsar/TransUNet/TransUNet_2025-06-18_22-47-13/model.hdf5"),
                                   custom_objects={'TransUNet': TransUNet}
                                   ), 
                               "TransUNet_2025-06-18_22-47-13", 
                               logger)
    '''
    
    '''
    logger = setup_logger()
    # Choose a function to execute
    a = TrainingSession(logger, DatasetRegistry.PALSAR, [UNetPP])
    SavesManager.set_save_paths(os.path.join(os.getcwd(), "saves/palsar"), "UNetPP_2025-06-19_22-30-51")
    EvaluationSession.evaluate(a.test_img_paths, 
                               a.test_mask_paths, 
                               load_model(
                                   os.path.join(os.getcwd(), "saves/palsar/UNetPP/UNetPP_2025-06-19_22-30-51/model.hdf5"),
                                   custom_objects={'UNetPP': UNetPP}
                                   ), 
                               "UNetPP_2025-06-19_22-30-51", 
                               logger)
    '''
    
                               
    train_eval_session(logger)
    #track_emissions_session(logger)
    #test_prediction()
    #test_denoising()
    #denoise_dataset(DatasetRegistry.SENTINEL)
    
    # Model summaries
    #UNetL.show_model_summary()         #   1.9 million parameters
    #UNetPPL.show_model_summary()       #   2.2 million parameters
    #UNet.show_model_summary()          #   7.8 million parameters
    #UNetPP.show_model_summary()        #   9.0 million parameters
    #TransUNet.show_model_summary()     # 100.9 million parameters
    LightMUNet.show_model_summary()     #   8.6 million parameters

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
        