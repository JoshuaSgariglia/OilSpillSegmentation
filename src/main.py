from config import INPUT_WIDTH, DatasetRegistry, ParametersRegistry
from models.UNetL import UNetL
from models.UNet import UNet
from models.UNetPPL import UNetPPL
from models.UNetPP import UNetPP
from models.TransUNet import TransUNet
from keras.models import Model
from predict import EvaluationSession
from utils.CO2Tracker import CO2Tracker
from utils.DatasetUtils import DatasetUtils
from train import TrainingAndEvaluationSession
from utils.misc import config_gpu, setup_logger

# Functions that can be executed in main
test_prediction = EvaluationSession.test_prediction
test_denoising = DatasetUtils.test_denoising
denoise_dataset = DatasetUtils.denoise_dataset

# Training and evaluation session
def train_eval_session():
    # Instantiate objects needed for training and evaluation
    logger = setup_logger()
    datasets = [DatasetRegistry.PALSAR, DatasetRegistry.SENTINEL]
    models = [TransUNet]
    params = [ParametersRegistry.AUTOMATIC]
    
    # Create session
    for dataset in datasets:
        TrainingAndEvaluationSession(logger, dataset, models, params, True).start_model_wise()

# Track training and evaluation emissions
def track_emissions_session(): 
    logger = setup_logger()
    datasets = [DatasetRegistry.PALSAR, DatasetRegistry.SENTINEL]
    models = [UNet]
    
    # Create session
    for dataset in datasets:
        CO2Tracker.track_emissions(logger, dataset, models)

def main():
    # Prepare GPU
    config_gpu()

    # Choose a function to execute
    train_eval_session()
    #track_emissions_session()
    #test_prediction()
    #test_denoising()
    #denoise_dataset(DatasetRegistry.SENTINEL)
    
    # Model summaries
    #UNetL.show_model_summary()         # 1.9 million parameters
    #UNetPPL.show_model_summary()       # 2.2 million parameters
    #UNet.show_model_summary()          # 7.6 million parameters
    #UNetPP.show_model_summary()        # 9.0 million parameters
    #TransUNet.show_model_summary()     # million parameters

if __name__ == "__main__":
    main()