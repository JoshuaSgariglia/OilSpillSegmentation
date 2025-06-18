from config import INPUT_WIDTH, DatasetRegistry, ParametersRegistry
from models.UNetL import UNetL
from models.UNet import UNet
from models.UNetPPL import UNetPPL
from models.UNetPP import UNetPP
from models.TransUNet import TransUNet
from predict import EvaluationSession
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
    models = [UNetL, UNet]
    params = [ParametersRegistry.AUTOMATIC]
    
    # Create session
    for dataset in datasets:
        TrainingAndEvaluationSession(logger, dataset, models, params, True).start_model_wise()


def main():
    # Prepare GPU
    config_gpu()

    # Choose a function to execute
    train_eval_session()
    #test_prediction()
    #test_denoising()
    #denoise_dataset()

if __name__ == "__main__":
    main()