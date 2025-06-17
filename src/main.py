from config import DatasetRegistry
from models.UNetL import UNetL
from utils.DatasetUtils import DatasetUtils
from train import TrainingAndEvaluationSession
from utils.misc import Parameters, config_gpu, setup_logger

test_denoising = DatasetUtils.test_denoising
denoise_dataset = DatasetUtils.denoise_dataset

def train_eval_session():
    # Instantiate objects needed for training and evaluation
    dataset = DatasetRegistry.PALSAR
    models = [UNetL()]
    params = [Parameters()]
    logger = setup_logger()
    
    # Create session
    TrainingAndEvaluationSession(dataset, models, params, logger).start_model_wise()


def main():
    # Prepare GPU
    config_gpu()

    train_eval_session()
    #test_denoising()
    #denoise_dataset()

if __name__ == "__main__":
    main()