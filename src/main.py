from config import DatasetRegistry
from models.UNetL import UNetL
from utils.DatasetUtils import test_denoising
from train import TrainingAndEvaluationSession
from utils.misc import Parameters, config_gpu, setup_logger

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
    
    test_denoising()

if __name__ == "__main__":
    main()