import os
from config import DatasetRegistry, Paths
from models.UNetL import UNetL
from predict import EvaluationSession
from train import TrainingAndEvaluationSession, TrainingSession
from utils.misc import Parameters, config_gpu, setup_logger


def main():
    # Prepare GPU
    config_gpu()
    
    # Instantiate objects needed for training and evaluation
    dataset = DatasetRegistry.PALSAR
    models = [UNetL()]
    params = [Parameters()]
    logger = setup_logger()
    
    # Create session
    TrainingAndEvaluationSession(dataset, models, params, logger).start_model_wise()

if __name__ == "__main__":
    main()