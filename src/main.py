import os
from config import DatasetRegistry, Paths
from models.UNetL import UNetL
from predict import EvaluationSession
from train import TrainingSession
from utils.misc import Parameters, config_gpu, current_datetime, setup_logger


def main():
    # Prepare GPU
    config_gpu()
    
    # Create directory for logs if it doesn't exist
    os.makedirs(Paths.LOGS, exist_ok=True)
    
    # Instantiate objects needed for training
    models = [UNetL()]
    params = [Parameters()]
    logger = setup_logger(os.path.join(Paths.LOGS, f"log_{current_datetime()}.txt"))
    
    # Initialize objects needed for evaluation
    model_names = [model.name for model in models]
    
    training_session = TrainingSession(DatasetRegistry.PALSAR, models, params, logger)
    
    eval_session = EvaluationSession(training_session.test_img_paths, training_session.test_mask_paths, model_names, logger)

if __name__ == "__main__":
    main()