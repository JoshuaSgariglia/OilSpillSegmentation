import os
from config import DatasetRegistry, Parameters, Paths
from models.UNetL import UNetL
from predict import EvaluationSession
from train import TrainingSession
from utils import config_gpu, setup_logger


def main():
    # Prepare GPU
    config_gpu()
    
    # Create directory for logs if it doesn't exist
    log_path = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_path, exist_ok=True)
    
    # Instantiate objects needed for training
    models = [UNetL()]
    params = [Parameters()]
    logger = setup_logger(os.path.join(log_path, f"log_{Paths.current_datetime()}.txt"))
    
    # Initialize objects needed for evaluation
    model_names = [model.name for model in models]
    
    training_session = TrainingSession(DatasetRegistry.PALSAR, models, params, logger)
    
    eval_session = EvaluationSession(training_session.test_img_paths, training_session.test_mask_paths, model_names, logger)

if __name__ == "__main__":
    main()