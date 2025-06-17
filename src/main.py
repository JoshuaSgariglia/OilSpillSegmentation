from config import INPUT_WIDTH, DatasetRegistry, ParametersRegistry
from models.UNetL import UNetL
from models.UNet import UNet
from models.UNetPPL import UNetPPL
from models.UNetPP import UNetPP
from predict import EvaluationSession
from utils.DatasetUtils import DatasetUtils
from train import TrainingAndEvaluationSession
from utils.misc import config_gpu, setup_logger
from transunet import TransUNet

# TransUNet Initializer
def TUNet():
    return TransUNet(image_size=INPUT_WIDTH, num_classes=1, pretrain=False)

# Functions that can be executed in main
test_prediction = EvaluationSession.test_prediction
test_denoising = DatasetUtils.test_denoising
denoise_dataset = DatasetUtils.denoise_dataset


def train_eval_session():
    # Instantiate objects needed for training and evaluation
    datasets = [DatasetRegistry.PALSAR, DatasetRegistry.SENTINEL]
    models = [UNetL(), UNet()]
    params = [ParametersRegistry.AUTOMATIC]
    logger = setup_logger()
    
    # Create session
    for dataset in datasets:
        TrainingAndEvaluationSession(dataset, models, params, logger).start_model_wise()


def main():
    # Prepare GPU
    config_gpu()

    train_eval_session()
    #test_prediction()
    #test_denoising()
    #denoise_dataset()

if __name__ == "__main__":
    main()