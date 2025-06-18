from config import INPUT_WIDTH, ParametersRegistry
from dataclass import Parameters
from utils.misc import ParametersLoaderModel
from transunet import TransUNet

class TransUNet(ParametersLoaderModel):
    NAME = "TransUNet"
    
    @classmethod
    def get_parameters_values(cls) -> list[Parameters]:
        return cls.generate_parameters_list(ParametersRegistry.UNET)
    
    def __init__(self, image_size=INPUT_WIDTH, num_classes=1, pretrain=False):
        return TransUNet(image_size=image_size, num_classes=num_classes, pretrain=pretrain, name=self.NAME)
         