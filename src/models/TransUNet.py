from transunet import TransUNet as ImportedTransUNet
from utils.misc import ParametersLoaderModel
from config import INPUT_WIDTH, ParametersRegistry

class TransUNet(ParametersLoaderModel):
    NAME = "TransUNet"

    @classmethod
    def get_parameters_values(cls):
        return cls.generate_parameters_list(ParametersRegistry.TRANSUNET)

    def __init__(self, image_size=INPUT_WIDTH, num_classes=1, pretrain=False, **kwargs):
        # Instantiate the imported TransUNet model
        inner_model = ImportedTransUNet(image_size=image_size, num_classes=num_classes, pretrain=pretrain, **kwargs)

        # Call super().__init__ with the imported model's inputs and outputs
        super().__init__(inputs=inner_model.input, outputs=inner_model.output, name='TransUNet')