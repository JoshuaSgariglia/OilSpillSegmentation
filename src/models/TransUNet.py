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
    
    '''  
    def get_config(self):
        config = super().get_config()
        config.update({
            "image_size": self.image_size,
            "num_classes": self.num_classes,
            "pretrain": self.pretrain,
            **self.kwargs
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Get the argument names of __init__ (excluding 'self')
        sig = inspect.signature(cls.__init__)
        valid_keys = set(sig.parameters.keys()) - {'self', '*args', '**kwargs'}
        
        # Filter config to only include valid keys
        filtered_config = {k: v for k, v in config.items() if k in valid_keys}
        
        return cls(**filtered_config)
    '''
    