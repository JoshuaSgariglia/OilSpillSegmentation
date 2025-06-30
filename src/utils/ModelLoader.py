import os
from models import LightMUNet
from models.UNetL import UNetL
from models.TransUNet import PretrainedTransUNet, TransUNet
from models.UNet import UNet
from models.UNetPP import UNetPP
from models.UNetPPL import UNetPPL
from tensorflow.keras.models import load_model # type: ignore
from keras.models import Model


# Loads custom models
class ModelLoader:
    @staticmethod
    def load(dir_name: str, model_class: type[Model], full_path: bool = False) -> Model:
        model: Model = load_model(
            os.path.join(os.getcwd(), dir_name if full_path else f"saves/palsar/{model_class.NAME}/{dir_name}/model.tf"),
            custom_objects={model_class().name: model_class}
            )
        
        if model_class is LightMUNet:
            model.inp_channels = 1
        
        return model
    
    @classmethod
    def load_safe(cls, dir_name: str, full_path: bool = False) -> Model:
        model: Model
        for model_class in [UNetL, UNet, UNetPPL, UNetPP, TransUNet, PretrainedTransUNet, LightMUNet]:
            
            try:
                model = cls.load(dir_name, model_class, full_path)
                break
                    
            finally:
                continue
        
        return model