from logging import Logger
import cv2
import numpy as np
from dataclass import DatasetPaths
from utils.ModelLoader import ModelLoader
from utils.Denoiser import Denoiser
from utils.DatasetUtils import DatasetUtils
from utils.SavesManager import SavesManager
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from config import DatasetRegistry, Paths
import os
from PIL import Image
from keras.models import Model

 
class EvaluationSession:

    def __init__(self, saves_dir: str, image_paths: list[str], mask_paths: list[str], model_names: list[str], logger: Logger):
        self.saves_dir = saves_dir
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.model_names = model_names
        self.logger = logger
    
    #Evaluate all models
    def evaluate_all(self):
        self.logger.info("Start evaluating models")
        
        # Compute all
        for model_name in self.model_names:
                    
            self.logger.info(f"Start evaluating {model_name} models")
            
            # List all single subdirectories of model saves
            model_dir_names = SavesManager.get_all_dir_names_by_model_name(model_name, self.saves_dir)
            
            for model_dir_name in model_dir_names:
                
                # Get save paths for the model
                model_save_paths = SavesManager.set_save_paths(self.saves_dir, model_dir_name)
                
                # Only evaluates models not already evaluated
                if not os.path.isfile(model_save_paths.EVALUATION):
                
                    # Load the model
                    model = SavesManager.load_model()
                    
                    self.logger.info(f"Model {model_dir_name} loaded successfully")
                    
                    # Evaluate model and save computed metrics
                    self.evaluate(self.image_paths, self.mask_paths, model, model_dir_name, self.logger)
                        
                else:
                    self.logger.info(f"Model {model_dir_name} already evaluated (skipped)")
                
            self.logger.info(f"Completed evaluation of all {model_name} models")
        self.logger.info(f"Completed evaluation of all models")
        
        # Reset save paths
        SavesManager.reset_save_paths()
        
        self.logger.info(f"Evaluation session ended")
        
    # Evaluate model
    @classmethod
    def evaluate(cls, image_paths: str, mask_paths: str, model: Model, model_dir_name: str, logger: Logger, save: bool = True, extra_filter: callable = Denoiser.gaussian_blur):
        
        logger.info(f"Start evaluating model {model_dir_name}")
        
        # Initialize empty tensors
        seg_list, pr_list = np.array([]).astype('uint8'), np.array([]).astype('uint8')
        
        logger.info("Evaluating model...")
        
        # Predict the image
        for image_path, mask_path in zip(image_paths, mask_paths):  
            mask, predict = cls.predict_image(image_path, mask_path, model, extra_filter) 
            
            # Append images in the tensors
            seg_list = np.hstack((seg_list, mask.flatten().astype('uint8')))
            pr_list = np.hstack((pr_list, predict.flatten().astype('uint8')))
            

        # Compute metrics
        conf_matrix = confusion_matrix(seg_list, pr_list, labels=[0, 1])
        accuracy = accuracy_score(seg_list, pr_list)
        precision = precision_score(seg_list, pr_list, average = None)
        recall = recall_score(seg_list, pr_list, average = None)
        f1 = f1_score(seg_list, pr_list, average = None)
        iou = cls.compute_iou(conf_matrix)
        
        logger.info(f"Model {model_dir_name} evaluated successfully")
        
        # Save metrics into dictionary
        evaluation = dict(
            accuracy = accuracy,
            precision_background = precision[0],
            precision_oil_spill = precision[1],
            precision = np.mean(list(precision)),
            recall_background = recall[0],
            recall_oil_spill = recall[1],
            recall = np.mean(list(recall)),
            f1_background = f1[0],
            f1_oil_spill = f1[1],
            f1 = np.mean(f1),
            conf_matrix = conf_matrix.tolist(),
            iou_background = iou[0],
            iou_oil_spill = iou[1],
            iou = np.mean(list(iou))
        )
            
        # Save metrics dictionary in JSOn file
        if save:
            SavesManager.save_evaluation(evaluation)
            
        logger.info(f"Model {model_dir_name} metrics saved successfully")
                    
    # Predict image
    @staticmethod
    def predict_image(image_path: str, mask_path: str, model: Model, extra_filter: callable = Denoiser.gaussian_blur):
        # Load image and its mask (includes preprocessing)
        image = DatasetUtils.load_image(image_path, extra_filter)
        mask = DatasetUtils.load_mask(mask_path)
        
        # Get number of channels
        if hasattr(model, "inp_channels"):
            n_channels = model.inp_channels
        else:
            n_channels = model.input_shape[-1]
         
        # Convert image from 3-channels to 1-channel (needed for TransUNet)   
        image = np.repeat(image, 3, axis=2) if n_channels == 3 else image
        
        # Predict
        predict = model.predict(np.expand_dims(image, axis=0))
        
        # Postprocessing
        mask = mask[:, :, 0]    # Remove the last dimension
        predict = (predict[0,:,:,0] >= 0.5).astype('uint8')     # Binarize with threshold
        
        return mask, predict

    # Compute IoU
    @staticmethod
    def compute_iou(confusion_matrix):
        # Compute IoU
        intersection = np.diag(confusion_matrix)
        predicted_set = confusion_matrix.sum(axis=0)
        ground_truth_set = confusion_matrix.sum(axis=1)
        union = ground_truth_set + predicted_set - intersection
        iou = intersection / union.astype(np.float32)
        return iou

    # Predict and save single image
    @classmethod
    def predict_and_save_image(cls, image_path: str, mask_path: str, model: Model):
        # Predict image 
        mask, prediction = cls.predict_image(image_path, mask_path, model)
        
        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                                            
        os.makedirs(Paths.PREDICTIONS, exist_ok=True)
        Image.fromarray((image).astype(np.uint8)).save(os.path.join(Paths.PREDICTIONS, "image.png"))
        Image.fromarray((mask*255.0).astype(np.uint8)).save(os.path.join(Paths.PREDICTIONS, "mask.png"))
        Image.fromarray((prediction*255.0).astype(np.uint8)).save(os.path.join(Paths.PREDICTIONS, "predict.png"))

        mask = mask.flatten()
        prediction = prediction.flatten()

        # Compute metrics
        conf_matrix = confusion_matrix(mask, prediction, labels=[0, 1])
        print(f"Confusion Matrix: {conf_matrix}")
        print(f"Accuracy: {accuracy_score(mask, prediction)}")
        print(f"Precision: {precision_score(mask, prediction, average = None)}")
        print(f"Recall: {recall_score(mask, prediction, average = None)}")
        print(f"F1 Score: {f1_score(mask, prediction, average = None)}")
        print(f"IoU: {cls.compute_iou(conf_matrix)}")
    
    @staticmethod
    def test_prediction(dataset: DatasetPaths = DatasetRegistry.PALSAR, denoised: bool = True, image_number: int = 123, 
                        model_path: str = "saves/palsar/best_models/UNetPP/model.hdf5"):
        # Test single image
        image_path = os.path.join(dataset.TEST_IMAGES_DENOISED_PATH if denoised else dataset.TEST_IMAGES_PATH, f'{image_number}.png')
        mask_path = os.path.join(dataset.TEST_LABELS_PATH, f'{image_number}.png') 

        # Load model
        model = ModelLoader.load_safe(os.path.join(os.getcwd(), model_path), full_path=True)

        EvaluationSession.predict_and_save_image(image_path, mask_path, model)


if __name__ == "__main__":
    EvaluationSession.test_prediction()
    