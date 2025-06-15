import glob
import cv2
import numpy as np
from utils import load_image,load_mask
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
#from transunet import TransUNet
from models import UNet, UNetL, UNetPP, UNetPPL
from config import DatasetRegistry
import os 
from tensorflow.keras.models import load_model
from matplotlib.pyplot import imshow,show

m=load_model(f"{os.getcwd()}/checkpoints/")


image_path = os.path.join(DatasetRegistry.PALSAR.TEST_IMAGES_PATH, '0.png')
mask_path = os.path.join(DatasetRegistry.PALSAR.TEST_LABELS_PATH, '0.png')          
           
def Predict_image(img_Path, mask_Path, m):
    image= load_image(img_Path)
    mask = load_mask(mask_Path)
    predict = m.predict(np.expand_dims(image, axis=0))
    print(predict.shape)
    predict = (predict[0,:,:,0] >= 0.5).astype('uint8')
    return image, mask, predict


result = Predict_image (image_path, mask_path,m)

for img in result:
    imshow(img)
    show()

 
##########################################################################

def compute_iou(y_true, y_pred):
     # ytrue, ypred is a flatten vector
     current = confusion_matrix(y_true, y_pred, labels=[0, 1])
     print(current)
     # compute mean iou
     intersection = np.diag(current)
     ground_truth_set = current.sum(axis=1)
     predicted_set = current.sum(axis=0)
     union = ground_truth_set + predicted_set - intersection
     IoU = intersection / union.astype(np.float32)
     tmp=np.mean(IoU)
     #print(tmp)
     return tmp


# Compute all
for mi in output_models:
    try:
        m = DeeplabV3Plus.DeeplabV3Plus(n_classes, input_height=input_height, input_width=input_width)
        #m = UNet.UNet(n_classes, input_height=input_height, input_width=input_width)
        #m = TransUNet(image_size=256, grid=(16,16), num_classes=2, pretrain=True)
        m.load_weights(mi)
        seg_list,pr_list=np.array([]).astype('uint8'),np.array([]).astype('uint8')
        for i, (image_path, mask_path) in  :  
            _, mask, predict = Predict_image (image_path, mask_path,m) 
            seg_list=np.hstack((seg_list,mask.flatten().astype('uint8')))
            pr_list=np.hstack((pr_list,predict.flatten().astype('uint8')))

       # f = open ("result_"+output_name+".txt", "a")
        print(mi+"\t"+str(accuracy_score(seg_list,pr_list))+"\t"+
              str(precision_score(seg_list,pr_list))+"\t"+
              str(recall_score(seg_list,pr_list))+"\t"+
              str(f1_score(seg_list,pr_list))+"\t"+
              str(compute_iou(seg_list,pr_list)),file=f
              )    
        print(mi+"\t"+str(accuracy_score(seg_list,pr_list))+"\t"+
              str(precision_score(seg_list,pr_list,labels =[0,1],pos_label=1))+"\t"+
              str(recall_score(seg_list,pr_list,labels =[0,1],pos_label=1))+"\t"+
              str(f1_score(seg_list,pr_list,labels =[0,1],pos_label=1))+"\t"+
              str(compute_iou(seg_list,pr_list))
              ) 
        print(mi+"\t"+str(accuracy_score(seg_list,pr_list))+"\t"+
              str(precision_score(seg_list,pr_list,labels =[0,1],pos_label=0))+"\t"+
              str(recall_score(seg_list,pr_list,labels =[0,1],pos_label=0))+"\t"+
              str(f1_score(seg_list,pr_list,labels =[0,1],pos_label=0))+"\t"+
              str(compute_iou(seg_list,pr_list))
              )  
       # f.close()
    except Exception as r:
        f = open ("result_"+output_name+".txt", "a")
        print(mi,r,file=f)
        f.close()


        
