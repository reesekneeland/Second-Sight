import os
import sys
import torch
from torchmetrics.functional import pearson_corrcoef
from torch.autograd import Variable
import numpy as np
from nsd_access import NSDAccess
import glob
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch.nn as nn
from pycocotools.coco import COCO
sys.path.append('src')
from utils import *
import wandb
import copy
from tqdm import tqdm
import nibabel as nib
from alexnet_encoder import AlexNetEncoder


print(os.getcwd())
directory_path = '/export/raid1/home/ojeda040/Second-Sight/reconstructions/SCS 10:250:5 HS nsd_general AE'
#subdirs = [os.path.join(d, o) for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]
#length_subdirs = len(subdirs)


#print(length_subdirs)

alexnet_predictions = {}
masks = {1:[1,2], 2: [1,2,3,4], 3:[1,2,3,4,5,6,7]}
image_counter = 0
images = []

AN =  AlexNetEncoder()


for i in range(25):
    path = directory_path + "/" + str(i)
    for filename in os.listdir(path): 
        with open(os.path.join(path, filename), 'r') as f:
            if('iter' in filename):
                image_pil = Image.open(path + '/' + filename)
                images.append(image_pil)
                image_counter += 1
                if(image_counter == 10):
                    alexnet_predictions[i] = AN.predict(images, masks[1])
                    image_counter = 0
                    images = []
            
    break
                
                
print(alexnet_predictions[0].shape)
