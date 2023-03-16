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
from nsd_access import NSDAccess
import torch.nn as nn
from pycocotools.coco import COCO
sys.path.append('src')
from utils import *
import wandb
import copy
from tqdm import tqdm
import nibabel as nib
from alexnet_encoder import AlexNetEncoder
from autoencoder import AutoEncoder
from pearson import PearsonCorrCoef

print(os.getcwd())
directory_path = '/export/raid1/home/ojeda040/Second-Sight/reconstructions/SCS 10:250:5 HS nsd_general AE'
#subdirs = [os.path.join(d, o) for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]
#length_subdirs = len(subdirs)


#print(length_subdirs)

alexnet_predictions = {}
brain_masks = {1:[1,2], 2: [3,4], 3:[5,6], 4:[7], 5:[1,2,3,4,5,6,7]}
image_counter = 0
images = []
device="cuda:0"

AN =  AlexNetEncoder()

AE = AutoEncoder(hashNum = "582",
                 lr=0.0000001,
                 vector="alexnet_encoder_sub1", #c_img_0, c_text_0, z_img_mixer
                 encoderHash="579",
                 log=False, 
                 batch_size=750,
                 device=device
                )
mask_path = "/export/raid1/home/ojeda040/Second-Sight/masks/"
masks = {0:torch.full((11838,), False),
            1:torch.load(mask_path + "V1.pt"),
            2:torch.load(mask_path + "V2.pt"),
            3:torch.load(mask_path + "V3.pt"),
            4:torch.load(mask_path + "V4.pt"),
            5:torch.load(mask_path + "V5.pt"),
            6:torch.load(mask_path + "V6.pt"),
            7:torch.load(mask_path + "V7.pt")}

_, _, x_param, x_test, _, _, _, _, param_trials, test_trials = load_nsd(vector="c_img_0", loader=False, average=False, nest=True)
#x_test_ae = torch.zeros((x_test.shape[0], 11838))
x_test_ae = torch.zeros((50, 11838))
# for i in tqdm(range(x_test.shape[0]), desc="Autoencoding samples and averaging"):
for i in tqdm(range(50), desc="Autoencoding samples and averaging"):
    x_test_ae[i] = torch.mean(AE.predict(x_test[i]),dim=0)
beta = x_test_ae

beta_mask = masks[0]
for i in brain_masks[5]:
    beta_mask = torch.logical_or(beta_mask, masks[i])
    
beta_mask = ~beta_mask
# print(type(beta_mask))
# print(np.unique(beta_mask, return_counts=True))
# print(np.unique(~beta_mask, return_counts=True))
    

for i in range(25):
    path = directory_path + "/" + str(i)
    for filename in os.listdir(path): 
        with open(os.path.join(path, filename), 'r') as f:
            if('iter' in filename):
                image_pil = Image.open(path + '/' + filename)
                images.append(image_pil)
                image_counter += 1
                if(image_counter == 10):
                    alexnet_predictions[i] = AN.predict(images, brain_masks[5])
                    image_counter = 0
                    images = []
 
beta_i = beta
for i in range(25):
    
    beta_primes = alexnet_predictions[i].moveaxis(0, 1).to(device)
    
    beta = beta_i[i][beta_mask]
                
    xDup = beta.repeat(beta_primes.shape[1], 1).moveaxis(0, 1).to(device)
    PeC = PearsonCorrCoef(num_outputs=beta_primes.shape[1]).to(device) 
    print(xDup.shape, beta_primes.shape)
    scores = PeC(xDup, beta_primes)
    scores_np = scores.detach().cpu().numpy()
    
    np.save("/export/raid1/home/kneel027/Second-Sight/logs/SCS 10:250:5 HS nsd_general AE/" + str(i) + "_score_list_higher_visual.npy", scores_np)
