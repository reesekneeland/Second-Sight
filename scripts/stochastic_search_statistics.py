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
from reconstructor import Reconstructor
from pearson import PearsonCorrCoef
import cv2
<<<<<<< HEAD
import math

=======
from random import randrange
import transformers
from transformers import CLIPTokenizerFast, AutoProcessor, CLIPModel, CLIPVisionModelWithProjection
>>>>>>> f7f280056404f12cdd50bcb87780c2f504760c5d

class Stochastic_Search_Statistics():
    
    def __init__(self):

        self.directory_path = '/export/raid1/home/ojeda040/Second-Sight/reconstructions/SCS 10:250:5 HS nsd_general AE'
        #subdirs = [os.path.join(d, o) for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]
        #length_subdirs = len(subdirs)
        # self.R = Reconstructor(device=self.device)
        self.device="cuda:1"
        model_id = "openai/clip-vit-large-patch14"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.visionmodel = CLIPVisionModelWithProjection.from_pretrained(model_id).to(self.device)


    def generate_brain_predictions(self):
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
            path = self.directory_path + "/" + str(i)
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
        
    def calculate_ssim(self):
        count = 0

        for i in range(25):
            ground_truth   = cv2.imread('/export/raid1/home/ojeda040/Second-Sight/reconstructions/SCS VD 10:250:5 HS nsd_general AE/' + str(i) + '/Ground Truth.png')
            reconstruction = cv2.imread('/export/raid1/home/ojeda040/Second-Sight/reconstructions/SCS VD 10:250:5 HS nsd_general AE/' + str(i) + '/Search Reconstruction.png')
            
            ground_truth = cv2.resize(ground_truth, (425, 425))
            reconstruction = cv2.resize(reconstruction, (425, 425))

            ground_truth = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2GRAY)
            reconstruction = cv2.cvtColor(reconstruction, cv2.COLOR_BGR2GRAY)

            count += ssim_scs(ground_truth, reconstruction)
            
        print(count / 25)
        
    def calculate_pixel_correlation(self, ground_truth, reconstruction):
        
        count = 0

        for i in range(25):
            ground_truth   = Image.open('/export/raid1/home/ojeda040/Second-Sight/reconstructions/SCS VD 10:250:5 HS nsd_general AE/' + str(i) + '/Ground Truth.png')
            reconstruction = Image.open('/export/raid1/home/ojeda040/Second-Sight/reconstructions/SCS VD 10:250:5 HS nsd_general AE/' + str(i) + '/Search Reconstruction.png')
            
            count += pixel_correlation(ground_truth, reconstruction)
            
        print(count / 25)
        
        
    #two_way_prob is the two way identification experiment between the given image and a random test sample with respect to the ground truth
    #clip_pearson is the pearson correlation score between the clips of the two given images
    def calculate_clip_similarity(self, ground_truth, image):
        random_image = Image.open("/export/raid1/home/kneel027/Second-Sight/logs/shared1000_images/" + str(randrange(0,999)) + ".png")
        PeC = PearsonCorrCoef().to(self.device)

        inputs = self.processor(images=[ground_truth, image, random_image], return_tensors="pt", padding=True).to(self.device)
        outputs = self.visionmodel(**inputs)
        
        gt_feature = outputs.image_embeds[0].reshape((768))
        reconstruct_feature = outputs.image_embeds[1].reshape((768))
        rand_image_feature = outputs.image_embeds[2].reshape((768))
        rand_image_feature /= rand_image_feature.norm(dim=-1, keepdim=True)
        gt_feature /= gt_feature.norm(dim=-1, keepdim=True)
        reconstruct_feature /= reconstruct_feature.norm(dim=-1, keepdim=True)
        
        loss = torch.stack([gt_feature @ reconstruct_feature, gt_feature @ rand_image_feature]) * 100
        two_way_prob = loss.softmax(dim=0)[1]
        clip_pearson = PeC(gt_feature.flatten(), reconstruct_feature.flatten())
        return two_way_prob, clip_pearson
        
        
        
    def create_dataframe(self):
        
        # create an Empty DataFrame
        # object With column names only
        df_V1 = pd.DataFrame(columns = ['ROI', 'ID', 'Iter', 'Strength', 'Brain Correlation', 'SSIM', 'Pixel Correlation', 'CLIP Pearson', 'CLIP Two-way'])
        df_V2 = pd.DataFrame(columns = ['ROI', 'ID', 'Iter', 'Strength', 'Brain Correlation', 'SSIM', 'Pixel Correlation', 'CLIP Pearson', 'CLIP Two-way'])
        
        # append rows to an empty DataFrame
        for i in range(25):
            for j in range(10):
                
                strength = 1.0-0.4*(math.pow(j/10, 3))
                
                row = pd.DataFrame({'ROI' : 'V1', 'ID' : str(i), 'Iter' : str(j), 'Strength' : str(strength), 'Brain Correlation' : str(brain_correlation_V1[i][j]),
                                'SSIM' : '1', 'Pixel Correlation' : '1', 'CLIP' : '1' }, index=[i])
                
                row2 = pd.DataFrame({'ROI' : 'V1', 'ID' : str(i), 'Iter' : '0', 'Strength' : '1.0', 'Brain Correlation' : '2200',
                                'SSIM' : '1', 'Pixel Correlation' : '1', 'CLIP' : '1' }, index=[i + 25])
                
                df_V1 = pd.concat([df_V1, row])
                df_V2 = pd.concat([df_V2, row2])
        
        df = pd.concat([df_V1, df_V2])
        print(df.shape)
        print(df)
    
    
def main():
    
    SCS = Stochastic_Search_Statistics()
    
    #SCS.generate_brain_predictions() 
    #SCS.calculate_ssim()    
    #SCS.calculate_pixel_correlation()
    # SCS.create_dataframe()
    
    gt = Image.open("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/reconstructions/SCS VD PCA LR 10:100:4 0.4 Exponential Strength AE/3/Ground Truth.png")
    im1 = Image.open("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/reconstructions/SCS VD PCA 10:100:4 HS nsd_general AE/0/Search Reconstruction.png")
    surfer = Image.open("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/surfer.png")
    SCS.calculate_clip_similarity(gt, im1)
    SCS.calculate_clip_similarity(gt, surfer)
        
if __name__ == "__main__":
    main()