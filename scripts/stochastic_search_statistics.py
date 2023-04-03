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
import random
import transformers
from transformers import CLIPTokenizerFast, AutoProcessor, CLIPModel, CLIPVisionModelWithProjection
import math
import re

class Stochastic_Search_Statistics():
    
    def __init__(self):

        self.directory_path = '/export/raid1/home/ojeda040/Second-Sight/reconstructions/SCS 10:250:5 HS nsd_general AE'
        #subdirs = [os.path.join(d, o) for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]
        #length_subdirs = len(subdirs)
        # self.R = Reconstructor(device=self.device)
        self.device="cuda:3"
        model_id = "openai/clip-vit-large-patch14"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.visionmodel = CLIPVisionModelWithProjection.from_pretrained(model_id).to(self.device)
        self.PeC = PearsonCorrCoef().to(self.device)
        self.brain_masks =  {1:[1,2], 2: [3,4], 3:[5,6], 4:[7], 5:[1,2,3,4,5,6,7]}
        self.mask_path = "/export/raid1/home/ojeda040/Second-Sight/masks/"
        self.masks = {0:torch.full((11838,), False),
                    1:torch.load(self.mask_path + "V1.pt"),
                    2:torch.load(self.mask_path + "V2.pt"),
                    3:torch.load(self.mask_path + "V3.pt"),
                    4:torch.load(self.mask_path + "V4.pt"),
                    5:torch.load(self.mask_path + "V5.pt"),
                    6:torch.load(self.mask_path + "V6.pt"),
                    7:torch.load(self.mask_path + "V7.pt")}

    def autoencoded_brain_samples(self):
        
        AE = AutoEncoder(hashNum = "582",
                        lr=0.0000001,
                        vector="alexnet_encoder_sub1", #c_img_0, c_text_0, z_img_mixer
                        encoderHash="579",
                        log=False, 
                        batch_size=750,
                        device="cuda:0"
                        )
        
        # Load the test samples
        _, _, x_param, x_test, _, _, _, y_test, param_trials, test_trials = load_nsd(vector="images", loader=False, average=False, nest=True)
        #print(y_test[0].reshape((425,425,3)).numpy().shape)
        # test = Image.fromarray(y_test[0].reshape((425,425,3)).numpy().astype(np.uint8))
        # test.save("/home/naxos2-raid25/ojeda040/local/ojeda040/Second-Sight/logs/test.png")
        
        
        #x_test_ae = torch.zeros((x_test.shape[0], 11838))
        x_test_ae = torch.zeros((50, 11838))
        
        # 
        # for i in tqdm(range(x_test.shape[0]), desc="Autoencoding samples and averaging"):
        for i in tqdm(range(50), desc = "Autoencoding samples and averaging" ):
            x_test_ae[i] = torch.mean(AE.predict(x_test[i]),dim=0)
        
        return x_test_ae
    
    def return_all_masks(self):
        
        # Instantiate all the mask variales
        brain_mask_V1 = self.masks[0]
        brain_mask_V2 = self.masks[0]
        brain_mask_V3 = self.masks[0]
        brain_mask_V4 = self.masks[0]
        brain_mask_early_visual = self.masks[0]
        brain_mask_higher_visual = self.masks[0]
        
        # Fill V1 mask
        for i in self.brain_masks[1]:
            brain_mask_V1 = torch.logical_or(brain_mask_V1, self.masks[i])
        
        # Fill V2 mask
        for i in self.brain_masks[2]:
            brain_mask_V2 = torch.logical_or(brain_mask_V2, self.masks[i])
        
        # Fill V3 Mask
        for i in self.brain_masks[3]:
            brain_mask_V3 = torch.logical_or(brain_mask_V3, self.masks[i])
        
        # Fill V4 Mask
        for i in self.brain_masks[4]:
            brain_mask_V4 = torch.logical_or(brain_mask_V4, self.masks[i])
            
        # Fill early visual mask
        for i in self.brain_masks[5]:
            brain_mask_early_visual = torch.logical_or(brain_mask_early_visual, self.masks[i])
           
        # Negate the early visual cortex to get the higher visual cortex region.  
        brain_mask_higher_visual = ~brain_mask_early_visual
        
        return brain_mask_V1, brain_mask_V2, brain_mask_V3, brain_mask_V4, brain_mask_early_visual, brain_mask_higher_visual

    def generate_brain_predictions(self):
        
        alexnet_predictions = {}
        image_counter = 0
        images = []
        device = "cuda:0"

        AN =  AlexNetEncoder()

        # Autoencoded avearged brain samples 
        beta = self.autoencoded_brain_samples()
        
        # Grab the necessary brain masks
        brain_mask_V1, _, _, _, _, _ = self.return_all_masks()
            

        for i in range(1):
            path = self.directory_path + "/" + str(i)
            for filename in os.listdir(path): 
                with open(os.path.join(path, filename), 'r') as f:
                    if('iter' in filename):
                        image_pil = Image.open(path + '/' + filename)
                        images.append(image_pil)
                        image_counter += 1
                        if(image_counter == 10):
                            alexnet_predictions[i] = AN.predict(images, brain_mask_V1)
                            image_counter = 0
                            images = []
        
        beta_i = beta
        for i in range(1):
            
            print(alexnet_predictions[i].shape)
            beta_primes = alexnet_predictions[i].moveaxis(0, 1).to(device)
            print(beta_primes.shape)
            
            beta = beta_i[i][brain_mask_V1]
                        
            xDup = beta.repeat(beta_primes.shape[1], 1).moveaxis(0, 1).to(device)
            PeC = PearsonCorrCoef(num_outputs=beta_primes.shape[1]).to(device) 
            print(xDup.shape, beta_primes.shape)
            scores = PeC(xDup, beta_primes)
            scores_np = scores.detach().cpu().numpy()
            print(scores_np)
            
            #np.save("/export/raid1/home/kneel027/Second-Sight/logs/SCS 10:250:5 HS nsd_general AE/" + str(i) + "_score_list_higher_visual.npy", scores_np)
            
    def generate_pearson_correlation(self, alexnet_predictions, beta_sample, brain_mask, unmasked = True):
        
        # print(alexnet_prediction.shape)
        # # For a single prediction reshape the prediction to work. 
        # if(len(alexnet_prediction.shape) < 2): 
        #     alexnet_prediction = alexnet_prediction[None,:]
            
        # # Mask the beta scan. 
        # if(not unmasked):
        #     beta = beta_sample[brain_mask]
        # else:
        #     beta = beta_sample
        
        # # Move the axis of the numpy array. 
        # beta_primes = alexnet_prediction.moveaxis(0, 1).to(self.device)
        
        # # Calculate the pearson correlation   
        # xDup = beta.repeat(beta_primes.shape[1], 1).moveaxis(0, 1).to(self.device)
        # PeC = PearsonCorrCoef(num_outputs=beta_primes.shape[1]).to(self.device) 
        # print(xDup.shape, beta_primes.shape)
        # if(xDup.shape[1] == 1):
        #     scores = PeC(xDup[:,0], beta_primes[:,0])
        # else:
        #     scores = PeC(xDup, beta_primes)
        # scores_np = scores.detach().cpu().numpy()
        # print(scores_np)
        
            
        # print(alexnet_predictions[i].shape)
        beta_primes = alexnet_predictions.moveaxis(0, 1).to(self.device)
        
        if(not unmasked):
            beta = beta_sample[brain_mask]
        else:
            beta = beta_sample
                    
        xDup = beta.repeat(beta_primes.shape[1], 1).moveaxis(0, 1).to(self.device)
        PeC = PearsonCorrCoef(num_outputs=beta_primes.shape[1]).to(self.device) 
        print(xDup.shape, beta_primes.shape)
        scores = PeC(xDup, beta_primes)
        scores_np = scores.detach().cpu().numpy()
        
        return scores_np
        
    def calculate_ssim(self, ground_truth_path, reconstruction_path):

        ground_truth   = cv2.imread(ground_truth_path)
        reconstruction = cv2.imread(reconstruction_path)
        
        ground_truth = cv2.resize(ground_truth, (425, 425))
        reconstruction = cv2.resize(reconstruction, (425, 425))

        ground_truth = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2GRAY)
        reconstruction = cv2.cvtColor(reconstruction, cv2.COLOR_BGR2GRAY)

        return ssim_scs(ground_truth, reconstruction)
            
        
    def calculate_pixel_correlation(self, ground_truth, reconstruction):
        
        return pixel_correlation(ground_truth, reconstruction)
        
        
        
    #two_way_prob is the two way identification experiment between the given image and a random search reconstruction of a different sample with respect to the ground truth
    #clip_pearson is the pearson correlation score between the clips of the two given images
    def calculate_clip_similarity(self, experiment_name, sample):
        with torch.no_grad():
            exp_path = "/export/raid1/home/kneel027/Second-Sight/reconstructions/" + experiment_name + "/"
            folders = sorted([int(f.name) for f in os.scandir(exp_path) if f.is_dir()])
            rand_list = [i for i in range(len(folders)) if folders[i] != sample and os.listdir(exp_path + str(folders[i]) + "/")]
            rand_index = random.choice(rand_list)
            random_image = Image.open(exp_path + str(folders[rand_index]) + "/Search Reconstruction.png")
            image = Image.open(exp_path + str(sample) + "/Search Reconstruction.png")
            ground_truth = Image.open(exp_path + str(sample) + "/Ground Truth.png")
            
            inputs = self.processor(images=[ground_truth, image, random_image], return_tensors="pt", padding=True).to(self.device)
            outputs = self.visionmodel(**inputs)
            
            gt_feature = outputs.image_embeds[0].reshape((768))
            reconstruct_feature = outputs.image_embeds[1].reshape((768))
            rand_image_feature = outputs.image_embeds[2].reshape((768))
            rand_image_feature /= rand_image_feature.norm(dim=-1, keepdim=True)
            gt_feature /= gt_feature.norm(dim=-1, keepdim=True)
            reconstruct_feature /= reconstruct_feature.norm(dim=-1, keepdim=True)
            
            loss = (torch.stack([gt_feature @ reconstruct_feature, gt_feature @ rand_image_feature]) *100)
            two_way_prob = loss.softmax(dim=0)[0]
            clip_pearson = self.PeC(gt_feature.flatten(), reconstruct_feature.flatten())
        return float(two_way_prob), float(clip_pearson)
    
    
    def image_indices(self, folder):
        
        # Directory path
        dir_path = "/export/raid1/home/ojeda040/Second-Sight/reconstructions/" + folder + "/"
        
        # Grab the list of files
        files = []
        for path in os.listdir(dir_path):
            
            # check if current path is a file
            if os.path.isfile(os.path.join(dir_path, path)):
                files.append(path)
        
        # Get just the image number and then sort the list. 
        indicies = []
        for i in range(len(files)):
            indicies.append(int(re.search(r'\d+', files[i]).group()))
       
        indicies.sort()
        
        return indicies
        
        
        
    def create_dataframe(self, folder):
        
        # Path to the folder
        log_path       = "/export/raid1/home/ojeda040/Second-Sight/logs/" + folder + "/"
        directory_path = "/export/raid1/home/ojeda040/Second-Sight/reconstructions/" + folder + "/"
        
        
        # List of image numbers created. 
        #idx = self.image_indices(folder)
        idx = [0, 1, 2, 3, 4, 25, 26, 27, 28, 29]
         
        # Instantiate the alexnet class for predicts
        AN =  AlexNetEncoder()
        
        # Autoencoded avearged brain samples 
        beta_samples = self.autoencoded_brain_samples()
        
        # Grab the necessary brain masks
        brain_mask_V1, brain_mask_V2, brain_mask_V3, brain_mask_V4, brain_mask_early_visual, brain_mask_higher_visual = self.return_all_masks()
        
        # create an Empty DataFrame
        # object With column names only
        # Sample Indicator: 
        #   0 --> Ground Truth
        #   1 --> Ground Truth CLIP
        #   2 --> Decoded CLIP Only
        #   3 --> Search Reconstruction
        df = pd.DataFrame(columns = ['ID', 'Iter', 'Sample Indicator', 'Strength', 'Brain Correlation V1', 'Brain Correlation V2', 
                                     'Brain Correlation V3', 'Brain Correlation V4', 'Brain Correlation Early Visual', 'Brain Correlation Higher Visual',
                                     'Brain Correlation Unmasked', 'SSIM', 'Pixel Correlation', 'CLIP Pearson', 'CLIP Two-way'])
        
        # Seach iteration count. 
        iter_count = 0
        
        # Dataframe index count. 
        df_row_num = 0
        
        # Set of images in a folder that need to be AlexNet predicted. 
        folder_image_set = []
        
        # Append rows to an empty DataFrame
        for i in tqdm(idx, desc="creating dataframe rows"):
            
            # Create the path
            path = directory_path + str(i)
            
            for filename in os.listdir(path): 
                
                # Ground Truth Image
                ground_truth_path = path + '/' + 'Ground Truth.png'
                ground_truth = Image.open(ground_truth_path)
                
                # Ground Truth CLIP Image
                ground_truth_CLIP_path = path + '/' + 'Ground Truth CLIP.png'
                ground_truth_CLIP = Image.open(ground_truth_CLIP_path)
                
                # Decoded CLIP Only Image
                decoded_CLIP_only_path = path + '/' + 'Decoded CLIP Only.png'
                decoded_CLIP_only = Image.open(decoded_CLIP_only_path)
                
                # Search Reconstruction Image
                search_reconstruction_path = path + '/' + 'Search Reconstruction.png'
                
                # CLIP metrics calculation
                two_way_prob, clip_pearson = self.calculate_clip_similarity(folder, i)
                
                with open(os.path.join(path, filename), 'r') as f:
                    if('iter' in filename):
                        
                        # Reconstruction path
                        reconstruction_path = path + '/' + filename
                        
                        # Iter reconstruction image
                        reconstruction = Image.open(reconstruction_path)
                        folder_image_set.append(reconstruction)
                        
                        # Pix Corr metrics calculation
                        pix_corr = self.calculate_pixel_correlation(ground_truth, reconstruction)
                        
                        # SSIM metrics calculation
                        ssim_ground_truth          = self.calculate_ssim(ground_truth_path, reconstruction_path)
                        ssim_search_reconstruction = self.calculate_ssim(search_reconstruction_path, reconstruction_path)
                        
                        # Calculate the strength at that reconstruction iter image. 
                        strength = 1.0-0.6*(math.pow(iter_count/10, 3))
                
                        # Make data frame row
                        if(ssim_search_reconstruction == 1.00):
                            row = pd.DataFrame({'ID' : str(i), 'Iter' : str(iter_count), 'Sample Indicator' : "3", 'Strength' : str(round(strength, 10)),
                                                'SSIM' : str(round(ssim_ground_truth, 10)), 'Pixel Correlation' : str(round(pix_corr, 10)),
                                                'CLIP Pearson' : str(round(clip_pearson, 10)), 'CLIP Two-way' : str(round(two_way_prob, 10))}, index=[df_row_num])
                        
                        else:
                            row = pd.DataFrame({'ID' : str(i), 'Iter' : str(iter_count), 'Strength' : str(round(strength, 10)), 
                                                'SSIM' : str(round(ssim_ground_truth, 10)), 'Pixel Correlation' : str(round(pix_corr, 10)), 
                                                'CLIP Pearson' : str(round(clip_pearson, 10))},  index=[df_row_num])
                        
                        # Add the row to the dataframe
                        df = pd.concat([df, row])
                        
                        # Iterate the counts
                        iter_count += 1
                        df_row_num += 1
                        

            # Add the images for alexnet predictions
            folder_image_set.append(decoded_CLIP_only)
            folder_image_set.append(ground_truth_CLIP)
            folder_image_set.append(ground_truth)
                        
            # Alexnet predictions for each region
            alexnet_prediction_V1               = AN.predict(folder_image_set, brain_mask_V1, False)
            alexnet_prediction_V2               = AN.predict(folder_image_set, brain_mask_V2, False)
            alexnet_prediction_V3               = AN.predict(folder_image_set, brain_mask_V3, False)
            alexnet_prediction_V4               = AN.predict(folder_image_set, brain_mask_V4, False)
            alexnet_prediction_early_visual     = AN.predict(folder_image_set, brain_mask_early_visual, False)
            alexnet_prediction_higher_visual    = AN.predict(folder_image_set, brain_mask_higher_visual, False)
            alexnet_prediction_unmasked         = AN.predict(folder_image_set, brain_mask_higher_visual, True)
            
            # Pearson correlations for each reconstruction region
            pearson_correlation_V1              = self.generate_pearson_correlation(alexnet_prediction_V1, beta_samples[i], brain_mask_V1, unmasked=False)
            pearson_correlation_V2              = self.generate_pearson_correlation(alexnet_prediction_V2, beta_samples[i], brain_mask_V2, unmasked=False)
            pearson_correlation_V3              = self.generate_pearson_correlation(alexnet_prediction_V3, beta_samples[i], brain_mask_V3, unmasked=False)
            pearson_correlation_V4              = self.generate_pearson_correlation(alexnet_prediction_V4, beta_samples[i], brain_mask_V4, unmasked=False)
            pearson_correlation_early_visual    = self.generate_pearson_correlation(alexnet_prediction_early_visual, beta_samples[i], brain_mask_early_visual, unmasked=False)
            pearson_correlation_higher_visual   = self.generate_pearson_correlation(alexnet_prediction_higher_visual, beta_samples[i], brain_mask_higher_visual, unmasked=False)
            pearson_correlation_unmasked        = self.generate_pearson_correlation(alexnet_prediction_unmasked, beta_samples[i], brain_mask_higher_visual, unmasked=True)
                        
            # Make data frame row for decoded clip only
            pix_corr_decoded = self.calculate_pixel_correlation(ground_truth, decoded_CLIP_only)
            ssim_decoded = self.calculate_ssim(ground_truth_path, decoded_CLIP_only_path)
            row_decoded = pd.DataFrame({'ID' : str(i), 'Sample Indicator' : "2", 'SSIM' : str(round(ssim_decoded, 10)), 'Pixel Correlation' : str(round(pix_corr_decoded, 10))}, index=[df_row_num])
            df_row_num += 1
            df = pd.concat([df, row_decoded])
            
            # Make data frame row for ground truth CLIP
            pix_corr_decoded = self.calculate_pixel_correlation(ground_truth, ground_truth_CLIP)
            ssim_decoded = self.calculate_ssim(ground_truth_path, ground_truth_CLIP_path)
            row_ground_truth_CLIP = pd.DataFrame({'ID' : str(i), 'Sample Indicator' : "1", 'SSIM' : str(round(ssim_decoded, 10)), 'Pixel Correlation' : str(round(pix_corr_decoded, 10))}, index=[df_row_num])
            df_row_num += 1
            df = pd.concat([df, row_ground_truth_CLIP])
            
            # Make data frame row for ground truth Image
            row_ground_truth = pd.DataFrame({'ID' : str(i), 'Sample Indicator' : "0", 'Strength' : str(round(strength, 10))}, index=[df_row_num])
            df_row_num += 1
            df = pd.concat([df, row_ground_truth])
            
            for image_index in range(len(folder_image_set)): 
                df.at[((df_row_num - len(folder_image_set)) + image_index), 'Brain Correlation V1']             =  pearson_correlation_V1[image_index] 
                df.at[((df_row_num - len(folder_image_set)) + image_index), 'Brain Correlation V2']             =  pearson_correlation_V2[image_index]      
                df.at[((df_row_num - len(folder_image_set)) + image_index), 'Brain Correlation V3']             =  pearson_correlation_V3[image_index]
                df.at[((df_row_num - len(folder_image_set)) + image_index), 'Brain Correlation V4']             =  pearson_correlation_V4[image_index]
                df.at[((df_row_num - len(folder_image_set)) + image_index), 'Brain Correlation Early Visual']   =  pearson_correlation_early_visual[image_index]
                df.at[((df_row_num - len(folder_image_set)) + image_index), 'Brain Correlation Higher Visual']  =  pearson_correlation_higher_visual[image_index]  
                df.at[((df_row_num - len(folder_image_set)) + image_index), 'Brain Correlation Unmasked']       =  pearson_correlation_unmasked[image_index]  
                
            # Reset the iter_count and folder_image for the next folder. 
            iter_count = 0
            folder_image_set = []    
                                           
                        
        print(df.shape)
        print(df)
        df.to_csv(log_path + "statistics_df_10.csv")
    
    
def main():
    
    SCS = Stochastic_Search_Statistics()
    
    #SCS.generate_brain_predictions() 
    #SCS.calculate_ssim()    
    #SCS.calculate_pixel_correlation()
    
    SCS.create_dataframe("SCS VD PCA LR 10:250:5 0.4 Exp AE")
    #SCS.create_dataframe("SCS VD PCA LR 10:250:5 0.3 Exp2 AE")
    #SCS.create_dataframe("SCS VD PCA LR 10:250:5 0.6 Exp3 AE")
    
    # gt = Image.open("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/reconstructions/SCS VD PCA LR 10:100:4 0.4 Exponential Strength AE/1/Ground Truth.png")
    # garbo = Image.open("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/reconstructions/SCS VD PCA 10:100:4 HS nsd_general AE/0/Search Reconstruction.png")
    # reconstruct = Image.open("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/reconstructions/SCS VD PCA LR 10:100:4 0.4 Exponential Strength AE/1/Search Reconstruction.png")
    # surfer = Image.open("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/surfer.png")
    # print(SCS.calculate_clip_similarity("SCS VD PCA LR 10:250:5 0.4 Exp AE", 10))
    # print(SCS.calculate_clip_similarity("SCS VD PCA LR 10:250:5 0.4 Exp AE", 3))
    # print(SCS.calculate_clip_similarity("SCS VD PCA LR 10:250:5 0.4 Exp AE", 26))
        
if __name__ == "__main__":
    main()