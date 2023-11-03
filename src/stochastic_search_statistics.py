import os
import sys
import torch
import pandas as pd
from PIL import Image
sys.path.append('src')
from utils import *
from tqdm import tqdm
from autoencoder import AutoEncoder
from torchmetrics import PearsonCorrCoef
from stochastic_search import StochasticSearch
import cv2
import random
from transformers import AutoProcessor, CLIPVisionModelWithProjection
from gnet8_encoder import GNet8_Encoder
import math
import json
import re
import numpy as np
from skimage import io
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim
import os.path
import torchvision.models as tvmodels
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.transforms as T
import scipy as sp
from scipy.stats import pearsonr,binom,linregress
import clip

# CNN Metrics
def initialize_net_metrics(device):
    print("initalizing net metrics")
    # print(len(images))
    net_models = {}
    global feat_list
    feat_list = []
    def fn(module, inputs, outputs):
        feat_list.append(outputs.cpu().numpy())

    net_list = [
        ('Inception V3','avgpool'),
        ('CLIP Two-way','final'),
        ('AlexNet 2',2),
        ('AlexNet 5',5),
        ('AlexNet 7',7),
        ('EffNet-B','avgpool'),
        ('SwAV','avgpool')
        ]

    for (net_name,layer) in net_list:
        print(net_name)
        if net_name == 'Inception V3': # SD Brain uses this
            net_models[f'{net_name}{layer}'] = tvmodels.inception_v3(pretrained=True)
            if layer == 'avgpool':
                net_models[f'{net_name}{layer}'].avgpool.register_forward_hook(fn) 
            elif layer == 'lastconv':
                net_models[f'{net_name}{layer}'].Mixed_7c.register_forward_hook(fn)
                
        elif 'AlexNet' in net_name:
            net_models[f'{net_name}{layer}'] = tvmodels.alexnet(pretrained=True)
            if layer==2:
                net_models[f'{net_name}{layer}'].features[4].register_forward_hook(fn)
            elif layer==5:
                net_models[f'{net_name}{layer}'].features[11].register_forward_hook(fn)
            elif layer==7:
                net_models[f'{net_name}{layer}'].classifier[5].register_forward_hook(fn)
                
        elif net_name == 'CLIP Two-way':
            model, _ = clip.load("ViT-L/14", device=device)
            net_models[f'{net_name}{layer}'] = model.visual
            net_models[f'{net_name}{layer}'] = net_models[f'{net_name}{layer}'].to(torch.float32)
            if layer==7:
                net_models[f'{net_name}{layer}'].transformer.resblocks[7].register_forward_hook(fn)
            elif layer==12:
                net_models[f'{net_name}{layer}'].transformer.resblocks[12].register_forward_hook(fn)
            elif layer=='final':
                net_models[f'{net_name}{layer}'].register_forward_hook(fn)
        
        elif net_name == 'EffNet-B':
            net_models[f'{net_name}{layer}'] = tvmodels.efficientnet_b1(weights=True)
            net_models[f'{net_name}{layer}'].avgpool.register_forward_hook(fn) 
            
        elif net_name == 'SwAV':
            net_models[f'{net_name}{layer}'] = torch.hub.load('facebookresearch/swav:main', 'resnet50')
            net_models[f'{net_name}{layer}'].avgpool.register_forward_hook(fn) 
        net_models[f'{net_name}{layer}'].eval()
        net_models[f'{net_name}{layer}'].to(device)  
    return net_models

class batch_generator_external_images(Dataset):

    def __init__(self, images, net_name= 'CLIP Two-way'):
        self.images = images
        self.net_name = net_name
        
        if self.net_name == 'CLIP Two-way':
            self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        else:
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.num_test = len(images)
        
    def __getitem__(self,idx):
        img = self.images[idx]
        img = T.functional.resize(img,(224,224))
        img = T.functional.to_tensor(img).float()
        img = self.normalize(img)
        return img

    def __len__(self):
        return  self.num_test


class Stochastic_Search_Statistics():
    
    def __init__(self, subject = 1, device = "cuda"):

        self.device=device
        self.subject = subject
        self.net_models = initialize_net_metrics(self.device)
        # model_id = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        model_id = "openai/clip-vit-large-patch14"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.visionmodel = CLIPVisionModelWithProjection.from_pretrained(model_id).to(self.device)
        self.AEModel = AutoEncoder(config="gnet",
                                    inference=True,
                                    subject=self.subject,
                                    device=self.device)
        self.PeC = PearsonCorrCoef().to(self.device)
        self.PeC1 = PearsonCorrCoef(num_outputs=1).to(self.device) 
        self.mask_path = "data/preprocessed_data/subject{}/masks/".format(subject)
        subject_sizes = [0, 15724, 14278, 0, 0, 13039, 0, 12682]
        self.masks = {"nsd_general":torch.full((subject_sizes[self.subject],), True),
                        "V1":torch.load(self.mask_path + "V1.pt"),
                        "V2":torch.load(self.mask_path + "V2.pt"),
                        "V3":torch.load(self.mask_path + "V3.pt"),
                        "V4":torch.load(self.mask_path + "V4.pt"),
                        "early_vis":torch.load(self.mask_path + "early_vis.pt"),
                        "higher_vis":torch.load(self.mask_path + "higher_vis.pt")}  
    

    # Calcaulate the pearson correlation for each region of the brain.
    def calculate_brain_correlations(self, beta, beta_prime):
        masked_brain_correlations = {}
        # Calculate brain predictions
        for name, mask in self.masks.items():
            scores_raw = self.PeC1(beta.flatten()[mask].to(self.device), beta_prime.flatten()[mask].to(self.device))
            scores = float(scores_raw.detach().cpu().numpy())
            masked_brain_correlations[name] = scores
        
        return masked_brain_correlations
        
    # SSIM metric calculation
    def calculate_ssim(self, ground_truth, reconstruction):

        ground_truth   = ground_truth.resize((425, 425))
        reconstruction = reconstruction.resize((425, 425))
        
        ground_truth = np.array(ground_truth) / 255.0
        reconstruction = np.array(reconstruction) / 255.0

        ground_truth = rgb2gray(ground_truth)
        reconstruction = rgb2gray(reconstruction)

        return ssim(reconstruction, ground_truth, gaussian_weights=True, sigma=1.5, use_sample_covariance=True, data_range=ground_truth.max()-ground_truth.min())
            
        
    # Pixel Correlation Metric 
    def calculate_pixel_correlation(self, ground_truth, reconstruction):
        
        return pixel_correlation(ground_truth, reconstruction)
        
    
    #two_way_prob is the two way identification experiment between the given image and a random search reconstruction of a different sample with respect to the ground truth
    #clip_pearson is the pearson correlation score between the clips of the two given images
    #Sample type controls which of the types of image to pick a random sample between
        #   0 --> 0.png
        #   1 --> 1.png
        #   2 --> 2.png
        #   3 --> 3.png
        #   4 --> 4.png
        #   5 --> Ground Truth
    def calculate_clip_similarity_papaer(self, experiment_name, sample, sampleType=1, subject = 1):
        with torch.no_grad():
            exp_path = "/export/raid1/home/ojeda040/Second-Sight/reconstructions/subject{}/{}/".format(subject, experiment_name)
            
            folders = sorted([int(f.name) for f in os.scandir(exp_path) if f.is_dir() and f.name != 'results'])
            rand_list = [i for i in range(len(folders)) if folders[i] != sample and os.listdir(exp_path + str(folders[i]) + "/")]
            rand_index = random.choice(rand_list)
            sampleTypes = {0: "0.png", 1: "1.png", 2: "2.png", 3: "3.png", 4: "4.png", 5: "Ground Truth.png"}
            random_image = Image.open(exp_path + str(folders[rand_index]) + "/" + sampleTypes[sampleType])
            image = Image.open(exp_path + str(sample) + "/" + sampleTypes[sampleType])
            ground_truth = Image.open(exp_path + str(sample) + "/Ground Truth.png")
            
            inputs = self.processor(images=[ground_truth, image, random_image], return_tensors="pt", padding=True).to(self.device)
            outputs = self.visionmodel(**inputs)
            
            gt_feature = outputs.image_embeds[0].reshape((768))
            reconstruct_feature = outputs.image_embeds[1].reshape((768))
            clip_cosine_sim = torch.nn.functional.cosine_similarity(gt_feature, reconstruct_feature, dim=0)
            rand_image_feature = outputs.image_embeds[2].reshape((768))
            rand_image_feature /= rand_image_feature.norm(dim=-1, keepdim=True)
            gt_feature /= gt_feature.norm(dim=-1, keepdim=True)
            reconstruct_feature /= reconstruct_feature.norm(dim=-1, keepdim=True)
            
            loss = (torch.stack([gt_feature @ reconstruct_feature, gt_feature @ rand_image_feature]) *100)
            two_way_prob = loss.softmax(dim=0)[0]
            clip_pearson = self.PeC(gt_feature.flatten(), reconstruct_feature.flatten())
        return float(two_way_prob), float(clip_pearson), float(clip_cosine_sim)
        
    # clip_pearson is the pearson correlation score between the clips of the two given images
    def calculate_clip_cosine_sim(self, ground_truth, prediction):
        with torch.no_grad():
            inputs = self.processor(images=[ground_truth, prediction], return_tensors="pt", padding=True).to(self.device)
            outputs = self.visionmodel(**inputs)
            
            gt_feature = outputs.image_embeds[0].reshape((768))
            reconstruct_feature = outputs.image_embeds[1].reshape((768))
            clip_cosine_sim = torch.nn.functional.cosine_similarity(gt_feature, reconstruct_feature, dim=0)
        return float(clip_cosine_sim)
    

    
    def pairwise_corr_all(self, ground_truth, predictions):
        r = np.corrcoef(ground_truth, predictions)     #cosine_similarity(ground_truth, predictions)#
        r = r[:len(ground_truth), len(ground_truth):]  # rows: groundtruth, columns: predicitons
       
        # congruent pairs are on diagonal
        congruents = np.diag(r)
        
        # for each column (predicition) we should count the number of rows (groundtruth) that the value is lower than the congruent (e.g. success).
        success = r < congruents
        success_cnt = np.sum(success, 0)
        
        # note: diagonal of 'success' is always zero so we can discard it. That's why we divide by len-1
        perf = np.mean(success_cnt) / (len(ground_truth)-1)
        p = 1 - binom.cdf(perf*len(ground_truth)*(len(ground_truth)-1), len(ground_truth)*(len(ground_truth)-1), 0.5)
        
        return perf, p


    
    # CNN Metrics
    def net_metrics(self, images):
        
        # print(len(images))
        feat_list_dict = {}
        global feat_list
        feat_list = []
        def fn(module, inputs, outputs):
            feat_list.append(outputs.cpu().numpy())

        net_list = [
            ('Inception V3','avgpool'),
            ('CLIP Two-way','final'),
            ('AlexNet 2',2),
            ('AlexNet 5',5),
            ('AlexNet 7',7),
            ('EffNet-B','avgpool'),
            ('SwAV','avgpool')
            ]

        device = 0
        net = None
        batchsize=64

        for (net_name,layer) in net_list:
            feat_list = []
            dataset = batch_generator_external_images(images)
            loader = DataLoader(dataset,batchsize,shuffle=False)
            
            with torch.no_grad():
                for i,x in enumerate(loader):
                    x = x.to(self.device)
                    _ = self.net_models[f'{net_name}{layer}'](x)
                    
              
            feat_list = np.concatenate(feat_list)
    
            feat_list_dict[net_name] = feat_list
            
        return feat_list_dict
    
    
    # Grab the image indicies to create calculations on. 
    def image_indices(self, experiment_folder):
        # Grab the list of folders
        folders = [d for d in os.listdir(experiment_folder) if os.path.isdir(os.path.join(experiment_folder, d))]

        # Get just the image number and then sort the list. 
        indicies = []
        for i in range(len(folders)):
            try:
                int(folders[i]) 
                indicies.append(int(re.search(r'\d+', folders[i]).group()))
            except:
                pass
       
        indicies.sort()
        
        return indicies
    
    #TODO: Add check to see if files exist
    def create_beta_primes_mi(self, experiment_folder):
        
        GNet = GNet8_Encoder(device=self.device, subject=self.subject)
        # List of image numbers created. 
        idx = self.image_indices(experiment_folder)
        # Append rows to an empty DataFrame
        for i in tqdm(idx, desc="creating beta primes"):
            sample_path = f"{experiment_folder}{i}/"
            images = []
            names = []
            for j in range(5):
                image = Image.open(f"{sample_path}{j}.png")
                images.append(image)
                names.append(j)
                
            ground_truth_image = Image.open(f"{sample_path}ground_truth.png")
            images.append(ground_truth_image)
            names.append("ground_truth")
            if "secondsight" not in experiment_folder:
                low_level_image = Image.open(f"{sample_path}low_level.png")
                images.append(low_level_image)
                names.append("low_level")
            
            beta_primes = GNet.predict(images)
            for beta_prime, name in zip(beta_primes, names):
                torch.save(beta_prime, f"{sample_path}{name}_beta_prime.pt")
        
    def create_beta_primes_ss(self, experiment_folder):
        
        GNet = GNet8_Encoder(device=self.device, subject=self.subject)
        # List of image numbers created. 
        idx = self.image_indices(experiment_folder)
        # Append rows to an empty DataFrame
        for i in tqdm(idx, desc="creating beta primes"):
            sample_path = f"{experiment_folder}{i}/"
            images = []
            names = []

            ground_truth_image = Image.open(f"{sample_path}Ground Truth.png")
            images.append(ground_truth_image)
            names.append("ground_truth")

            ground_truth_image = Image.open(f"{sample_path}search_reconstruction.png")
            images.append(ground_truth_image)
            names.append("search_reconstruction")
            
            low_level_image = Image.open(f"{sample_path}MindEye.png")
            images.append(low_level_image)
            names.append("mindeye")

            low_level_image = Image.open(f"{sample_path}MindEye blurry.png")
            images.append(low_level_image)
            names.append("mindeye_blurry")
            
            beta_primes = GNet.predict(images)
            for beta_prime, name in zip(beta_primes, names):
                torch.save(beta_prime, "{}/{}_beta_prime.pt".format(experiment_folder + str(i), name))
                
    def generate_features_nsd_vision(self, method, low=False):
        folder_images = []
        directory_path = f"output/second_sight_paper/{method}/subject{self.subject}/"
        if low:
            filename = "low_level.png"
            feature_path = f"output/second_sight_paper/dataframes/{method}/subject{self.subject}/features_low/"
        else:
            filename = "0.png"
            feature_path = f"output/second_sight_paper/dataframes/{method}/subject{self.subject}/features/"
        os.makedirs(feature_path, exist_ok=True)
        
        for i in range(982):
            folder_images.append(Image.open(f"{directory_path}{i}/{filename}"))
        net_predictions = self.net_metrics(folder_images)
        for sample in tqdm(range(len(folder_images))):
            # Add the prediction at it's respected index to the dataframe. 
            os.makedirs(f"{feature_path}{sample}/", exist_ok=True)
        # Grab the key value pair in the dictionary. 
            for net_name, feature_list in net_predictions.items(): 
            # Iterate over the list of predictions
                np.save(f"{feature_path}{sample}/{net_name}.npy", feature_list[sample].flatten())
    # This is the method used for creating dataframes around file structures that are as follows:
    # {experiment_name}
    #   - subject{subject}
    #       - {image_number}
    #           - 0.png
    #           - 1.png
    #           - 2.png
    #           - 3.png
    #           - 4.png
    #           - low_level.png
    #           - ground_truth.png
    
    def create_dataframe_mi(self, method, mode):
        # Path to the folder
        if mode == "nsd_vision":
            directory_path = f"output/{method}_{mode}/subject{self.subject}/"
            dataframe_path = f"output/dataframes/{method}_{mode}/subject{self.subject}/"
            _, _, beta_samples, _, _, _, _ = load_nsd(vector="images", subject=self.subject, loader=False, average=True, nest=False)
        else:
            directory_path = f"output/mental_imagery_paper/{mode}/{method}/subject{self.subject}/"
            dataframe_path = f"output/mental_imagery_paper/{mode}/{method}/"
            beta_samples, _ = load_nsd_mental_imagery(vector = "c", subject=self.subject, mode=mode, stimtype="all", average=True, nest=True)
        os.makedirs(dataframe_path, exist_ok=True)
        os.makedirs(f"{dataframe_path}features/", exist_ok=True)
        
        # Create betas if needed
        self.create_beta_primes_mi(directory_path)
        
        # List of image numbers created. 
        idx = self.image_indices(directory_path)
        
        print("IDX: ", len(idx), idx)
        # Create an Empty DataFrame
        # Object With column names only
        # Sample Indicator: 
            #   10 --> ground_truth
            #   11 --> low_level
            #   12 --> final reconstruction
            #   13 --> best_selected_image
        df = pd.DataFrame(columns = ['ID', 'Subject', 'Method', 'Mode', 'Sample Count', 'Batch Number', 'Sample Indicator', 'Strength', 'Brain Correlation V1', 'Brain Correlation V2', 
                                     'Brain Correlation V3', 'Brain Correlation V4', 'Brain Correlation Early Visual', 'Brain Correlation Higher Visual',
                                     'Brain Correlation NSD General', 'SSIM', 'Pixel Correlation', 'CLIP Cosine', 'CLIP Two-way', 'AlexNet 2', 
                                     'AlexNet 5', 'AlexNet 7', 'Inception V3', 'EffNet-B', 'SwAV', 'CLIP Two-way 1000', 'AlexNet 2 1000', 
                                     'AlexNet 5 1000', 'AlexNet 7 1000', 'Inception V3 1000'])
        
        # Dataframe index count. 
        df_row_num = 0
        
        # Images per folder for net metrics.
        folder_images = []
        
        # Append rows to an empty DataFrame
        for i in tqdm(idx, desc="creating dataframe rows"):
            sample_path = f"{directory_path}{i}/"
            # Ground Truth Image
            ground_truth = Image.open(f'{sample_path}ground_truth.png')
            clip_cosine_sim_gt = self.calculate_clip_cosine_sim(ground_truth, ground_truth)
            pix_corr_gt = self.calculate_pixel_correlation(ground_truth, ground_truth)
            ssim_gt = self.calculate_ssim(ground_truth, ground_truth)
            
            beta_prime = torch.load(f"{sample_path}ground_truth_beta_prime.pt")
            brain_correlations = self.calculate_brain_correlations(beta_samples[i], beta_prime)
            folder_images.append(ground_truth)
            
            df.loc[df_row_num] = {'ID' : i, 'Subject' : self.subject, 'Method' : method, 'Mode' : mode, 'Sample Indicator' : 10, 'Strength' : np.nan, 'Brain Correlation V1' : brain_correlations["V1"],
                        'Brain Correlation V2' : brain_correlations["V2"], 'Brain Correlation V3' : brain_correlations["V3"], 
                        'Brain Correlation V4' : brain_correlations["V4"], 'Brain Correlation Early Visual' : brain_correlations["early_vis"],
                        'Brain Correlation Higher Visual' : brain_correlations["higher_vis"], 'Brain Correlation NSD General' : brain_correlations["nsd_general"],
                        'SSIM' : ssim_gt, 'Pixel Correlation' : pix_corr_gt, 'CLIP Cosine' : clip_cosine_sim_gt}
            df_row_num += 1
            if method != "secondsight":
                # Low Level Image
                low_level = Image.open(f'{sample_path}low_level.png')
                clip_cosine_sim_low = self.calculate_clip_cosine_sim(ground_truth, low_level)
                pix_corr_low = self.calculate_pixel_correlation(ground_truth, low_level)
                ssim_low = self.calculate_ssim(ground_truth, low_level)
                
                beta_prime = torch.load(f"{sample_path}low_level_beta_prime.pt")
                brain_correlations = self.calculate_brain_correlations(beta_samples[i], beta_prime)
                folder_images.append(low_level)
                
                df.loc[df_row_num] = {'ID' : i, 'Subject' : self.subject, 'Method' : method, 'Mode' : mode, 'Sample Indicator' : 11, 'Strength' : np.nan, 'Brain Correlation V1' : brain_correlations["V1"],
                            'Brain Correlation V2' : brain_correlations["V2"], 'Brain Correlation V3' : brain_correlations["V3"], 
                            'Brain Correlation V4' : brain_correlations["V4"], 'Brain Correlation Early Visual' : brain_correlations["early_vis"],
                            'Brain Correlation Higher Visual' : brain_correlations["higher_vis"], 'Brain Correlation NSD General' : brain_correlations["nsd_general"],
                            'SSIM' : ssim_low, 'Pixel Correlation' : pix_corr_low, 'CLIP Cosine' : clip_cosine_sim_low}
                df_row_num += 1
            for j in range(5):
                rep = Image.open(f'{sample_path}{j}.png')       
                # Make dataframe row for rep of reconstruction
                pix_corr_rep = self.calculate_pixel_correlation(ground_truth, rep)
                ssim_rep = self.calculate_ssim(ground_truth, rep)
                clip_cosine_sim_rep = self.calculate_clip_cosine_sim(ground_truth, rep)
                # Pearson correlation for each region of the brain. 
                beta_prime = torch.load(f"{sample_path}{j}_beta_prime.pt")
                brain_correlations = self.calculate_brain_correlations(beta_samples[i], beta_prime)
                folder_images.append(rep)
                
                df.loc[df_row_num] = {'ID' : i, 'Subject' : self.subject, 'Method' : method, 'Mode' : mode, 'Sample Count': j, 'Sample Indicator' : 12, 'Strength' : None, 'Brain Correlation V1' : brain_correlations["V1"],
                            'Brain Correlation V2' : brain_correlations["V2"], 'Brain Correlation V3' : brain_correlations["V3"], 
                            'Brain Correlation V4' : brain_correlations["V4"], 'Brain Correlation Early Visual' : brain_correlations["early_vis"],
                            'Brain Correlation Higher Visual' : brain_correlations["higher_vis"], 'Brain Correlation NSD General' : brain_correlations["nsd_general"],
                            'SSIM' : ssim_rep, 'Pixel Correlation' : pix_corr_rep, 'CLIP Cosine' : clip_cosine_sim_rep}
                df_row_num += 1
                
            net_predictions = self.net_metrics(folder_images)
            for sample in range(len(folder_images)):
                # Add the prediction at it's respected index to the dataframe. 
                dataframe_index = df_row_num - len(folder_images) + sample
                os.makedirs(f"{dataframe_path}features/{dataframe_index}/", exist_ok=True)
            # Grab the key value pair in the dictionary. 
                for net_name, feature_list in net_predictions.items(): 
                # Iterate over the list of predictions
                    np.save(f"{dataframe_path}features/{dataframe_index}/{net_name}.npy", feature_list[sample].flatten())
            folder_images = []
        
        # Computing CNN metrics for whole dataframe
        df_ground_truth     = df.loc[(df['Sample Indicator'] == 10)]
        df_low_level        = df.loc[(df['Sample Indicator'] == 11)]
        df_final_samples    = df.loc[(df['Sample Indicator'] == 12)]
        df_final_samples_0  = df_final_samples.loc[(df_final_samples['Sample Count'] == 0)]
        df_final_samples_1  = df_final_samples.loc[(df_final_samples['Sample Count'] == 1)]
        df_final_samples_2  = df_final_samples.loc[(df_final_samples['Sample Count'] == 2)]
        df_final_samples_3  = df_final_samples.loc[(df_final_samples['Sample Count'] == 3)]
        df_final_samples_4  = df_final_samples.loc[(df_final_samples['Sample Count'] == 4)]

        # Compute CNN Metrics and Two-Way comparisons WITHIN dataframe
        cnn_metrics_low_level = compute_cnn_metrics(create_cnn_numpy_array(df_ground_truth, f"{dataframe_path}features/"), create_cnn_numpy_array(df_low_level, f"{dataframe_path}features/"))
        cnn_metrics_0 = compute_cnn_metrics(create_cnn_numpy_array(df_ground_truth, f"{dataframe_path}features/"), create_cnn_numpy_array(df_final_samples_0, f"{dataframe_path}features/"))
        cnn_metrics_1 = compute_cnn_metrics(create_cnn_numpy_array(df_ground_truth, f"{dataframe_path}features/"), create_cnn_numpy_array(df_final_samples_1, f"{dataframe_path}features/"))
        cnn_metrics_2 = compute_cnn_metrics(create_cnn_numpy_array(df_ground_truth, f"{dataframe_path}features/"), create_cnn_numpy_array(df_final_samples_2, f"{dataframe_path}features/"))
        cnn_metrics_3 = compute_cnn_metrics(create_cnn_numpy_array(df_ground_truth, f"{dataframe_path}features/"), create_cnn_numpy_array(df_final_samples_3, f"{dataframe_path}features/"))
        cnn_metrics_4 = compute_cnn_metrics(create_cnn_numpy_array(df_ground_truth, f"{dataframe_path}features/"), create_cnn_numpy_array(df_final_samples_4, f"{dataframe_path}features/"))
        sample_rep_metrics = [cnn_metrics_0, cnn_metrics_1, cnn_metrics_2, cnn_metrics_3, cnn_metrics_4]
        net_list = [
            'Inception V3',
            'CLIP Two-way',
            'AlexNet 2',
            'AlexNet 5',
            'AlexNet 7',
            'EffNet-B',
            'SwAV']
        for index, row in df.iterrows():
            sample_id = row['ID']
            if row['Sample Indicator'] == 11:
                for net_name in net_list:
                    df.at[index, net_name] = cnn_metrics_low_level[net_name][sample_id]
            elif row['Sample Indicator'] == 12:
                sample_count = int(row['Sample Count'])
                for net_name in net_list:
                    df.at[index, net_name] = sample_rep_metrics[sample_count][net_name][sample_id]          
                 
        # Compute Two-Way metrics against shared1000 samples
        if method != "secondsight":
            cnn_metrics_low_level = compute_cnn_metrics_shared1000(create_cnn_numpy_array(df_ground_truth, f"{dataframe_path}features/"), create_cnn_numpy_array(df_low_level, f"{dataframe_path}features/"), method, self.subject, low=True)
        cnn_metrics_0 = compute_cnn_metrics_shared1000(create_cnn_numpy_array(df_ground_truth, f"{dataframe_path}features/"), create_cnn_numpy_array(df_final_samples_0, f"{dataframe_path}features/"), method, self.subject)
        cnn_metrics_1 = compute_cnn_metrics_shared1000(create_cnn_numpy_array(df_ground_truth, f"{dataframe_path}features/"), create_cnn_numpy_array(df_final_samples_1, f"{dataframe_path}features/"), method, self.subject)
        cnn_metrics_2 = compute_cnn_metrics_shared1000(create_cnn_numpy_array(df_ground_truth, f"{dataframe_path}features/"), create_cnn_numpy_array(df_final_samples_2, f"{dataframe_path}features/"), method, self.subject)
        cnn_metrics_3 = compute_cnn_metrics_shared1000(create_cnn_numpy_array(df_ground_truth, f"{dataframe_path}features/"), create_cnn_numpy_array(df_final_samples_3, f"{dataframe_path}features/"), method, self.subject)
        cnn_metrics_4 = compute_cnn_metrics_shared1000(create_cnn_numpy_array(df_ground_truth, f"{dataframe_path}features/"), create_cnn_numpy_array(df_final_samples_4, f"{dataframe_path}features/"), method, self.subject)
        sample_rep_metrics = [cnn_metrics_0, cnn_metrics_1, cnn_metrics_2, cnn_metrics_3, cnn_metrics_4]
        net_list = [
            'Inception V3',
            'CLIP Two-way',
            'AlexNet 2',
            'AlexNet 5',
            'AlexNet 7']
        for index, row in df.iterrows():
            sample_id = row['ID']
            if method != "secondsight" and row['Sample Indicator'] == 11:
                for net_name in net_list:
                    df.at[index, f"{net_name} 1000"] = cnn_metrics_low_level[net_name][sample_id]
            elif row['Sample Indicator'] == 12:
                sample_count = int(row['Sample Count'])
                for net_name in net_list:
                    df.at[index, f"{net_name} 1000"] = sample_rep_metrics[sample_count][net_name][sample_id]       
                        
        print(df.shape)
        print(df)
        df.to_csv(f"{dataframe_path}subject{self.subject}_statistics_{len(idx)}.csv")
        
    def create_dataframe_SS(self, experiment_name, logging = False, mode = "nsd_vision", make_beta_primes=True):
        # Path to the folder
        # directory_path = "/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight-Archive/reconstructions/subject{}/{}/".format(self.subject, experiment_name)
        # dataframe_path = "Second-Sight-Archive/reconstructions/subject{}/dataframes/".format(self.subject)
        directory_path = "output/{}/subject{}/".format(experiment_name, self.subject)
        dataframe_path = "output/dataframes/{}/subject{}/".format(experiment_name, self.subject)
        os.makedirs(dataframe_path, exist_ok=True)
        
        # Create betas if needed
        if make_beta_primes:
            self.create_beta_primes_ss(directory_path)
        
        # List of image numbers created. 
        # idx = [i for i in range(982)]
        # idx = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 71, 72, 73, 74, 75, 77, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 106, 107, 108, 109, 110, 111, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140]
        idx = self.image_indices(directory_path)
        # idx = self.paper_image_indices
        print("IDX: ", len(idx), idx)
        # Autoencoded avearged brain samples 
        if mode == "nsd_vision":
            _, _, beta_samples, _, _, _, _ = load_nsd(vector="images", subject=self.subject, loader=False, average=False, nest=True)
            averaged_beta_samples = torch.mean(beta_samples, axis=1)
        else:
            beta_samples, _ = load_nsd_mental_imagery(vector = "c", subject=self.subject, mode=mode, stimtype="all", average=False, nest=True)
            averaged_beta_samples = torch.mean(beta_samples, axis=1)
        
        
        # Create an Empty DataFrame
        # Object With column names only
        # Sample Indicator: 
            #   0 --> iter_0
            #   1 --> iter_1
            #   2 --> iter_2
            #   3 --> iter_3
            #   4 --> iter_4
            #   5 --> iter_5
            #   6 --> iter_6
            #   7 --> iter_7
            #   8 --> iter_8
            #   9 --> iter_9
            #   10 --> ground_truth
            #   11 --> MindEye reconstruction blurry (low level)
            #   12 --> final reconstruction distribution
            #   13 --> search reconstruction
            #   14 --> MindEye reconstruction
        df = pd.DataFrame(columns = ['ID', 'Subject', 'Method', 'Mode', 'Sample Count', 'Batch Number', 'Sample Indicator', 'Strength', # Misc metadata
                                     
                                     'Brain Correlation V1', 'Brain Correlation V2', 'Brain Correlation V3', 'Brain Correlation V4', #Brain correlation scores
                                     'Brain Correlation Early Visual', 'Brain Correlation Higher Visual','Brain Correlation NSD General', 
                                     
                                     'Target Variance', 'Beta Prime Variance V1', 'Beta Prime Variance V2', 'Beta Prime Variance V3' # Brain variance metrics
                                     'Beta Prime Variance V4', 'Beta Prime Variance Early Visual', 'Beta Prime Variance Higher Visual', 'Beta Prime Variance NSD General'
                                     
                                     'SSIM', 'Pixel Correlation', 'CLIP Cosine', 'CLIP Two-way', 'AlexNet 2', #Feature metrics
                                     'AlexNet 5', 'AlexNet 7', 'Inception V3', 'EffNet-B', 'SwAV' ])
        
        # Dataframe index count. 
        df_row_num = 0
        
        # Images per folder for net metrics.
        folder_images = []
        
        # Folders in the directory and sample number for dataframe operations
        folders = {}
        if(logging):
            folders = {"iter_0" : 0, "iter_1" : 1 , "iter_2" : 2, "iter_3" : 3, "iter_4" : 4, "iter_5" : 5, "iter_6" : 6, "iter_7" : 7, "iter_8" : 8, "iter_9" : 9, "best_distribution": 12}
        else:
            folders = {"best_distribution": 12}
        
        # Append rows to an empty DataFrame
        for i in tqdm(idx, desc="creating dataframe rows"):
            sample_path = f"{directory_path}{i}/"
            # var_list = np.load(sample_path + "var_list.npy")
            # print(f"{i}: Variance List: {var_list}")
            ae_beta = self.AEModel.predict(prepare_betas(beta_samples[i])).detach().cpu()
            target_variance = bootstrap_variance(ae_beta)


            # Ground Truth Image
            ground_truth = Image.open(f'{sample_path}Ground Truth.png')
            clip_cosine_sim_gt = self.calculate_clip_cosine_sim(ground_truth, ground_truth)
            pix_corr_gt = self.calculate_pixel_correlation(ground_truth, ground_truth)
            ssim_gt = self.calculate_ssim(ground_truth, ground_truth)
            
            beta_prime = torch.load(f"{sample_path}ground_truth_beta_prime.pt")
            brain_correlations = self.calculate_brain_correlations(averaged_beta_samples[i], beta_prime)
            folder_images.append(ground_truth)
            
            df.loc[df_row_num] = {'ID' : i, 'Subject' : self.subject, 'Method' : experiment_name, 'Mode' : mode,'Sample Indicator' : 10, 'Strength' : np.nan, 'Brain Correlation V1' : brain_correlations["V1"],
                        'Brain Correlation V2' : brain_correlations["V2"], 'Brain Correlation V3' : brain_correlations["V3"], 
                        'Brain Correlation V4' : brain_correlations["V4"], 'Brain Correlation Early Visual' : brain_correlations["early_vis"],
                        'Brain Correlation Higher Visual' : brain_correlations["higher_vis"], 'Brain Correlation NSD General' : brain_correlations["nsd_general"],
                        'Target Variance' : target_variance,
                        'SSIM' : ssim_gt, 'Pixel Correlation' : pix_corr_gt, 'CLIP Cosine' : clip_cosine_sim_gt}
            df_row_num += 1

            # Low Level Image
            low_level = Image.open(f'{sample_path}MindEye blurry.png')
            clip_cosine_sim_low = self.calculate_clip_cosine_sim(ground_truth, low_level)
            pix_corr_low = self.calculate_pixel_correlation(ground_truth, low_level)
            ssim_low = self.calculate_ssim(ground_truth, low_level)
            
            beta_prime = torch.load(f"{sample_path}mindeye_blurry_beta_prime.pt")
            brain_correlations = self.calculate_brain_correlations(averaged_beta_samples[i], beta_prime)
            folder_images.append(low_level)
            
            df.loc[df_row_num] = {'ID' : i, 'Subject' : self.subject, 'Method' : experiment_name, 'Mode' : mode, 'Sample Indicator' : 11, 'Strength' : np.nan, 'Brain Correlation V1' : brain_correlations["V1"],
                        'Brain Correlation V2' : brain_correlations["V2"], 'Brain Correlation V3' : brain_correlations["V3"], 
                        'Brain Correlation V4' : brain_correlations["V4"], 'Brain Correlation Early Visual' : brain_correlations["early_vis"],
                        'Brain Correlation Higher Visual' : brain_correlations["higher_vis"], 'Brain Correlation NSD General' : brain_correlations["nsd_general"],
                        'Target Variance' : target_variance,
                        'SSIM' : ssim_low, 'Pixel Correlation' : pix_corr_low, 'CLIP Cosine' : clip_cosine_sim_low}
            df_row_num += 1

            # MindEye Reconstruction Image
            mindeye = Image.open(f'{sample_path}MindEye.png')
            clip_cosine_sim_me = self.calculate_clip_cosine_sim(ground_truth, mindeye)
            pix_corr_me = self.calculate_pixel_correlation(ground_truth, mindeye)
            ssim_me = self.calculate_ssim(ground_truth, mindeye)
            
            beta_prime = torch.load(f"{sample_path}mindeye_beta_prime.pt")
            brain_correlations = self.calculate_brain_correlations(averaged_beta_samples[i], beta_prime)
            folder_images.append(mindeye)
            
            df.loc[df_row_num] = {'ID' : i, 'Subject' : self.subject, 'Method' : experiment_name, 'Mode' : mode, 'Sample Indicator' : 14, 'Strength' : 0.85, 'Brain Correlation V1' : brain_correlations["V1"],
                        'Brain Correlation V2' : brain_correlations["V2"], 'Brain Correlation V3' : brain_correlations["V3"], 
                        'Brain Correlation V4' : brain_correlations["V4"], 'Brain Correlation Early Visual' : brain_correlations["early_vis"],
                        'Brain Correlation Higher Visual' : brain_correlations["higher_vis"], 'Brain Correlation NSD General' : brain_correlations["nsd_general"],
                        'Target Variance' : target_variance,
                        'SSIM' : ssim_me, 'Pixel Correlation' : pix_corr_me, 'CLIP Cosine' : clip_cosine_sim_me}
            df_row_num += 1
            
            # Search Reconstruction Image
            search_reconstruction = Image.open(f'{sample_path}search_reconstruction.png')
            clip_cosine_sim_sr = self.calculate_clip_cosine_sim(ground_truth, search_reconstruction)
            pix_corr_sr = self.calculate_pixel_correlation(ground_truth, search_reconstruction)
            ssim_sr = self.calculate_ssim(ground_truth, search_reconstruction)
            
            beta_prime = torch.load(f"{sample_path}search_reconstruction_beta_prime.pt")
            brain_correlations = self.calculate_brain_correlations(averaged_beta_samples[i], beta_prime)
            folder_images.append(search_reconstruction)
            
            df.loc[df_row_num] = {'ID' : i, 'Subject' : self.subject, 'Method' : experiment_name, 'Mode' : mode, 'Sample Indicator' : 13, 'Strength' : np.nan, 'Brain Correlation V1' : brain_correlations["V1"],
                        'Brain Correlation V2' : brain_correlations["V2"], 'Brain Correlation V3' : brain_correlations["V3"], 
                        'Brain Correlation V4' : brain_correlations["V4"], 'Brain Correlation Early Visual' : brain_correlations["early_vis"],
                        'Brain Correlation Higher Visual' : brain_correlations["higher_vis"], 'Brain Correlation NSD General' : brain_correlations["nsd_general"],
                        'Target Variance' : target_variance,
                        'SSIM' : ssim_sr, 'Pixel Correlation' : pix_corr_sr, 'CLIP Cosine' : clip_cosine_sim_sr}
            df_row_num += 1

            for folder, sample_number in folders.items():
                if os.path.exists(f'{sample_path}{folder}/'):
                    if folder == "best_distribution":
                        folder_path = f'{sample_path}{folder}/'
                        with open(f'{folder_path}strength.txt', 'r') as f:
                            strength = float(f.read())
                        iter_variance_dict = {"nsd_general": None,"V1":None,"V2":None,"V3":None,"V4":None,"early_vis":None,"higher_vis":None} 
                    elif folder == "iter_0":
                        folder_path = f'{sample_path}{folder}/'
                        strength = 0.92
                        iter_variance_dict = get_iter_variance(f'{sample_path}{folder}/', self.masks)
                        # tqdm.write(f"TARGET VAR: {target_variance:.5f}, NSD VAR: {iter_variance_dict['nsd_general']:.5f}, V1 VAR: {iter_variance_dict['V1']:.5f}, V2 VAR: {iter_variance_dict['V2']:.5f}, V3 VAR: {iter_variance_dict['V3']:.5f}, V4 VAR: {iter_variance_dict['V4']:.5f}, EARLY VIS VAR: {iter_variance_dict['early_vis']:.5f}, HIGHER VIS VAR: {iter_variance_dict['higher_vis']:.5f}")
                    else:
                        folder_path = f'{sample_path}{folder}/best_batch/'
                        strength = float(torch.load(f'{sample_path}{folder}/iter_strength.pt'))
                        iter_variance_dict = get_iter_variance(f'{sample_path}{folder}/', self.masks)
                        # tqdm.write(f"TARGET VAR: {target_variance:.5f}, NSD VAR: {iter_variance_dict['nsd_general']:.5f}, V1 VAR: {iter_variance_dict['V1']:.5f}, V2 VAR: {iter_variance_dict['V2']:.5f}, V3 VAR: {iter_variance_dict['V3']:.5f}, V4 VAR: {iter_variance_dict['V4']:.5f}, EARLY VIS VAR: {iter_variance_dict['early_vis']:.5f}, HIGHER VIS VAR: {iter_variance_dict['higher_vis']:.5f}")
                    for rep in range(5):
                        reconstruction = Image.open(f'{folder_path}images/{rep}.png')
                        pix_corr_rep = self.calculate_pixel_correlation(ground_truth, reconstruction)
                        ssim_rep = self.calculate_ssim(ground_truth, reconstruction)
                        clip_cosine_sim_rep = self.calculate_clip_cosine_sim(ground_truth, reconstruction)
                        # Pearson correlation for each region of the brain. 
                        beta_prime = torch.load(f"{folder_path}beta_primes/{rep}.pt")
                        brain_correlations = self.calculate_brain_correlations(averaged_beta_samples[i], beta_prime)
                        
                        folder_images.append(reconstruction)
                        
                        df.loc[df_row_num] = {'ID' : i, 'Subject' : self.subject, 'Method' : experiment_name, 'Mode' : mode, 'Sample Count': rep, 'Sample Indicator' : sample_number, 'Strength' : strength, 
                                              'Brain Correlation V1' : brain_correlations["V1"],'Brain Correlation V2' : brain_correlations["V2"], 'Brain Correlation V3' : brain_correlations["V3"], 
                                            'Brain Correlation V4' : brain_correlations["V4"], 'Brain Correlation Early Visual' : brain_correlations["early_vis"],
                                            'Brain Correlation Higher Visual' : brain_correlations["higher_vis"], 'Brain Correlation NSD General' : brain_correlations["nsd_general"],
                                            'Target Variance' : target_variance, 'Beta Prime Variance V1' : iter_variance_dict["V1"], 'Beta Prime Variance V2' : iter_variance_dict["V2"],
                                            'Beta Prime Variance V3' : iter_variance_dict["V3"], 'Beta Prime Variance V4' : iter_variance_dict["V4"], 'Beta Prime Variance Early Visual' : iter_variance_dict["early_vis"],
                                            'Beta Prime Variance Higher Visual' : iter_variance_dict["higher_vis"], 'Beta Prime Variance NSD General' : iter_variance_dict["nsd_general"],
                                            'SSIM' : ssim_rep, 'Pixel Correlation' : pix_corr_rep, 'CLIP Cosine' : clip_cosine_sim_rep}
                        df_row_num += 1
                    
            # Calculate CNN metrics
            net_predictions = self.net_metrics(folder_images)
            for sample in range(len(folder_images)):
                # Add the prediction at it's respected index to the dataframe. 
                dataframe_index = df_row_num - len(folder_images) + sample
                os.makedirs(f"{dataframe_path}features/{dataframe_index}/", exist_ok=True)
            # Grab the key value pair in the dictionary. 
                for net_name, feature_list in net_predictions.items(): 
                # Iterate over the list of predictions
                    np.save(f"{dataframe_path}features/{dataframe_index}/{net_name}.npy", feature_list[sample].flatten())
            folder_images = []
        
        # Computing CNN metrics for whole dataframe
        df_ground_truth         = df.loc[(df['Sample Indicator'] == 10)]
        df_low_level            = df.loc[(df['Sample Indicator'] == 11)]
        df_final_samples        = df.loc[(df['Sample Indicator'] == 12)]
        df_search_reconstruction= df.loc[(df['Sample Indicator'] == 13)]
        df_mindeye              = df.loc[(df['Sample Indicator'] == 14)]
        df_final_samples_0      = df_final_samples.loc[(df_final_samples['Sample Count'] == 0)]
        df_final_samples_1      = df_final_samples.loc[(df_final_samples['Sample Count'] == 1)]
        df_final_samples_2      = df_final_samples.loc[(df_final_samples['Sample Count'] == 2)]
        df_final_samples_3      = df_final_samples.loc[(df_final_samples['Sample Count'] == 3)]
        df_final_samples_4      = df_final_samples.loc[(df_final_samples['Sample Count'] == 4)]

        cnn_metrics_low_level = compute_cnn_metrics(create_cnn_numpy_array(df_ground_truth, f"{dataframe_path}features/"), create_cnn_numpy_array(df_low_level, f"{dataframe_path}features/"))
        cnn_metrics_search_reconstruction = compute_cnn_metrics(create_cnn_numpy_array(df_ground_truth, f"{dataframe_path}features/"), create_cnn_numpy_array(df_search_reconstruction, f"{dataframe_path}features/"))
        cnn_metrics_mindeye = compute_cnn_metrics(create_cnn_numpy_array(df_ground_truth, f"{dataframe_path}features/"), create_cnn_numpy_array(df_mindeye, f"{dataframe_path}features/"))
        cnn_metrics_0 = compute_cnn_metrics(create_cnn_numpy_array(df_ground_truth, f"{dataframe_path}features/"), create_cnn_numpy_array(df_final_samples_0, f"{dataframe_path}features/"))
        cnn_metrics_1 = compute_cnn_metrics(create_cnn_numpy_array(df_ground_truth, f"{dataframe_path}features/"), create_cnn_numpy_array(df_final_samples_1, f"{dataframe_path}features/"))
        cnn_metrics_2 = compute_cnn_metrics(create_cnn_numpy_array(df_ground_truth, f"{dataframe_path}features/"), create_cnn_numpy_array(df_final_samples_2, f"{dataframe_path}features/"))
        cnn_metrics_3 = compute_cnn_metrics(create_cnn_numpy_array(df_ground_truth, f"{dataframe_path}features/"), create_cnn_numpy_array(df_final_samples_3, f"{dataframe_path}features/"))
        cnn_metrics_4 = compute_cnn_metrics(create_cnn_numpy_array(df_ground_truth, f"{dataframe_path}features/"), create_cnn_numpy_array(df_final_samples_4, f"{dataframe_path}features/"))
        sample_rep_metrics = [cnn_metrics_0, cnn_metrics_1, cnn_metrics_2, cnn_metrics_3, cnn_metrics_4]
        net_list = [
            'Inception V3',
            'CLIP Two-way',
            'AlexNet 2',
            'AlexNet 5',
            'AlexNet 7',
            'EffNet-B',
            'SwAV']
        for index, row in tqdm(df.iterrows(), desc="adding CNN metrics"):
            sample_id = row['ID']
            if row['Sample Indicator'] == 11:
                for net_name in net_list:
                    df.at[index, net_name] = cnn_metrics_low_level[net_name][sample_id]
            elif row['Sample Indicator'] == 13:
                for net_name in net_list:
                    df.at[index, net_name] = cnn_metrics_search_reconstruction[net_name][sample_id]
            elif row['Sample Indicator'] == 14:
                for net_name in net_list:
                    df.at[index, net_name] = cnn_metrics_mindeye[net_name][sample_id]
            elif row['Sample Indicator'] == 12:
                sample_count = int(row['Sample Count'])
                for net_name in net_list:
                    df.at[index, net_name] = sample_rep_metrics[sample_count][net_name][sample_id]
                                           
        print(df.shape)
        print(df)
        df.to_csv(dataframe_path + "statistics_df_" + experiment_name + "_" + str(len(idx)) +  ".csv")
    
    
def main():
    
    # SSS = Stochastic_Search_Statistics(subject = 1, device="cuda:1")
    # # SSS.create_dataframe_SS("ss_mi_vision", logging = True, mode="vision")
    # # SSS.create_dataframe_SS("ss_mi_imagery", logging = True, mode="imagery")
    # SSS.create_dataframe_mi(mode="vision", method="mindeye")
    # SSS.create_dataframe_mi(mode="vision", method="tagaki")
    # SSS.create_dataframe_mi(mode="vision", method="braindiffuser")
    # SSS.create_dataframe_mi(mode="imagery", method="mindeye")
    # SSS.create_dataframe_mi(mode="imagery", method="tagaki")
    # SSS.create_dataframe_mi(mode="imagery", method="braindiffuser")
    # SSS.create_dataframe_mi(mode="vision", method="secondsight")
    # SSS.create_dataframe_mi(mode="imagery", method="secondsight")
    

    # SSS = Stochastic_Search_Statistics(subject = 2, device="cuda:1")
    # # SSS.create_dataframe_SS("ss_mi_vision", logging = True, mode="vision")
    # # SSS.create_dataframe_SS("ss_mi_imagery", logging = True, mode="imagery")
    # SSS.create_dataframe_mi(mode="vision", method="mindeye")
    # SSS.create_dataframe_mi(mode="vision", method="tagaki")
    # SSS.create_dataframe_mi(mode="vision", method="braindiffuser")
    # SSS.create_dataframe_mi(mode="imagery", method="mindeye")
    # SSS.create_dataframe_mi(mode="imagery", method="tagaki")
    # SSS.create_dataframe_mi(mode="imagery", method="braindiffuser")
    # SSS.create_dataframe_mi(mode="vision", method="secondsight")
    # SSS.create_dataframe_mi(mode="imagery", method="secondsight")

    # SSS = Stochastic_Search_Statistics(subject = 5, device="cuda:1")
    # # SSS.create_dataframe_SS("ss_mi_vision", logging = True, mode="vision")
    # # SSS.create_dataframe_SS("ss_mi_imagery", logging = True, mode="imagery")
    # SSS.create_dataframe_mi(mode="vision", method="mindeye")
    # SSS.create_dataframe_mi(mode="vision", method="tagaki")
    # SSS.create_dataframe_mi(mode="vision", method="braindiffuser")
    # SSS.create_dataframe_mi(mode="imagery", method="mindeye")
    # SSS.create_dataframe_mi(mode="imagery", method="tagaki")
    # SSS.create_dataframe_mi(mode="imagery", method="braindiffuser")
    # SSS.create_dataframe_mi(mode="vision", method="secondsight")
    # SSS.create_dataframe_mi(mode="imagery", method="secondsight")

    # SSS = Stochastic_Search_Statistics(subject = 7, device="cuda:1")
    # # SSS.create_dataframe_SS("ss_mi_vision", logging = True, mode="vision")
    # # SSS.create_dataframe_SS("ss_mi_imagery", logging = True, mode="imagery")
    # SSS.create_dataframe_mi(mode="vision", method="mindeye")
    # SSS.create_dataframe_mi(mode="vision", method="tagaki")
    # SSS.create_dataframe_mi(mode="vision", method="braindiffuser")
    # SSS.create_dataframe_mi(mode="imagery", method="mindeye")
    # SSS.create_dataframe_mi(mode="imagery", method="tagaki")
    # SSS.create_dataframe_mi(mode="imagery", method="braindiffuser")
    # SSS.create_dataframe_mi(mode="vision", method="secondsight")
    # SSS.create_dataframe_mi(mode="imagery", method="secondsight")

    SSS = Stochastic_Search_Statistics(subject = 1, device="cuda:1")
    # SSS.create_dataframe_SS("mindeye_extension_v6", logging = True, make_beta_primes=False)
    SSS.create_dataframe_mi(mode="nsd_vision", method="tagaki")
    

    SSS = Stochastic_Search_Statistics(subject = 2, device="cuda:1")
    SSS.create_dataframe_SS("mindeye_extension_v6", logging = True, make_beta_primes=False)
    SSS.create_dataframe_mi(mode="nsd_vision", method="tagaki")

    SSS = Stochastic_Search_Statistics(subject = 5, device="cuda:1")
    SSS.create_dataframe_SS("mindeye_extension_v6", logging = True)
    SSS.create_dataframe_mi(mode="nsd_vision", method="tagaki")

    SSS = Stochastic_Search_Statistics(subject = 7, device="cuda:1")
    SSS.create_dataframe_SS("mindeye_extension_v6", logging = True)
    SSS.create_dataframe_mi(mode="nsd_vision", method="tagaki")

if __name__ == "__main__":
    main()
