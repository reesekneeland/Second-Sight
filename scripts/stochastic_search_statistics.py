import os
import sys
import torch
import pandas as pd
from PIL import Image
sys.path.append('src')
from utils import *
from tqdm import tqdm
from alexnet_encoder import AlexNetEncoder
from gnet8_encoder import GNet8_Encoder
from autoencoder import AutoEncoder
from torchmetrics import PearsonCorrCoef
from stochastic_search import StochasticSearch
import cv2
import random
from transformers import AutoProcessor, CLIPVisionModelWithProjection
import math
import re
import numpy as np
from skimage import io
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim
import os.path


class Stochastic_Search_Statistics():
    
    def __init__(self, big=False, subject = 1, device="cuda"):

        self.device=device
        # model_id = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        model_id = "openai/clip-vit-large-patch14"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.visionmodel = CLIPVisionModelWithProjection.from_pretrained(model_id).to(self.device)
        self.PeC = PearsonCorrCoef().to(self.device)
        self.mask_path = "/export/raid1/home/ojeda040/Second-Sight/masks/subject{}/".format(subject)
        if big:
            self.masks = {0:torch.full((15724,), False),
                        1:torch.load(self.mask_path + "V1_big.pt"),
                        2:torch.load(self.mask_path + "V2_big.pt"),
                        3:torch.load(self.mask_path + "V3_big.pt"),
                        4:torch.load(self.mask_path + "V4_big.pt"),
                        5:torch.load(self.mask_path + "early_vis_big.pt"),
                        6:torch.load(self.mask_path + "higher_vis_big.pt")}  
        else:
            self.masks = {0:torch.full((11838,), False),
                        1:torch.load(self.mask_path + "V1.pt"),
                        2:torch.load(self.mask_path + "V2.pt"),
                        3:torch.load(self.mask_path + "V3.pt"),
                        4:torch.load(self.mask_path + "V4.pt"),
                        5:torch.load(self.mask_path + "early_vis.pt"),
                        6:torch.load(self.mask_path + "higher_vis.pt")}  

    def autoencoded_brain_samples(self, subject = 1):
        
        AE = AutoEncoder(config="dualAutoEncoder",
                        inference=True,
                        subject=subject,
                        device="cuda:3")
        
        # Load the test samples
        _, _, x_test, _, _, y_test, test_trials = load_nsd(vector="images", subject=subject, loader=False, average=False, nest=True)
        #print(y_test[0].reshape((425,425,3)).numpy().shape)
        # test = Image.fromarray(y_test[0].reshape((425,425,3)).numpy().astype(np.uint8))
        # test.save("/home/naxos2-raid25/ojeda040/local/ojeda040/Second-Sight/logs/test.png")
        
        
        #x_test_ae = torch.zeros((x_test.shape[0], 11838))
        x_test_ae = torch.zeros((100, x_test.shape[2]))
        
        for i in tqdm(range(100), desc = "Autoencoding samples and averaging" ):
            repetitions = []
            for j in range(3):
                if(torch.count_nonzero(x_test[i,j]) > 0):
                    repetitions.append(x_test[i,j])
                
            x_test_ae[i] = torch.mean(AE.predict(torch.stack(repetitions)),dim=0)
        
        return x_test_ae
    
    def return_all_masks(self):
        
        return self.masks[1], self.masks[2], self.masks[3], self.masks[4], self.masks[5], self.masks[6]

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
        
        # TODO: OLD PREDICTION WAY 
        # print(alexnet_predictions.shape)
        # beta_primes = alexnet_predictions.moveaxis(0, 1).to(self.device)
        
        # if(not unmasked):
        #     beta = beta_sample[brain_mask]
        # else:
        #     beta = beta_sample
                    
        # xDup = beta.repeat(beta_primes.shape[1], 1).moveaxis(0, 1).to(self.device)
        # PeC = PearsonCorrCoef(num_outputs=beta_primes.shape[1]).to(self.device) 
        # print(xDup.shape, beta_primes.shape)
        # scores = PeC(xDup, beta_primes)
        # scores_np = scores.detach().cpu().numpy()
        
        beta_primes = alexnet_predictions.to(self.device)
        
        if(not unmasked):
            beta = beta_sample[brain_mask]
        else:
            beta = beta_sample
                    
                    
        xDup = beta.to(self.device)
        PeC = PearsonCorrCoef(num_outputs=1).to(self.device) 
        print(xDup.shape, beta_primes.shape)
        scores = PeC(xDup, beta_primes)
        scores_np = scores.detach().cpu().numpy()
        
        return scores_np
        
    def calculate_ssim(self, ground_truth_path, reconstruction_path):

        ground_truth   = Image.open(ground_truth_path).resize((425, 425))
        reconstruction = Image.open(reconstruction_path).resize((425, 425))
        
        ground_truth = np.array(ground_truth) / 255.0
        reconstruction = np.array(reconstruction) / 255.0

        ground_truth = rgb2gray(ground_truth)
        reconstruction = rgb2gray(reconstruction)

        return ssim(reconstruction, ground_truth, multichannel=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0)
            
        
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
        
    #two_way_prob is the two way identification experiment between the given image and a random search reconstruction of a different sample with respect to the ground truth
    #clip_pearson is the pearson correlation score between the clips of the two given images
    #Sample type controls which of the types of image to pick a random sample between
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
        #   10 --> Ground Truth
        #   11 --> Decoded VDVAE Only
        #   12--> Decoded CLIP Only
        #   13 --> Decoded CLIP+VDVAE
        #   14 --> Search Reconstruction
        #   15 --> Library Reconstruction
        #   16 --> Ground Truth CLIP
        #   17 --> Ground Truth VDVAE
        #   18 --> Ground Truth CLIP+VDVAE
    def calculate_clip_similarity(self, experiment_name, sample, sampleType=1, subject = 1):
        with torch.no_grad():
            exp_path = "/export/raid1/home/ojeda040/Second-Sight/reconstructions/subject{}/{}/".format(subject, experiment_name)
            
            folders = sorted([int(f.name) for f in os.scandir(exp_path) if f.is_dir() and f.name != 'results'])
            rand_list = [i for i in range(len(folders)) if folders[i] != sample and os.listdir(exp_path + str(folders[i]) + "/")]
            rand_index = random.choice(rand_list)
            sampleTypes = {0: "iter_0.png", 1: "iter_1.png", 2: "iter_2.png", 3: "iter_3.png", 4: "iter_4.png", 
                           5: "iter_5.png", 6: "iter_6.png", 7: "iter_7.png", 8: "iter_8.png", 9: "iter_9.png", 
                           10: "Ground Truth.png", 11: "Decoded VDVAE.png", 12: "Decoded CLIP Only.png", 
                           13: "Decoded CLIP+VDVAE.png", 14: "Search Reconstruction.png", 15: "Library Reconstruction.png",
                           16: "Ground Truth CLIP.png", 17: "Ground Truth VDVAE.png", 18: "Ground Truth CLIP+VDVAE.png"}
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
    
    class batch_generator_external_images(Dataset):

        def __init__(self, data_path ='', prefix='', net_name='clip'):
            self.data_path = data_path
            self.prefix = prefix
            self.net_name = net_name
            
            if self.net_name == 'clip':
                self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
            else:
                self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                self.num_test = 982
            
        def __getitem__(self,idx):
            img = Image.open('{}/{}{}.png'.format(self.data_path,self.prefix,idx))
            img = T.functional.resize(img,(224,224))
            img = T.functional.to_tensor(img).float()
            img = self.normalize(img)
            return img

        def __len__(self):
            return  self.num_test




    def net_metrics():
        
        global feat_list
        feat_list = []
        def fn(module, inputs, outputs):
            feat_list.append(outputs.cpu().numpy())

        net_list = [
            ('inceptionv3','avgpool'),
            ('clip','final'),
            ('alexnet',2),
            ('alexnet',5),
            ('alexnet',7),
            ('efficientnet','avgpool'),
            ('swav','avgpool')
            ]

        device = 1
        net = None
        batchsize=64



        for (net_name,layer) in net_list:
            feat_list = []
            print(net_name,layer)
            dataset = batch_generator_external_images(data_path=images_dir,net_name=net_name,prefix='')
            loader = DataLoader(dataset,batchsize,shuffle=False)
            
            if net_name == 'inceptionv3': # SD Brain uses this
                net = tvmodels.inception_v3(pretrained=True)
                if layer== 'avgpool':
                    net.avgpool.register_forward_hook(fn) 
                elif layer == 'lastconv':
                    net.Mixed_7c.register_forward_hook(fn)
                    
            elif net_name == 'alexnet':
                net = tvmodels.alexnet(pretrained=True)
                if layer==2:
                    net.features[4].register_forward_hook(fn)
                elif layer==5:
                    net.features[11].register_forward_hook(fn)
                elif layer==7:
                    net.classifier[5].register_forward_hook(fn)
                    
            elif net_name == 'clip':
                model, _ = clip.load("ViT-L/14", device='cuda:{}'.format(device))
                net = model.visual
                net = net.to(torch.float32)
                if layer==7:
                    net.transformer.resblocks[7].register_forward_hook(fn)
                elif layer==12:
                    net.transformer.resblocks[12].register_forward_hook(fn)
                elif layer=='final':
                    net.register_forward_hook(fn)
            
            elif net_name == 'efficientnet':
                net = tvmodels.efficientnet_b1(weights=True)
                net.avgpool.register_forward_hook(fn) 
                
            elif net_name == 'swav':
                net = torch.hub.load('facebookresearch/swav:main', 'resnet50')
                net.avgpool.register_forward_hook(fn) 
            net.eval()
            net.cuda(device)    
            
            with torch.no_grad():
                for i,x in enumerate(loader):
                    print(i*batchsize)
                    x = x.cuda(device)
                    _ = net(x)
            if net_name == 'clip':
                if layer == 7 or layer == 12:
                    feat_list = np.concatenate(feat_list,axis=1).transpose((1,0,2))
                else:
                    feat_list = np.concatenate(feat_list)
            else:   
                feat_list = np.concatenate(feat_list)
            
            
            file_name = '{}/{}_{}.npy'.format(feats_dir,net_name,layer)
            np.save(file_name,feat_list)
    
    # Method to grab existing image distributions from a given spot in a search (when it crosses threshold)
    # experiment_title: title of experiment to use
    # sample: sample number to use, individual value of idx list generated in experiment
    # iteration: iteration number to generate a distribution at: 
    #   - provide the iteration number for the iteration AFTER the threshold is crossed
    #   - provide -1 to generate distribution before search is initiated (decoded clip + VDVAE)
    #   - provide -2 to generate distribution before search is initiated (decoded clip only)
    #   - provide -3 to generate distribution before search is initiated (decoded VDVAE only)
    #   - provide last iteration number (5 for searches of 6 iterations) to generate distribution from final state
    # n: number of images to generate in distribution (there will always be at least 10)
    #   - leave it empty to return all available images
    def grab_image_distribution(self, experiment_title, sample, iteration, n=12):
        iter_path = "reconstructions/subject{}/{}/{}/iter_{}/".format(self.subject, experiment_title, sample, iteration)
        batch = int(torch.load(iter_path+"best_im_batch_index.pt"))
        batch_path = iter_path+"batch_{}/".format(batch)
        png_files = [f for f in os.listdir(batch_path) if f.endswith('.png')]
        images = []
        if n == -1:
            n = len(png_files)
        while n>0:
            images.append(Image.open(batch_path+png_files[n-1]))
            n -=1
        return images
    
    def image_indices_paper(self, folder, subject = 1):
        
        # Directory path
        dir_path = "/export/raid1/home/ojeda040/Second-Sight/reconstructions/subject" + str(subject) + "/" + folder + "/"
        
        # Grab the list of files
        files = []
        for path in os.listdir(dir_path):
            files.append(path)
                
        # Get just the image number and then sort the list. 
        indicies = []
        for i in range(len(files)):
            indicies.append(int(re.search(r'\d+', files[i]).group()))
       
        indicies.sort()
        
        return indicies
    
    def image_indices(self, folder, subject = 1):
        
        # Directory path
        dir_path = "/export/raid1/home/ojeda040/Second-Sight/reconstructions/subject" + str(subject) + "/" + folder + "/"
        
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
        
    
    def create_papers_dataframe(self, folder, subject = 1):
        
        # Path to the folder
        directory_path = "/export/raid1/home/ojeda040/Second-Sight/reconstructions/subject" + str(subject) + "/" + folder + "/"
        
        
        # List of image numbers created. 
        idx = self.image_indices_paper(folder, subject = subject)
        
        # create an Empty DataFrame
        # object With column names only
        # Sample Indicator: 
            #   0 --> Ground Truth
            #   1 --> Reconstructed Image
        df = pd.DataFrame(columns = ['ID', 'Regen Number', 'Sample Indicator', 'SSIM', 'Pixel Correlation', 'CLIP Pearson', 'CLIP Two-way'])
        
        # regen count. 
        regen_count = 0
        
        # Dataframe index count. 
        df_row_num = 0
        
        # Append rows to an empty DataFrame
        for i in tqdm(idx, desc="creating dataframe rows"):
            
            # Create the path
            path = directory_path + str(i)
            
            for filename in os.listdir(path): 
                
                # Ground Truth Image
                ground_truth_path = path + '/' + 'Ground Truth.png'
                ground_truth = Image.open(ground_truth_path)
                
                if("Ground Truth.png" not in filename):
                  
                    # Reconstruction path
                    reconstruction_path = path + '/' + filename
                
                    # Reconstruction image
                    reconstruction = Image.open(reconstruction_path)
                
                    # Pix Corr metrics calculation
                    pix_corr = self.calculate_pixel_correlation(ground_truth, reconstruction)
                
                    # SSIM metrics calculation
                    ssim_ground_truth          = self.calculate_ssim(ground_truth_path, reconstruction_path)
                    print(ground_truth_path)
                    print(reconstruction_path)
                    print(ssim_ground_truth)
                
                    # CLIP metrics calculation
                    two_way_prob, clip_pearson, _ = self.calculate_clip_similarity_papaer(folder, i, regen_count, subject = subject)
        
                    # Make data frame row
                    row = pd.DataFrame({'ID' : str(i), 'Regen Number' : str(regen_count), 'Sample Indicator' : "1",
                                        'SSIM' : str(round(ssim_ground_truth, 10)), 'Pixel Correlation' : str(round(pix_corr, 10)),
                                        'CLIP Pearson' : str(round(clip_pearson, 10)), 'CLIP Two-way' : str(round(two_way_prob, 10))}, index=[df_row_num])
                    
                    # Add the row to the dataframe
                    df = pd.concat([df, row])
                    
                    # Iterate the counts
                    regen_count += 1
                    df_row_num += 1
            
            # Make data frame row for ground truth Image
            two_way_prob_ground_truth, clip_pearson_ground_truth, _ = self.calculate_clip_similarity_papaer(folder, i, 5, subject = subject)
            row_ground_truth = pd.DataFrame({'ID' : str(i), 'Sample Indicator' : "0", 'CLIP Pearson' : str(round(clip_pearson_ground_truth, 10)),
                                             'CLIP Two-way' : str(round(two_way_prob_ground_truth, 10))}, index=[df_row_num])
            df_row_num += 1
            df = pd.concat([df, row_ground_truth])
            
                
            # Reset the iter_count and folder_image for the next folder. 
            regen_count = 0 
                                           
                        
        print(df.shape)
        print(df)
        df.to_csv(directory_path + "statistics_df_" + str(len(idx)) +  ".csv")
        
    def create_beta_primes(self, folder, subject = 1):
        
        folder_image_set = []
        
        folders = ["vdvae_distribution", "clip_distribution", "clip+vdvae_distribution"]
        
        directory_path = "/export/raid1/home/ojeda040/Second-Sight/reconstructions/subject{}/{}/".format(str(subject), folder)
        
        existing_path = directory_path + "/0/clip_distribution/0_beta_prime.pt"
        
        if(not os.path.exists(existing_path)):
        
            SCS = StochasticSearch(modelParams=["gnetEncoder", "clipEncoder"], subject=subject, device="cuda:0")
            
            # List of image numbers created. 
            idx = self.image_indices(folder, subject = subject)
            
            # Append rows to an empty DataFrame
            for i in tqdm(idx, desc="creating dataframe rows"):
                
                for folder in folders:
                    
                    # Create the path
                    path = directory_path + str(i) + "/" + folder
                        
                    for filename in os.listdir(path): 
                        
                        reconstruction_image = Image.open(path + "/" + filename)
                        folder_image_set.append(reconstruction_image)
                        
                    beta_primes = SCS.predict(folder_image_set)
                    
                    for j in range(beta_primes.shape[0]):
                        torch.save(beta_primes[j], "{}/{}_beta_prime.pt".format(path, j))
                        
                    folder_image_set = []
                    
    def create_dataframe(self, experiment_name, subject = 1):
        
        # Path to the folder
        log_path       = "/export/raid1/home/ojeda040/Second-Sight/logs/subject" + str(subject) + "/" + experiment_name + "/"
        directory_path = "/export/raid1/home/ojeda040/Second-Sight/reconstructions/subject" + str(subject) + "/" + experiment_name + "/"
        
        # Create betas if needed
        self.create_beta_primes(experiment_name, subject = subject)
        
        # List of image numbers created. 
        idx = self.image_indices(experiment_name, subject = subject)
        
        # Autoencoded avearged brain samples 
        beta_samples = self.autoencoded_brain_samples(subject=subject)
        
        # Grab the necessary brain masks
        brain_mask_V1, brain_mask_V2, brain_mask_V3, brain_mask_V4, brain_mask_early_visual, brain_mask_higher_visual = self.return_all_masks()
        
        # Create an Empty DataFrame
        # Object With column names only
        # Sample Indicator: 
            #   0 --> Ground Truth
            #   1 --> VDVAE Distribution        (Decoded Distribution)
            #   2 --> Clip Distrubituon         (Decoded CLIP Only)
            #   3 --> Clip Distrubituon + VDVAE (Decoded CLIP + VDVAE)
            #   4 --> iter_0
            #   5 --> iter_1
            #   6 --> iter_2
            #   7 --> iter_3
            #   8 --> iter_4
            #   9 --> iter_5
            #  10 --> Search Reconstruction
        df = pd.DataFrame(columns = ['ID', 'Sample Count', 'Batch Number', 'Sample Indicator', 'Strength', 'Brain Correlation V1', 'Brain Correlation V2', 
                                     'Brain Correlation V3', 'Brain Correlation V4', 'Brain Correlation Early Visual', 'Brain Correlation Higher Visual',
                                     'Brain Correlation NSD General', 'SSIM', 'Pixel Correlation', 'CLIP Cosine', 'CLIP Two-way', 'Alexnet 2', 
                                     'Alexnet 5', 'Alexnet 7', 'Inception V3', 'EffNet-B', 'SwAV' ])
        
        # Sample count. 
        sample_count = 0
        
        # Dataframe index count. 
        df_row_num = 0
        
        # Folders in the directory
        folders = {"clip_distribution" : 1, "vdvae_distribution" : 2, "clip+vdvae_distribution" : 3, "iter_0" : 4, "iter_1" : 5 , "iter_2" : 6, "iter_3" : 7, "iter_4" : 8, "iter_5" : 9}
        
        # Append rows to an empty DataFrame
        for i in tqdm(idx, desc="creating dataframe rows"):
            
            for folder, sample_number in folders.items():
                
                # Create the path
                path = directory_path + str(i) + "/" + folder
                
                # Ground Truth Image
                ground_truth_path = directory_path + str(i) + '/' + 'Ground Truth.png'
                ground_truth = Image.open(ground_truth_path)
                
                # Search Reconstruction 
                search_reconstruction_path = directory_path + str(i) + '/' + 'Search Reconstruction.png'
                search_reonstruction = Image.open(search_reconstruction_path)
            
                if("iter" in folder):
                    
                    batch_number = torch.load(path + "/best_batch_index.pt")
                    
                    for filename in os.listdir(path + "/batch_" + str(int(batch_number))): 
                        
                        if(".pt" in filename):
                            continue
                        
                        if(sample_count == 5):
                            break
                    
                        # Reconstruction path
                        reconstruction_path = path + '/batch_' + str(int(batch_number)) + '/' + filename
                        print(filename)
                        print(reconstruction_path)
                        
                        # Reconstruction image
                        reconstruction = Image.open(reconstruction_path)
                        
                        # Pix Corr metrics calculation
                        pix_corr = self.calculate_pixel_correlation(ground_truth, reconstruction)
                        
                        # SSIM metrics calculation
                        ssim    = self.calculate_ssim(ground_truth_path, reconstruction_path)
                        
                        # CLIP metrics calculation
                        two_way_prob, clip_pearson, _ = self.calculate_clip_similarity(experiment_name, i, sample_count, subject = subject)
                        
                        # Calculate the strength at that reconstruction iter image. 
                        strength = 0.92-0.3*(math.pow((sample_count + 1)/ 6, 3))
                        
                        # Calculate brain predictions
                        brain_prediction_nsd_general        = torch.load(path + '/batch_' + str(int(batch_number)) + "/" + str(sample_count) + "_beta_prime.pt")
                        brain_prediction_V1                 = brain_prediction_nsd_general[brain_mask_V1]
                        brain_prediction_V2                 = brain_prediction_nsd_general[brain_mask_V2]
                        brain_prediction_V3                 = brain_prediction_nsd_general[brain_mask_V3]
                        brain_prediction_V4                 = brain_prediction_nsd_general[brain_mask_V4]
                        brain_prediction_early_visual       = brain_prediction_nsd_general[brain_mask_early_visual]
                        brain_prediction_higher_visual      = brain_prediction_nsd_general[brain_mask_higher_visual]
                        
                        # Pearson correlations for each reconstruction region
                        pearson_correlation_V1              = float(self.generate_pearson_correlation(brain_prediction_V1, beta_samples[i], brain_mask_V1, unmasked=False))
                        pearson_correlation_V2              = float(self.generate_pearson_correlation(brain_prediction_V2, beta_samples[i], brain_mask_V2, unmasked=False))
                        pearson_correlation_V3              = float(self.generate_pearson_correlation(brain_prediction_V3, beta_samples[i], brain_mask_V3, unmasked=False))
                        pearson_correlation_V4              = float(self.generate_pearson_correlation(brain_prediction_V4, beta_samples[i], brain_mask_V4, unmasked=False))
                        pearson_correlation_early_visual    = float(self.generate_pearson_correlation(brain_prediction_early_visual, beta_samples[i], brain_mask_early_visual, unmasked=False))
                        pearson_correlation_higher_visual   = float(self.generate_pearson_correlation(brain_prediction_higher_visual, beta_samples[i], brain_mask_higher_visual, unmasked=False))
                        pearson_correlation_nsd_general     = float(self.generate_pearson_correlation(brain_prediction_nsd_general, beta_samples[i], brain_mask_higher_visual, unmasked=True))
                        
                        row = pd.DataFrame({'ID' : str(i), 'Sample Count' : str(sample_count), 'Batch Number' : str(int(batch_number)), 'Sample Indicator' : str(sample_number), 'Strength' : str(round(strength, 10)), 
                                            'Brain Correlation V1' : str(round(pearson_correlation_V1, 10)), 'Brain Correlation V2' : str(round(pearson_correlation_V2, 10)), 'Brain Correlation V3' : str(round(pearson_correlation_V3 , 10)), 
                                            'Brain Correlation V4' : str(round(pearson_correlation_V4, 10)), 'Brain Correlation Early Visual' : str(round(pearson_correlation_early_visual , 10)), 
                                            'Brain Correlation Higher Visual' : str(round(pearson_correlation_higher_visual, 10)), 'Brain Correlation NSD General' : str(round(pearson_correlation_nsd_general, 10)),
                                            'SSIM' : str(round(ssim, 10)), 'Pixel Correlation' : str(round(pix_corr, 10)), 'CLIP Cosine' : str(round(clip_pearson, 10)), 'CLIP Two-way' : str(round(two_way_prob, 10))},  index=[df_row_num])
                                
                        # Add the row to the dataframe
                        df = pd.concat([df, row])
                        
                        # Iterate the counts
                        sample_count += 1
                        df_row_num += 1
                    
                else: 
                    for filename in os.listdir(path): 
                        
                        if("beta" in filename):
                            continue
                        
                        if(sample_count == 5):
                            break
                        
                        # Reconstruction path
                        reconstruction_path = path + '/' + filename
                        print(reconstruction_path)
                        
                        # Reconstruction image
                        reconstruction = Image.open(reconstruction_path)
                        
                        # Pix Corr metrics calculation
                        pix_corr = self.calculate_pixel_correlation(ground_truth, reconstruction)
                        
                        # SSIM metrics calculation
                        ssim    = self.calculate_ssim(ground_truth_path, reconstruction_path)
                        
                        # CLIP metrics calculation
                        two_way_prob, clip_pearson, _ = self.calculate_clip_similarity(experiment_name, i, sample_count, subject = subject)
                        
                        # Calculate the strength at that reconstruction iter image. 
                        strength = 0.92-0.3*(math.pow((sample_count + 1)/ 6, 3))
                        
                        # Calculate brain predictions
                        brain_prediction_nsd_general        = torch.load(path + "/" + str(sample_count) + "_beta_prime.pt")
                        brain_prediction_V1                 = brain_prediction_nsd_general[brain_mask_V1]
                        brain_prediction_V2                 = brain_prediction_nsd_general[brain_mask_V2]
                        brain_prediction_V3                 = brain_prediction_nsd_general[brain_mask_V3]
                        brain_prediction_V4                 = brain_prediction_nsd_general[brain_mask_V4]
                        brain_prediction_early_visual       = brain_prediction_nsd_general[brain_mask_early_visual]
                        brain_prediction_higher_visual      = brain_prediction_nsd_general[brain_mask_higher_visual]
                        
                        # Pearson correlations for each reconstruction region
                        pearson_correlation_V1              = float(self.generate_pearson_correlation(brain_prediction_V1, beta_samples[i], brain_mask_V1, unmasked=False))
                        pearson_correlation_V2              = float(self.generate_pearson_correlation(brain_prediction_V2, beta_samples[i], brain_mask_V2, unmasked=False))
                        pearson_correlation_V3              = float(self.generate_pearson_correlation(brain_prediction_V3, beta_samples[i], brain_mask_V3, unmasked=False))
                        pearson_correlation_V4              = float(self.generate_pearson_correlation(brain_prediction_V4, beta_samples[i], brain_mask_V4, unmasked=False))
                        pearson_correlation_early_visual    = float(self.generate_pearson_correlation(brain_prediction_early_visual, beta_samples[i], brain_mask_early_visual, unmasked=False))
                        pearson_correlation_higher_visual   = float(self.generate_pearson_correlation(brain_prediction_higher_visual, beta_samples[i], brain_mask_higher_visual, unmasked=False))
                        pearson_correlation_nsd_general     = float(self.generate_pearson_correlation(brain_prediction_nsd_general, beta_samples[i], brain_mask_higher_visual, unmasked=True))
                        
                        row = pd.DataFrame({'ID' : str(i), 'Sample Count' : str(sample_count), 'Sample Indicator' : str(sample_number), 'Strength' : str(round(strength, 10)), 'Brain Correlation V1' : str(round(pearson_correlation_V1, 10)),
                                            'Brain Correlation V2' : str(round(pearson_correlation_V2, 10)), 'Brain Correlation V3' : str(round(pearson_correlation_V3, 10)), 
                                            'Brain Correlation V4' : str(round(pearson_correlation_V4, 10)), 'Brain Correlation Early Visual' : str(round(pearson_correlation_early_visual, 10)),
                                            'Brain Correlation Higher Visual' : str(round(pearson_correlation_higher_visual, 10)), 'Brain Correlation NSD General' : str(round(pearson_correlation_nsd_general, 10)),
                                            'SSIM' : str(round(ssim, 10)), 'Pixel Correlation' : str(round(pix_corr, 10)), 'CLIP Cosine' : str(round(clip_pearson, 10)), 'CLIP Two-way' : str(round(two_way_prob, 10))},  index=[df_row_num])
                                
                        # Add the row to the dataframe
                        df = pd.concat([df, row])
                        
                        # Iterate the counts
                        sample_count += 1
                        df_row_num += 1
                    
                # Reset the sample_count for the next folder. 
                sample_count = 0 
                        
            # Make dataframe row for search reconstruction
            pix_corr_search = self.calculate_pixel_correlation(ground_truth, search_reonstruction)
            ssim_search = self.calculate_ssim(ground_truth_path, search_reconstruction_path)
            two_way_prob_search, clip_pearson_search, _ = self.calculate_clip_similarity(experiment_name, i, 14, subject = subject)
            row_search = pd.DataFrame({'ID' : str(i), 'Sample Indicator' : "10", 'SSIM' : str(round(ssim_search, 10)), 'Pixel Correlation' : str(round(pix_corr_search, 10)), 
                                         'CLIP Cosine' : str(round(clip_pearson_search, 10)), 'CLIP Two-way' : str(round(two_way_prob_search, 10))}, index=[df_row_num])
            df_row_num += 1
            df = pd.concat([df, row_search])
            
            # Make data frame row for ground truth Image
            two_way_prob_ground_truth, clip_pearson_ground_truth, _ = self.calculate_clip_similarity(experiment_name, i, 10, subject = subject)
            row_ground_truth = pd.DataFrame({'ID' : str(i), 'Sample Indicator' : "0", 'Strength' : str(round(strength, 10)), 
                                              'CLIP Cosine' : str(round(clip_pearson_ground_truth, 10)), 'CLIP Two-way' : str(round(two_way_prob_ground_truth, 10))}, index=[df_row_num])
            df_row_num += 1
            df = pd.concat([df, row_ground_truth])
                
            # Reset the sample_count for the next folder. 
            sample_count = 0 
            
            print(df.shape)
            print(df)
                                           
                        
        print(df.shape)
        print(df)
        df.to_csv(log_path + "statistics_df_" + str(len(idx)) +  ".csv")
    
    
def main():
    
    SCS = Stochastic_Search_Statistics(big = True, subject = 7, device="cuda:0")
    
    #SCS.generate_brain_predictions() 
    #SCS.calculate_ssim()    
    #SCS.calculate_pixel_correlation()
    
    # GN = GNet8_Encoder(subject=1, device= "cuda:3")
    # SCS.create_dataframe("SCS UC LD 6:100:4 Dual Guided clip_iter 30", GN, subject=1)
    # SCS.create_dataframe("SCS UC LD topn 6:100:4 Dual Guided clip_iter 18", GN, subject=1)
    
    SCS.create_dataframe("Final Run: SCS UC LD 6:100:4 Dual Guided clip_iter", subject=7)
    
    #SCS.create_papers_dataframe("Brain Diffuser", subject = 1)
    
    # SCS.create_beta_primes("Final Run: SCS UC LD 6:100:4 Dual Guided clip_iter", subject = 7)
    
    #SCS.create_dataframe("SCS VD PCA LR 10:250:5 0.4 Exp AE")
    #SCS.create_dataframe("SCS VD PCA LR 10:250:5 0.3 Exp2 AE")
    #SCS.create_dataframe("SCS VD PCA LR 10:250:5 0.6 Exp3 AE NA")
    
    # gt = Image.open("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/reconstructions/SCS VD PCA LR 10:100:4 0.4 Exponential Strength AE/1/Ground Truth.png")
    # garbo = Image.open("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/reconstructions/SCS VD PCA 10:100:4 HS nsd_general AE/0/Search Reconstruction.png")
    # reconstruct = Image.open("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/reconstructions/SCS VD PCA LR 10:100:4 0.4 Exponential Strength AE/1/Search Reconstruction.png")
    # surfer = Image.open("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/surfer.png")
    # print(SCS.calculate_clip_similarity("SCS UC LD 6:100:4 Dual Guided clip_iter 28", 2, sampleType=14, subject = 1))
    # print(SCS.calculate_clip_similarity("SCS UC LD 6:100:4 Dual Guided clip_iter 28", 4, sampleType=14, subject = 1))
    # print(SCS.calculate_clip_similarity("SCS UC LD 6:100:4 Dual Guided clip_iter 28", 10, sampleType=14, subject = 1))
        
if __name__ == "__main__":
    main()