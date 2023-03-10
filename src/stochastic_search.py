import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
import torch
import numpy as np
from PIL import Image
from nsd_access import NSDAccess
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch.nn as nn
from pycocotools.coco import COCO
from utils import *
import copy
import math
import wandb
from tqdm import tqdm
from image_similarity_measures.quality_metrics import fsim
from encoder import Encoder
from decoder import Decoder
from alexnet_encoder import AlexNetEncoder
from autoencoder import AutoEncoder
from reconstructor import Reconstructor



def main():
    # os.chdir("/export/raid1/home/kneel027/Second-Sight/")
    S0 = StochasticSearch(device="cuda:0",
                          log=True,
                          n_iter=10,
                          n_samples=100,
                          n_branches=4)
    # S1 = StochasticSearch(device="cuda:0",
    #                       log=False,
    #                       n_iter=12,
    #                       n_samples=500,
    #                       n_branches=10)
    # S2 = StochasticSearch(device="cuda:0",
    #                       log=True,
    #                       n_iter=20,
    #                       n_samples=60,
    #                       n_branches=3)
    # S0.generateTestSamples(experiment_title="SCS 10:100:4 higher strength V1 AE", idx=[i for i in range(0, 10)], mask=[1], ae=True)
    S0.generateTestSamples(experiment_title="SCS 10:100:4 best case AlexNet", idx=[i for i in range(0, 10)], mask=[1,2,3,4,5,6,7], ae=True)
    # S0.generateTestSamples(experiment_title="SCS 10:100:4 higher strength V1234 AE", idx=[i for i in range(0, 10)], mask=[1,2,3,4], ae=True)
    # S1.generateTestSamples(experiment_title="SCS 12:500:10 higher strength V1234567 AE", idx=[i for i in range(0, 10)], mask=[1, 2, 3, 4, 5, 6, 7], ae=True)
    # S2.generateTestSamples(experiment_title="SCS 20:60:3 higher strength V1234567 AE", idx=[i for i in range(0, 10)], mask=[1, 2, 3, 4, 5, 6, 7], ae=True)
    # S2.generateTestSamples(experiment_title="SCS 20:60:3 higher strength V1 AE", idx=[i for i in range(0, 10)], mask=[1], ae=True)
    # S2.generateTestSamples(experiment_title="SCS 20:60:3 higher strength V1234 AE", idx=[i for i in range(0, 10)], mask=[1, 2, 3, 4], ae=True)

class StochasticSearch():
    def __init__(self, 
                device="cuda:0",
                log=True,
                n_iter=10,
                n_samples=10,
                n_branches=1):
        self.log = log
        self.device = device
        self.n_iter = n_iter
        self.n_samples = n_samples
        self.n_branches = n_branches
        self.R = Reconstructor(device="cuda:0")
        self.Alexnet = AlexNetEncoder()
        self.nsda = NSDAccess('/export/raid1/home/surly/raid4/kendrick-data/nsd', '/export/raid1/home/kneel027/nsd_local')
        mask_path = "/export/raid1/home/kneel027/Second-Sight/masks/"
        self.masks = {0:torch.full((11838,), False),
                      1:torch.load(mask_path + "V1.pt"),
                      2:torch.load(mask_path + "V2.pt"),
                      3:torch.load(mask_path + "V3.pt"),
                      4:torch.load(mask_path + "V4.pt"),
                      5:torch.load(mask_path + "V5.pt"),
                      6:torch.load(mask_path + "V6.pt"),
                      7:torch.load(mask_path + "V7.pt")}

    def generateNSamples(self, n, c, z=None, strength=1):
        images = []
        for i in tqdm(range(n), desc="Generating samples"):
            images.append(self.R.reconstruct(c=c, z=z, strength=strength))
        return images

    #clip is a 5x768 clip vector
    #beta is a 3x11838 tensor of brain data to reconstruct
    #cross validate says whether to cross validate between scans
    #n is the number of samples to generate at each iteration
    #max_iter caps the number of iterations it will perform
    def zSearch(self, clip, beta, n=10, max_iter=10, n_branches=1, mask=[]):
        z, best_image = None, None
        iter_images = [None] * n_branches
        best_image
        images, iter_scores, var_scores = [], [], []
        best_vector_corrrelation, best_var = -1, -1
        loss_counter = 0
        
        #Conglomerate masks
        if(len(mask)>0):
            beta_mask = self.masks[0]
            for i in mask:
                beta_mask = torch.logical_or(beta_mask, self.masks[i])
            beta = beta[beta_mask]
        for cur_iter in tqdm(range(max_iter), desc="search iterations"):
            # if(loss_counter > 3):
            #     break
            strength = 1.0-0.5*(cur_iter/max_iter)
            tqdm.write("Strength: " + str(strength) + ", N: " + str(n))
            
            samples = []
            for i in range(n_branches):
                if(iter_images[i]):
                    im_tensor = self.R.im2tensor(iter_images[i])
                    z = self.R.encode_latents(im_tensor)
                n_i = max(10, int(n/n_branches*strength))
                samples += self.generateNSamples(n_i, clip, z, strength)

            beta_primes = self.Alexnet.predict(samples, mask)
            # beta_primes = beta_primes[:, beta_mask]
            
            beta_primes = beta_primes.moveaxis(0, 1).to(self.device)
            
            xDup = beta.repeat(beta_primes.shape[1], 1).moveaxis(0, 1).to(self.device)
            PeC = PearsonCorrCoef(num_outputs=beta_primes.shape[1]).to(self.device) 
            print(xDup.shape, beta_primes.shape)
            scores = PeC(xDup, beta_primes)
            cur_var = float(torch.var(scores))
            topn_pearson = torch.topk(scores, n_branches)
            cur_vector_corrrelation = float(torch.max(scores))
            if(self.log):
                wandb.log({'Alexnet Brain encoding pearson correlation': cur_vector_corrrelation, 'score variance': cur_var})
            tqdm.write("VC: " + str(cur_vector_corrrelation) + ", Var: " + str(cur_var))
            images.append(samples[int(torch.argmax(scores))])
            iter_scores.append(cur_vector_corrrelation)
            var_scores.append(cur_var)
            for i in range(n_branches):
                iter_images[i] = samples[int(topn_pearson.indices[i])]
            if cur_vector_corrrelation > best_vector_corrrelation or best_vector_corrrelation == -1:
                best_vector_corrrelation = cur_vector_corrrelation
                best_image = samples[int(torch.argmax(scores))]
            # if cur_var < best_var or best_var == -1:
            #     best_var = cur_var
            # else:
            #     loss_counter +=1
            #     tqdm.write("loss counter: " + str(loss_counter))
        return best_image, images, iter_scores, var_scores




    def generateTestSamples(self, experiment_title, idx, mask=[], ae=False):    

        os.makedirs("reconstructions/" + experiment_title + "/", exist_ok=True)
        # Load data and targets
        _, _, x_param, x_test, _, _, targets_c_i, _, param_trials, test_trials = load_nsd(vector="c_img_0", loader=False, average=True)
        # _, _, _, _, _, _, targets_c_t, _, _, _ = load_nsd(vector="c_text_0", loader=False, average=True)
        gt_images = []
        for i in idx:
            nsdId = param_trials[i]
            ground_truth_np_array = self.nsda.read_images([nsdId], show=True)
            ground_truth = Image.fromarray(ground_truth_np_array[0])
            ground_truth = ground_truth.resize((512, 512), resample=Image.Resampling.LANCZOS)
            gt_images.append(ground_truth)
        Dc_i = Decoder(hashNum = "604",
                 vector="c_img_0", 
                 log=False, 
                 device=self.device
                 )
    
        Dc_t = Decoder(hashNum = "606",
                    vector="c_text_0", 
                    log=False, 
                    device=self.device
                    )

        AE = AutoEncoder(hashNum = "582",
                 lr=0.0000001,
                 vector="alexnet_encoder_sub1", #c_img_0, c_text_0, z_img_mixer
                 encoderHash="579",
                 log=False, 
                 batch_size=750,
                 device="cuda"
                )

        # Generating predicted and target vectors
        outputs_c_i = Dc_i.predict(x=x_param)
        outputs_c_t = Dc_t.predict(x=x_param)
        if(ae):
            x_param = AE.predict(x_param)
        
        for i in idx:
            c_combined = format_clip(torch.stack([outputs_c_i[i], outputs_c_t[i]]))
            c_combined_target = format_clip(torch.stack([targets_c_i[i], targets_c_t[i]]))
            
            if(self.log):
                wandb.init(
                    # set the wandb project where this run will be logged
                    project="StochasticSearch",
                    # track hyperparameters and run metadata
                    config={
                    "experiment": experiment_title,
                    "sample": i,
                    "masks": mask,
                    "n_iter": self.n_iter,
                    "n_samples": self.n_samples
                    }
                )
            z_c_reconstruction, image_list, score_list, var_list = self.zSearch(c_combined, x_param[i], n=self.n_samples, max_iter=self.n_iter, n_branches=self.n_branches, mask=mask)
            
            # returns a numpy array 
            nsdId = param_trials[i]
            ground_truth_np_array = self.nsda.read_images([nsdId], show=True)
            ground_truth = Image.fromarray(ground_truth_np_array[0])
            ground_truth = ground_truth.resize((512, 512), resample=Image.Resampling.LANCZOS)
            rows = int(math.ceil(len(image_list)/2 + 1))
            columns = 2
            images = [ground_truth, z_c_reconstruction]
            captions = ["Ground Truth", "Search Reconstruction"]
            for j in range(len(image_list)):
                images.append(image_list[j])
                captions.append("BC: " + str(round(score_list[j], 3)) + " VAR: " + str(round(var_list[j], 3)))
            print(len(images), len(captions), rows, columns)
            figure = tileImages(experiment_title + ": " + str(i), images, captions, rows, columns)
            if(self.log):
                wandb.finish()
            
            figure.save('reconstructions/' + experiment_title + '/' + str(i) + '.png')
if __name__ == "__main__":
    main()
