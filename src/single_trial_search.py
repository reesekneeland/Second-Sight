import os
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
import wandb
from tqdm import tqdm
from encoder import Encoder
from decoder import Decoder
from alexnet_encoder import Alexnet
from autoencoder import AutoEncoder
from reconstructor import Reconstructor


def main():
    os.chdir("/export/raid1/home/kneel027/Second-Sight/")
    S = SingleTrialSearch(device="cuda:1",
                          log=True,
                          n_iter=10,
                          n_samples=10)

    S.generateTestSamples(experiment_title="Single Trial Search", idx=[i for i in range(1, 20)], mask=[1])

class SingleTrialSearch():
    def __init__(self, 
                device="cuda:0",
                log=True,
                n_iter=10,
                n_samples=10):
        self.log = log
        self.device = device
        self.n_iter = n_iter
        self.n_samples = n_samples
        self.R = Reconstructor(device="cuda:0")
        self.Alexnet = Alexnet()
        self.nsda = NSDAccess('/home/naxos2-raid25/kneel027/home/surly/raid4/kendrick-data/nsd', '/home/naxos2-raid25/kneel027/home/kneel027/nsd_local')
        mask_path = "/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/masks/"
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
        for i in range(n):
            images.append(self.R.reconstruct(c=c, z=z, strength=strength))
        return images

    #clip is a 5x768 clip vector
    #beta is a 3x11838 tensor of brain data to reconstruct
    #cross validate says whether to cross validate between scans
    #n is the number of samples to generate at each iteration
    #max_iter caps the number of iterations it will perform
    def zSearch(self, clip, beta, n=10, max_iter=10, cross_validate=True, mask=[]):
        best_image = None
        PeC = PearsonCorrCoef(num_outputs=n).to(self.device)
        #still in development
        if(cross_validate):
            images = []
            iter_scores = []
            best_vc = -1
            loss_counter = 0
            z=None
            xDup = []
            for i in range(beta.shape[0]):
                xDup.append(beta[i].repeat(n, 1).moveaxis(0, 1).to(self.device))
            for i in range(beta.shape[0]):
                images_i = []
                iter_scores_i = []
                best_vc_i = -1
                loss_counter = 0
                z=None
                beta_i = beta[i]
                print(beta.shape)
                for cur_iter in tqdm(range(max_iter), desc="iterating over search samples"): 
                    # if(loss_counter < 3):
                    #     break
                    strength = 1.0-(cur_iter/max_iter)
                    # for i in range(beta.shape[0]):
                    if(best_image):
                        im_tensor = self.R.im2tensor(best_image)
                        z = self.R.encode_latents(im_tensor)

                    samples = self.generateNSamples(n, clip, z, strength)
                    beta_primes = self.Alexnet.predict(samples).moveaxis(0, 1).to(self.device)
                    scores = []
                    for j in range(beta.shape[0]):
                        if i != j:
                            score = PeC(xDup[j], beta_primes)
                            scores.append(torch.max(score))
                    cur_vc = torch.mean(scores)
                    if(self.log):
                        wandb.log({'Alexnet Brain encoding pearson correlation': cur_vc})
                    print(cur_vc)
                    images.append(samples[int(torch.argmax(scores))])
                    iter_scores.append(cur_vc)
                    if cur_vc > best_vc or best_vc == -1:
                        best_vc = cur_vc
                        best_image = samples[int(torch.argmax(scores))]
                    else:
                        loss_counter +=1
        else:
            images = []
            iter_scores = []
            best_vc = -1
            loss_counter = 0
            z=None
            
            beta_mask = self.masks[0]
            for i in mask:
                beta_mask = torch.logical_or(beta_mask, self.masks[i])
            print(np.unique(beta_mask, return_counts=True))
            beta = torch.mean(beta, dim=0)
            print(beta.shape)
            beta = beta[beta_mask]
            xDup = beta.repeat(n, 1).moveaxis(0, 1).to(self.device)
            for cur_iter in tqdm(range(max_iter), desc="iterating over search samples"): 
                # if(loss_counter < 3):
                #     break
                strength = 1.0-(cur_iter/max_iter)
                # for i in range(beta.shape[0]):
                if(best_image):
                    im_tensor = self.R.im2tensor(best_image)
                    z = self.R.encode_latents(im_tensor)

                samples = self.generateNSamples(n, clip, z, strength)
                beta_primes = self.Alexnet.predict(samples)
                print(beta_primes.shape)
                beta_primes = beta_primes[:, beta_mask]
                print(beta_primes.shape)
                beta_primes = beta_primes.moveaxis(0, 1).to(self.device)
                
                scores = PeC(xDup, beta_primes)
                cur_vc = float(torch.max(scores))
                if(self.log):
                    wandb.log({'Alexnet Brain encoding pearson correlation': cur_vc})
                print(cur_vc)
                images.append(samples[int(torch.argmax(scores))])
                iter_scores.append(cur_vc)
                if cur_vc > best_vc or best_vc == -1:
                    best_vc = cur_vc
                    best_image = samples[int(torch.argmax(scores))]
                else:
                    loss_counter +=1
            return best_image, images, iter_scores




    def generateTestSamples(self, experiment_title, idx, mask=[]):    

        os.makedirs("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/reconstructions/" + experiment_title + "/", exist_ok=True)
        # Load test data and targets
        _, _, _, _, x_test, _, _, _, _, targets_c_i, test_trials = load_nsd(vector="c_img_0", loader=False, average=False, nest=True)
        _, _, _, _, _, _, _, _, _, targets_c_t, _ = load_nsd(vector="c_text_0", loader=False, average=False, nest=False)
        _, _, _, _, _, _, _, _, _, targets_z, _ = load_nsd(vector="z_img_mixer", loader=False, average=False, nest=False)
        print(x_test.shape, targets_c_i.shape)
        Dc_i = Decoder(hashNum = "528",
                 vector="c_img_0", 
                 inpSize = 11838,
                 log=False, 
                 device=self.device,
                 parallel=False
                 )
    
        Dc_t = Decoder(hashNum = "529",
                    vector="c_text_0", 
                    inpSize = 11838,
                    log=False, 
                    device=self.device,
                    parallel=False
                    )

        AE = AutoEncoder(hashNum = "540",
                 lr=0.0000001,
                 vector="c_img_0", #c_img_0, c_text_0, z_img_mixer
                 encoderHash="536",
                 log=False, 
                 batch_size=750,
                 parallel=False,
                 device="cuda"
                )

        # Generating predicted and target vectors
        outputs_c_i = Dc_i.predict(x=torch.mean(x_test, dim=1))
        outputs_c_t = Dc_t.predict(x=torch.mean(x_test, dim=1))
        
        strength_c = 1
        strength_z = 0
        for i in idx:
            print(i)
            
            c_combined = format_clip(torch.stack([outputs_c_i[i], outputs_c_t[i]]))
            c_combined_target = format_clip(torch.stack([targets_c_i[i], targets_c_t[i]]))
            
            if(self.log):
                wandb.init(
                    # set the wandb project where this run will be logged
                    project="SingleTrialSearch",
                    # track hyperparameters and run metadata
                    config={
                    "experiment": experiment_title,
                    "sample": i,
                    "masks": mask,
                    "n_iter": self.n_iter,
                    "n_samples": self.n_samples
                    }
                )
            z_c_reconstruction, image_list, score_list = self.zSearch(c_combined, x_test[i], n=self.n_samples, max_iter=self.n_iter, cross_validate=False, mask=mask)
            if(self.log):
                wandb.finish()
            # returns a numpy array 
            nsdId = test_trials[i]
            ground_truth_np_array = self.nsda.read_images([nsdId], show=True)
            ground_truth = Image.fromarray(ground_truth_np_array[0])
            ground_truth = ground_truth.resize((512, 512), resample=Image.Resampling.LANCZOS)
            rows = int(self.n_iter/2 + 1)
            columns = 2
            images = [ground_truth, z_c_reconstruction]
            captions = ["Ground Truth", "Search Reconstruction"]
            for j in range(self.n_iter):
                images.append(image_list[j])
                captions.append("Brain Correlation: " + str(round(score_list[j], 4)))
            figure = tileImages(experiment_title + ": " + str(i), images, captions, rows, columns)
            
            
            figure.save('reconstructions/' + experiment_title + '/' + str(i) + '.png')
if __name__ == "__main__":
    main()
