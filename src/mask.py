import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3"
import torch
import numpy as np
from nsd_access import NSDAccess
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch.nn as nn
from utils import *
import wandb
from tqdm import tqdm
from encoder import Encoder
from pearson import PearsonCorrCoef, pearson_corrcoef

class Masker():
    def __init__(self, 
                 encoderHash,
                 vector, 
                 device="cuda:0"
                 ):

        # Set the parameters for pytorch model training
        self.vector = vector
        self.encoderModel = encoderHash + "_model_" + self.vector + ".pt"
        self.device = torch.device(device)
        self.sorted_indices = None
        self.pearson_scores = None
        # Initialize the data
        _, _, self.x_vox, self.x_thresh, self.x_test, _, _, self.y_vox, self.y_thresh, self.y_test, _ = load_nsd(vector="c_img_0", loader=False, average=True)

        # Initializes Weights and Biases to keep track of experiments and training runs


        self.E = Encoder(hashNum = encoderHash,
                        vector=self.vector,
                        log=False,
                        device=self.device
                        )
        
        self.latent_path = "/export/raid1/home/kneel027/Second-Sight/latent_vectors/" + self.encoderModel + "/"
        self.mask_path = "/export/raid1/home/kneel027/Second-Sight/masks/" + self.encoderModel + "/avg/"
        if(not os.path.isfile(self.latent_path + "avg_encoded_voxel_selection.pt")):
            torch.save(self.E.predict(x=self.y_vox), self.latent_path + "avg_encoded_voxel_selection.pt")
        if(not os.path.isfile(self.latent_path + "avg_encoded_threshold_selection.pt")):
            torch.save(self.E.predict(x=self.y_thresh), self.latent_path + "avg_encoded_threshold_selection.pt")
            
        self.x_vox_encoded = torch.load(self.latent_path + "avg_encoded_voxel_selection.pt", map_location=self.device)
        self.x_thresh_encoded = torch.load(self.latent_path + "avg_encoded_threshold_selection.pt", map_location=self.device)
        
        subj1 = nsda.stim_descriptions[nsda.stim_descriptions['subject1'] != 0]
        nsdIds = set(subj1['nsdId'].tolist())
        
        x_preds_full = torch.load(self.latent_path + "coco_brain_preds.pt", map_location=self.device)
        self.x_preds = torch.zeros((10500,11838))
        count = 0
        for pred in range(73000):
            if pred not in nsdIds and count <10500:
                self.x_preds[count] = x_preds_full[pred]
                count+=1
        
        self.PeC = PearsonCorrCoef(num_outputs=10500).to(self.device)
        self.PeC2 = PearsonCorrCoef().to(self.device)
    
    def orderVoxels(self):
        PeC = PearsonCorrCoef(num_outputs=11838).to(self.device)
        out = self.x_vox_encoded.to(self.device)
        target = self.x_vox.to(self.device)
        r = PeC(out, target)
        # r = np.array(r)
        self.sorted_indices = torch.stack([x for _, x in sorted(zip(r, torch.arange(11838)), reverse=True)])
        self.pearson_scores = torch.stack([x for x, _ in sorted(zip(r, torch.arange(11838)), reverse=True)])

    def create_mask(self, threshold):
        if(self.sorted_indices is None):
            self.orderVoxels()
        if(threshold==-1):
            print("grabbing negative threshold")
            lowestPearson = -(self.pearson_scores[-1])
            print(lowestPearson)
            num_voxels = next(x for x, val in enumerate(self.pearson_scores)
                                  if val < lowestPearson)
            # threshold = (self.pearson_scores>=lowestPearson).nonzero(as_tuple=True)[1][-1]
            print(num_voxels)
        else:
            num_voxels = int(11838*threshold)
        print("creating mask")
        # threshold = round((min(r) * -1), 6)
        # print("threshold: ", threshold)
        mask = torch.zeros(len(self.sorted_indices), dtype=bool)
        # for threshold in [0.0, 0.05, 0.1, 0.2]:
        #     threshmask = np.where(np.array(r) > threshold, mask, False)
        #     print(threshmask.shape)
        #     np.save("/export/raid1/home/kneel027/Second-Sight/masks/" + hashNum + "_" + self.vector + "2voxels_pearson_thresh" + str(threshold), threshmask)
        
        masked_indices = self.sorted_indices[0:num_voxels]
        mask[masked_indices] = True
        mask = mask.to(self.device)
        print(mask.sum())
        os.makedirs(self.mask_path, exist_ok=True)
        torch.save(mask, self.mask_path + str(threshold) + ".pt")
        
        
    def mask_voxels(self, threshold, voxels):
        if(not os.path.isfile(self.mask_path + str(threshold) + ".pt")):
            self.create_mask(threshold)
        mask = torch.load(self.mask_path + str(threshold) + ".pt", map_location=self.device)
        # for i in tqdm(range(len(voxels)), desc=(self.vector + " masking")):
        masked_x = voxels[:, mask].to(self.device)
        return masked_x
        
    def get_percentile_coco(self, threshold):
        threshold = float(threshold)
        if(not os.path.isfile(self.mask_path + str(threshold) + ".pt")):
            self.create_mask(threshold)
        mask = torch.load(self.mask_path + str(threshold) + ".pt", map_location=self.device)
        masked_threshold_x = self.x_thresh[:, mask].to(self.device)
        masked_threshold_encoded_x = self.x_thresh_encoded[:, mask].to(self.device)
        
        x_preds_m = self.x_preds[:, mask]
        x_preds_t = x_preds_m.moveaxis(0, 1).to(self.device)
        average_percentile = 0
        for i in tqdm(range(masked_threshold_x.shape[0]), desc="scanning library for threshold " + str(threshold)):
            xDup = masked_threshold_x[i].repeat(10500, 1).moveaxis(0, 1).to(self.device)
            scores = torch.zeros((10501,))
            # for batch in range(3):
            # for j in tqdm(range(73000), desc="batching sample"):
                # Pearson correlation
            # print(xDup.shape, masked_threshold_encoded_x[i].shape)
                # x_preds_t_b = x_preds_t[:,21000*batch:21000*batch+21000]
                # scores[21000*batch:21000*batch+21000] = self.PeC(xDup, x_preds_t_b).detach()
            scores = self.PeC(xDup, x_preds_t)
            scores[-1] = self.PeC2(masked_threshold_x[i], masked_threshold_encoded_x[i])
            scores.detach()
            sorted_scores, sorted_indices = torch.sort(scores, descending=True)
            rank = ((sorted_indices==10499).nonzero(as_tuple=True)[0])
            sorted_scores.detach()
            sorted_indices.detach()
            percentile = 1-float(rank/10501)
            average_percentile += percentile
            # tqdm.write(str(percentile))
        final_percentile = average_percentile/masked_threshold_x.shape[0]
        file = open(self.mask_path + "results_coco_inc.txt", 'a+')
        file.write(str(threshold) + ": " + str(final_percentile) + "\n")
        file.close()
        del scores
        del x_preds_t
        del x_preds_m
        # del x_preds_t_b
        torch.cuda.empty_cache()
        
    def make_histogram(self):
        x, y = [], []
        file = open(self.mask_path + "results_coco_inc.txt", 'r')
        for i, line in enumerate(file.readlines()):
            vals = line.split(": ")
            x.append(float(vals[0]))
            y.append(float(vals[1][:-2]))
        print(x[0:5], y[0:5])
        plt.plot(np.array(x), np.array(y))  # Plot the chart
        plt.savefig("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/charts/" + self.encoderModel + "_threshold_plot_avg_inc.png")
        print("best thresh: " + str(x[np.argmax(np.array(y))]))
        # #print(r)
        # #r = np.log(r)
        # plt.hist(r, bins=40, log=True)
        # #plt.yscale('log')
        # plt.savefig("/export/raid1/home/kneel027/Second-Sight/charts/" + hashNum + "_" + self.vector + "2voxels_pearson_histogram_log_applied.png")
        