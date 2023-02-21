import os
print(os.environ['CUDA_VISIBLE_DEVICES'])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3"
print(os.environ['CUDA_VISIBLE_DEVICES'])
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
                 device="cuda:2"
                 ):

        # Set the parameters for pytorch model training
        self.vector = vector
        self.encoderModel = encoderHash + "_model_" + self.vector + ".pt"
        os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3"
        self.device = torch.device(device)
        self.sorted_indices = None
        self.pearson_scores = None
        # Initialize the data
        _, _, self.x_vox, self.x_thresh, self.x_test, _, _, self.y_vox, self.y_thresh, self.y_test, _ = load_nsd_vs(vector="c_img_0", loader=False, average=False)

        # Initializes Weights and Biases to keep track of experiments and training runs


        self.E = Encoder(hashNum = encoderHash,
                        vector=self.vector,
                        log=False,
                        device=self.device
                        )
        vector_path = "/export/raid1/home/kneel027/Second-Sight/latent_vectors/" + self.encoderModel + "/"
        if(not os.path.isfile(vector_path + "encoded_voxel_selection.pt")):
            torch.save(self.E.predict(x=self.y_vox), vector_path + "encoded_voxel_selection.pt")
        if(not os.path.isfile(vector_path + "encoded_threshold_selection.pt")):
            torch.save(self.E.predict(x=self.y_thresh), vector_path + "encoded_threshold_selection.pt")
            
        self.x_vox_encoded = torch.load(vector_path + "encoded_voxel_selection.pt")
        self.x_thresh_encoded = torch.load(vector_path + "encoded_threshold_selection.pt")
    
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
        print("creating mask")
        # threshold = round((min(r) * -1), 6)
        # print("threshold: ", threshold)
        mask = torch.zeros(len(self.sorted_indices), dtype=bool)
        # for threshold in [0.0, 0.05, 0.1, 0.2]:
        #     threshmask = np.where(np.array(r) > threshold, mask, False)
        #     print(threshmask.shape)
        #     np.save("/export/raid1/home/kneel027/Second-Sight/masks/" + hashNum + "_" + self.vector + "2voxels_pearson_thresh" + str(threshold), threshmask)
        num_voxels = int(threshold * 11838)
        masked_indices = self.sorted_indices[0:num_voxels]
        mask[masked_indices] = True
        mask = mask.to(self.device)
        print(mask.sum())
        os.makedirs("/export/raid1/home/kneel027/Second-Sight/masks/" + self.encoderModel, exist_ok=True)
        torch.save(mask, "/export/raid1/home/kneel027/Second-Sight/masks/" + self.encoderModel + "/" + str(threshold) + ".pt")
        
        
    def mask_voxels(self, threshold, voxels):
        if(not os.path.isfile("/export/raid1/home/kneel027/Second-Sight/masks/" + self.encoderModel + "/" + str(threshold) + ".pt")):
            self.create_mask(threshold)
        mask = torch.load("/export/raid1/home/kneel027/Second-Sight/masks/" + self.encoderModel + "/" + str(threshold) + ".pt", map_location=self.device)
        # for i in tqdm(range(len(voxels)), desc=(self.vector + " masking")):
        masked_x = voxels[:, mask].to(self.device)
        return masked_x
           
    def get_percentile(self, threshold):
        print(os.environ['CUDA_VISIBLE_DEVICES'])
        print(self.device)
        print("finding percentile for " + str(threshold))
        masked_threshold_x = self.mask_voxels(threshold, self.x_thresh.to(self.device))
        masked_threshold_encoded_x = self.mask_voxels(threshold, self.x_thresh_encoded.to(self.device))
        latent_path = "/export/raid1/home/kneel027/Second-Sight/latent_vectors/" + self.encoderModel + "/"
        mask_path = "/export/raid1/home/kneel027/Second-Sight/masks/" + self.encoderModel + "/"
        mask = torch.load(mask_path + str(threshold) + ".pt", map_location=self.device)
        PeC = PearsonCorrCoef(num_outputs=22735).to(self.device)
        PeC2 = PearsonCorrCoef().to(self.device)
        average_percentile = 0
        for i in tqdm(range(masked_threshold_x.shape[0]), desc="scanning library for threshold " + str(threshold)):
            
            xDup = masked_threshold_x[i].repeat(22735, 1).moveaxis(0, 1).to(self.device)
            scores = torch.zeros((2819141,))
            for batch in tqdm(range(124), desc="batching sample"):
                x_preds = torch.load(latent_path + "/cc3m_batches/" + str(batch) + ".pt", map_location=self.device)
                x_preds_m = x_preds[:, mask]
                x_preds_t = x_preds_m.moveaxis(0, 1)
                
                # Pearson correlation
                scores[22735*batch:22735*batch+22735] = PeC(xDup, x_preds_t).detach()
            scores[-1] = PeC2(masked_threshold_x[i], masked_threshold_encoded_x[i]).detach()
            sorted_scores, sorted_indices = torch.sort(scores, descending=True)
            percentile = 1-float(sorted_indices[-1]/2819141)
            average_percentile += percentile
            tqdm.write(str(percentile))
        final_percentile = average_percentile/masked_threshold_x.shape[0]
        print("final percentile: ", str(final_percentile))
        file = open(mask_path + "results.txt", 'w+')
        file.write(str(threshold) + ": " + str(final_percentile))
        file.close()
        
        
        # #print(r)
        # #r = np.log(r)
        # plt.hist(r, bins=40, log=True)
        # #plt.yscale('log')
        # plt.savefig("/export/raid1/home/kneel027/Second-Sight/charts/" + hashNum + "_" + self.vector + "2voxels_pearson_histogram_log_applied.png")
        