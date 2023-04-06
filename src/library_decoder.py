import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
from utils import *
from tqdm import tqdm
from reconstructor import Reconstructor
from autoencoder  import AutoEncoder
from pearson import PearsonCorrCoef


# Config options:
#   - AlexNet
#   - c_img_vd
#   - c_text_vd
class LibraryDecoder():
    def __init__(self, 
                 vector="images",
                 config=["AlexNet"],
                 device="cuda",
                 mask=torch.full((11838,), True)
                 ):

        self.AEModels = []
        self.EncModels = []
        self.config = config
        self.vector = vector
        self.device = device
        self.mask = mask
        if(vector == "c_img_0" or vector == "c_text_0"):
            self.datasize = 768
        elif(vector == "z_img_mixer"):
            self.datasize = 16384
        elif(vector == "images"):
            self.datasize = 541875
            
        self.prep_path = "/export/raid1/home/kneel027/nsd_local/preprocessed_data/"
        self.latent_path = "latent_vectors/"

        for param in config:
            if param == "AlexNet":
                self.AEModels.append(AutoEncoder(hashNum = "582",
                                                vector="alexnet_encoder_sub1", 
                                                encoderHash="579",
                                                log=False, 
                                                device=self.device))
                self.EncModels.append("alexnet_encoder")
            elif param == "c_img_vd":
                self.AEModels.append(AutoEncoder(hashNum = "724",
                                        vector="c_img_vd", #c_img_0, c_text_0, z_img_mixer
                                        encoderHash="722",
                                        log=False, 
                                        batch_size=750,
                                        device=self.device))
                self.EncModels.append("722_model_c_img_vd.pt")
            elif param == "c_text_vd":
                self.AEModels.append(AutoEncoder(hashNum = "727",
                                                vector="c_text_vd", #c_img_0, c_text_0, z_img_mixer
                                                encoderHash="660",
                                                log=False, 
                                                device=self.device))
                self.EncModels.append("660_model_c_text_vd.pt")
                
        
        
    def predictVector_coco(self, x, average=True):
        x_preds = []
        for model in self.EncModels:
            modelPreds = torch.load(self.latent_path + model + "/coco_brain_preds.pt", map_location=self.device)
            x_preds.append(modelPreds[:, self.mask])
       
        y_full = torch.load(prep_path + self.vector + "/vector_73k.pt").reshape(73000, self.datasize)
        y = prune_vector(y_full)
        # print(x_preds.shape)
        if(average):
            x = x[:, :, self.mask]
        else:
            x = x[:, 0, self.mask]

        out = torch.zeros((x.shape[0], 5, self.datasize))
        ret_scores = torch.zeros((x.shape[0], 5))
        
        PeC = PearsonCorrCoef(num_outputs=21000).to(self.device)
        average_pearson = 0
        div = 0
        for sample in tqdm(range(x.shape[0]), desc="scanning library for " + self.vector):
            scores = torch.zeros((63000,))
            for mId, model in enumerate(self.EncModels):
                for rep in range(x.shape[1]):
                    x_rep = x[sample, rep]
                    if(torch.count_nonzero(x_rep) > 0):
                        x_ae = self.AEModels[mId].predict(x_rep)
                        xDup = x_ae.repeat(21000, 1).moveaxis(0, 1).to(self.device)
                        for coco_batch in range(3):
                            x_preds_t = []
                            x_preds_batch = x_preds[mId][21000*coco_batch:21000*coco_batch+21000]
                            x_preds_t = x_preds_batch.moveaxis(0, 1).to(self.device)
                            modelScore = PeC(xDup, x_preds_t).cpu().detach()
                            scores[21000*coco_batch:21000*coco_batch+21000] += modelScore.detach()
                    div +=1
            scores /= div
                    
                # Calculating the Average Pearson Across Samples
            top5_pearson = torch.topk(scores, 5)
            average_pearson += torch.mean(top5_pearson.values.detach()) 
            for rank, index in enumerate(top5_pearson.indices):
                out[sample, rank] = y[index]
                ret_scores[sample] = top5_pearson.values.detach()
            
        # torch.save(out, latent_path + encModel + "/" + vector + "_coco_library_preds.pt")
        print("Average Pearson Across Samples: ", (average_pearson / x.shape[0]) ) 
        return out, ret_scores
    
    def benchmark_library(self, idx, device="cuda:0", average=True):
        _, _, _, x_test, _, _, _, target, _, test_trials = load_nsd(vector=self.vector, loader=False, average=False, nest=True)

        out, scores = self.predictVector_coco(x=x_test[idx], average=average)
        out = out[:,0]
        criterion = nn.MSELoss()
        
        PeC = PearsonCorrCoef(num_outputs=x_test.shape[0]).to(device)
        target = target.to(device)
        out = out.to(device)

        loss = criterion(out, target)
        out = out.moveaxis(0,1).to(device)
        target = target.moveaxis(0,1).to(device)
        pearson_loss = torch.mean(PeC(out, target).detach())

        print("Vector Correlation: ", float(pearson_loss))
        print("Loss: ", float(loss))
        print("Top 5 Average Brain Score: ", float(torch.mean(scores)))
        print("Top 1 Brain Score: ", float(torch.mean(scores[:,0])))