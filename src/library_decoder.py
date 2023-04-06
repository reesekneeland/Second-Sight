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
        elif(vector == "c_img_uc"):
            self.datasize = 1024
        elif(vector == "c_text_uc"):
            self.datasize = 78848
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
            elif param == "c_img_uc":
                self.AEModels.append(AutoEncoder(hashNum = "743",
                                        vector="c_img_uc", #c_img_0, c_text_0, z_img_mixer
                                        encoderHash="738",
                                        log=False, 
                                        batch_size=750,
                                        device=self.device))
                self.EncModels.append("738_model_c_img_uc.pt")
            elif param == "c_text_uc":
                self.AEModels.append(AutoEncoder(hashNum = "742",
                                                vector="c_text_uc", #c_img_0, c_text_0, z_img_mixer
                                                encoderHash="739",
                                                log=False, 
                                                device=self.device))
                self.EncModels.append("739_model_c_text_uc.pt")
                
        
        
    def predictVector_coco(self, x, average=True):
        x_preds = []
        for model in self.EncModels:
            modelPreds = torch.load(self.latent_path + model + "/coco_brain_preds.pt", map_location=self.device)
            x_preds.append(modelPreds[:, self.mask])
       
        y_full = torch.load(prep_path + self.vector + "/vector_73k.pt").reshape(73000, self.datasize)
        y = prune_vector(y_full)[0:63000]
        # print(x_preds.shape)
        if(average):
            x = x[:, :, self.mask]
        else:
            x = x[:, 0, self.mask]

        out = torch.zeros((x.shape[0], 500, self.datasize))
        ret_scores = torch.zeros((x.shape[0], 500))
        
        PeC = PearsonCorrCoef(num_outputs=21000).to(self.device)
        average_pearson = 0
        
        for sample in tqdm(range(x.shape[0]), desc="scanning library for " + self.vector):
            scores = torch.zeros((y.shape[0],))
            div = 0
            for mId, model in enumerate(self.EncModels):
                # print("Model: ", model)
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
                # print("Best Score: ", torch.max(scores/div))
            scores /= div
                    
                # Calculating the Average Pearson Across Samples
            top5_pearson = torch.topk(scores, 500)
            average_pearson += torch.mean(top5_pearson.values.detach()) 
            for rank, index in enumerate(top5_pearson.indices):
                out[sample, rank] = y[index]
                ret_scores[sample] = top5_pearson.values.detach()
            
        # torch.save(out, latent_path + encModel + "/" + vector + "_coco_library_preds.pt")
        print("Average Pearson Across Samples: ", (average_pearson / x.shape[0]) ) 
        return out, ret_scores
    
    def benchmark(self, average=True):
        LD = LibraryDecoder(vector=self.vector,
                            config=self.config,
                            device=self.device)

        # Load data and targets
        _, _, _, x_test, _, _, _, target, _, _ = load_nsd(vector=self.vector, loader=False, average=False, nest=True)

        
        out, _ = LD.predictVector_coco(x_test, average=average)
        
        criterion = nn.MSELoss()
        
        PeC = PearsonCorrCoef(num_outputs=x_test.shape[0]).to(self.device)
        target = target.to(self.device)
        out = out.to(self.device)
        
        loss = criterion(out[:, 0], target)
        
        pearson_loss = torch.mean(PeC(out[:, 0].moveaxis(0,1).to(self.device), target.moveaxis(0,1).to(self.device)).detach())
        print("Vector Correlation: ", float(pearson_loss))
        print("Loss: ", float(loss))
        vc = []
        l2 = []
        for i in range(500):
            loss = criterion(torch.mean(out[:,0:i], dim=1), target)
            pearson_loss = torch.mean(PeC(torch.mean(out[:,0:i], dim=1).moveaxis(0,1).to(self.device), target.moveaxis(0,1).to(self.device)).detach())
            vc.append(float(pearson_loss))
            l2.append(float(loss))
            print("Vector Correlation Top " + str(i) + ": ", float(pearson_loss))
            print("Loss Top " + str(i) + ":", float(loss))
        np.save("logs/library_decoder_scores/" + self.vector + "_" + "_".join(self.config) + "_PeC.npy", np.array(vc))
        np.save("logs/library_decoder_scores/" + self.vector + "_" + "_".join(self.config) + "_L2.npy", np.array(l2))