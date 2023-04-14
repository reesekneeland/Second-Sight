import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch.nn as nn
from torchmetrics import PearsonCorrCoef
import h5py
from utils import *
# import wandb
import copy
from tqdm import tqdm
from cuml import Ridge
import cupy as cp
import cudf

# Main Class    
def main():
    hashNum = update_hash()
    # hashNum = "702"
    D = Decoder_Ridge(hashNum = hashNum,
                 vector="c_img_vd",
                 log=True, 
                 device="cuda:0"
                )
    
    D.train()
    
    # D.benchmark(average=False)
    # D.benchmark(average=True)
    
class Decoder_Ridge():
    def __init__(self, 
                 hashNum,
                 vector, 
                 log, 
                 device="cuda"
                 ):

        # Set the parameters for pytorch model training
        self.hashNum = hashNum
        self.vector = vector
        self.device = torch.device(device)
        self.log = log
        
        # Initializes Weights and Biases to keep track of experiments and training runs
        # if(self.log):
        #     wandb.init(
        #         # set the wandb project where this run will be logged
        #         project="decoder_ridge",
        #         # track hyperparameters and run metadata
        #         config={
        #         "hash": self.hashNum,
        #         "architecture": "Ridge Regression",
        #         "vector": self.vector,
        #         "dataset": "Z scored"
        #         }
        #     )
        self.ridge = Ridge(alpha=1.0, fit_intercept=True, normalize=False,
              solver="cd")
    

    def train(self):
        x_train, x_val, _, _, y_train, y_val, _, _, _, _ = load_nsd(vector=self.vector, 
                                                loader=False,
                                                average=False,
                                                pca=True)
        X = cudf.DataFrame()
        x_train = cp.array(torch.concat([x_train, x_val]).numpy())
        y_train = cp.array(torch.concat([y_train, y_val]).numpy())
        print(x_train.shape, y_train.shape)
        self.ridge.fit(x_train, y_train)
        params = self.ridge.get_params()
        print(type(params))
        cp.save(params, "models/" + self.hashNum + "_model_" + self.vector + ".npy")
        
    def predict(self, x, batch=False, batch_size=750):
        self.ridge.set_params(cp.load("models/" + self.hashNum + "_model_" + self.vector + ".npy"))
        out = self.ridge.predict(x)
        return out
    
    def benchmark(self, average=True):
        _, _, _, x_test, _, _, _, y_test_pca, _, _ = load_nsd(vector=self.vector, 
                                                loader=False,
                                                average=average,
                                                pca=True)
        _, _, _, _, _, _, _, y_test, _, _ = load_nsd(vector=self.vector, 
                                                loader=False,
                                                average=average,
                                                pca=False)
        # Load our best model into the class to be used for predictions
        self.ridge.set_params(cp.load("models/" + self.hashNum + "_model_" + self.vector + ".npy"))
        
        criterion = nn.MSELoss()
        PeC = PearsonCorrCoef(num_outputs=y_test.shape[0]).to(self.device)
        
        x_test = cp.array(x_test.numpy())
        y_test = y_test.to(self.device)
        y_test_pca = y_test_pca.to(self.device)
        
        pred_y_pca = torch.from_numpy(cp.asnumpy(self.ridge.predict(x_test))).to(self.device)
        
        loss_pca = criterion(pred_y_pca, y_test_pca.to(self.device))
              
        pearson_pca =torch.mean(PeC(pred_y_pca.moveaxis(0,1), y_test_pca.moveaxis(0,1)))
        
        pred_y = torch.from_numpy(self.pca.inverse_transform(pred_y_pca.to(torch.float64).detach().cpu().numpy())).to(self.device, torch.float32)
        
        pearson = torch.mean(PeC(pred_y.moveaxis(0,1), y_test.moveaxis(0,1)))
        loss = criterion(pred_y, y_test)
        
        print("Vector Correlation_PCA: ", float(pearson_pca))
        print("Vector Correlation: ", float(pearson))
        print("Loss_PCA: ", float(loss_pca))
        print("Loss: ", float(loss))
        
if __name__ == "__main__":
    main()
