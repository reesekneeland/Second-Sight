import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch.nn as nn
from pearson import PearsonCorrCoef
import h5py
from utils import *
import wandb
import copy
from tqdm import tqdm
from cuml import Ridge

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
        if(self.log):
            wandb.init(
                # set the wandb project where this run will be logged
                project="decoder_ridge",
                # track hyperparameters and run metadata
                config={
                "hash": self.hashNum,
                "architecture": "Ridge Regression",
                "vector": self.vector,
                "dataset": "Z scored"
                }
            )
        self.ridge = Ridge(alpha=1.0, fit_intercept=True, normalize=False,
              solver="cd")
    

    def train(self):
        x_train, x_val, _, _, y_train, y_val, _, _, _, _ = load_nsd(vector=self.vector, 
                                                loader=False,
                                                average=False,
                                                pca=False)
        # Set best loss to negative value so it always gets overwritten
        best_loss = -1.0
        loss_counter = 0
        print(x_train.shape, x_val.shape)
        x_train = torch.concat([x_train, x_val])
        y_train = torch.concat([y_train, y_val])
        print(x_train.shape, y_train.shape)
        
        self.ridge.fit(x_train, y_train)
        params = self.ridge.get_params()
        print(type(params))
        torch.save(params, "models/" + self.hashNum + "_model_" + self.vector + ".pt")
    def predict(self, x, batch=False, batch_size=750):
        self.model.load_state_dict(torch.load("models/" + self.hashNum + "_model_" + self.vector + ".pt", map_location=self.device))
        self.model.eval()
        self.model.to(self.device)
        out = self.model(x.to(self.device))#.to(torch.float16)
        return out
    
    def benchmark(self, average=True):
        _, _, _, x_test, _, _, _, y_test, _, _ = load_nsd(vector=self.vector, 
                                                batch_size=self.batch_size, 
                                                num_workers=self.num_workers, 
                                                loader=False,
                                                average=average,
                                                pca=False)
        # Load our best model into the class to be used for predictions
        self.model.load_state_dict(torch.load("models/" + self.hashNum + "_model_" + self.vector + ".pt"))
        self.model.eval()

        criterion = nn.MSELoss()
        PeC = PearsonCorrCoef(num_outputs=y_test.shape[0]).to(self.device)
        
        x_test = x_test.to(self.device)
        y_test = y_test.to(self.device)
        
        pred_y = self.model(x_test)
        
        pearson = torch.mean(PeC(pred_y.moveaxis(0,1), y_test.moveaxis(0,1)))
        loss = criterion(pred_y, y_test)
        
        print("Vector Correlation: ", float(pearson))
        print("Loss: ", float(loss))
        
if __name__ == "__main__":
    main()
