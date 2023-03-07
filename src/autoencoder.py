# Only GPU's in use
import os
import torch
from torch.optim import Adam
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch.nn as nn
from torchmetrics.functional import pearson_corrcoef
from pycocotools.coco import COCO
import h5py
from utils import *
import wandb
import copy
from tqdm import tqdm
from pearson import PearsonCorrCoef

    
class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(11838, 5000),
            torch.nn.ReLU(),
            # torch.nn.Linear(5000, 2500),
            # torch.nn.ReLU(),
            torch.nn.Linear(5000, 1000),
            torch.nn.ReLU(),
            torch.nn.Linear(1000, 500)
        )
         
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(500, 1000),
            torch.nn.ReLU(),
            # torch.nn.Linear(1000, 2500),
            # torch.nn.ReLU(),
            torch.nn.Linear(1000, 5000),
            torch.nn.ReLU(),
            torch.nn.Linear(5000, 11838)
        )
 
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Main Class    
class AutoEncoder():
    def __init__(self, 
                 hashNum,
                 vector, 
                 log=False, 
                 encoderHash=None,
                 lr=0.00001,
                 batch_size=750,
                 parallel=True,
                 device="cuda",
                 num_workers=16,
                 epochs=200
                 ):

        # Set the parameters for pytorch model training
        self.hashNum     = hashNum
        self.vector      = vector
        if(encoderHash):
            self.encoderModel = encoderHash + "_model_" + self.vector + ".pt"
        
        self.device      = torch.device(device)
        self.lr          = lr
        self.batch_size  = batch_size
        self.num_epochs  = epochs
        self.num_workers = num_workers
        self.log         = log
        self.parallel    = parallel
    
        # Initialize the Pytorch model class
        self.model = MLP()
        
        # Configure multi-gpu training
        if(self.parallel):
            self.model = nn.DataParallel(self.model)
        
        # Send model to Pytorch Device 
        self.model.to(self.device)
        
        # Initialize the data loaders
        self.trainLoader, self.valLoader, self.testLoader = None, None, None
        
        # Initializes Weights and Biases to keep track of experiments and training runs
        if(self.log):
            wandb.init(
                # set the wandb project where this run will be logged
                project="Autoencoder",
                # track hyperparameters and run metadata
                config={
                "hash": self.hashNum,
                "architecture": "Autoencoder",
                "encoder Hash": encoderHash,
                "vector": self.vector,
                "dataset": "Whole region visual cortex",
                "epochs": self.num_epochs,
                "learning_rate": self.lr,
                "batch_size:": self.batch_size,
                "num_workers": self.num_workers
                }
            )
    

    def train(self):
        self.trainLoader, self.valLoader, _, _, = load_nsd(vector=self.vector, 
                                                                batch_size=self.batch_size, 
                                                                num_workers=self.num_workers, 
                                                                ae=True,
                                                                encoderModel=self.encoderModel,
                                                                average=False,
                                                                old_norm=True)
        # Set best loss to negative value so it always gets overwritten
        best_loss = -1.0
        loss_counter = 0
        
        # Configure the pytorch objects, loss function (criterion)
        criterion = nn.MSELoss(reduction="sum")
        
        # Import gradients to wandb to track loss gradients
        if(self.log):
            wandb.watch(self.model, criterion, log="all")
        
        # Set the optimizer to Adam
        optimizer = Adam(self.model.parameters(), lr = self.lr)
        
        # Begin training, iterates through epochs, and through the whole dataset for every epoch
        for epoch in tqdm(range(self.num_epochs), desc="epochs"):
            
            # Entering training stage
            self.model.train()
            
            # Keep track of running loss for this training epoch
            running_loss = 0.0
            for i, data in enumerate(self.trainLoader):
                
                # Load the data out of our dataloader by grabbing the next chunk
                # The chunk is the same size as the batch size
                # x_data = Beta (Brain data)
                # y_data = Beta prime (Vector conditioned encoding)
                x_data, y_data = data
                
                # Moving the tensors to the GPU
                x_data = x_data.to(self.device)
                y_data = y_data.to(self.device)
                
                # Zero gradients in the optimizer
                optimizer.zero_grad()
                
                # Forward pass: Compute predicted y by passing x to the model
                with torch.set_grad_enabled(True):
                    
                    # Train the x data in the model to get the predicted y value. 
                    pred_y = self.model(x_data)
                    
                    # Compute the loss between the predicted y and the y data. 
                    loss = criterion(pred_y, y_data)
                    
                    # Perform weight updating
                    loss.backward()
                    optimizer.step()

                # tqdm.write('train loss: %.3f' %(loss.item()))
                # Add up the loss for this training round
                running_loss += loss.item()
                
            train_loss = running_loss/len(self.trainLoader)
            tqdm.write('[%d] train loss: %.8f' %
                (epoch + 1, train_loss ))
                #     # wandb.log({'epoch': epoch+1, 'loss': running_loss/(50 * self.batch_size)})
                
            # Entering validation stage
            # Set model to evaluation mode
            self.model.eval()
            running_test_loss = 0.0
            for i, data in enumerate(self.valLoader):
                
                # Loading in the test data
                # x_data = Beta (Brain data)
                # y_data = Beta prime (Vector conditioned encoding)
                x_data, y_data = data
                x_data = x_data.to(self.device)
                y_data = y_data.to(self.device)
                # Generating predictions based on the current model
                pred_y = self.model(x_data).to(self.device)
                
                # Compute the test loss 
                loss = criterion(pred_y, y_data)

                running_test_loss += loss.item()
                
            test_loss = running_test_loss / len(self.valLoader)
                
            # Printing and logging loss so we can keep track of progress
            tqdm.write('[%d] test loss: %.8f' %
                        (epoch + 1, test_loss))
            if(self.log):
                wandb.log({'test_loss': test_loss})
                    
            # Check if we need to save the model
            # Early stopping
            if(best_loss == -1.0 or test_loss < best_loss):
                best_loss = test_loss
                torch.save(best_loss, "best_loss_" + self.vector + ".pt")
                if(self.parallel):
                    torch.save(self.model.module.state_dict(), "/export/raid1/home/kneel027/Second-Sight/models/" + self.hashNum + "_model_" + self.vector + ".pt")
                else:
                    torch.save(self.model.state_dict(), "/export/raid1/home/kneel027/Second-Sight/models/" + self.hashNum + "_model_" + self.vector + ".pt")
                loss_counter = 0
            else:
                loss_counter += 1
                tqdm.write("loss counter: " + str(loss_counter))
                if(loss_counter >= 3):
                    break
                
        # Load our best model into the class to be used for predictions
        if(self.parallel):
            self.model.module.load_state_dict(torch.load("/export/raid1/home/kneel027/Second-Sight/models/" + self.hashNum + "_model_" + self.vector + ".pt", map_location=self.device))
        else:
            self.model.load_state_dict(torch.load("/export/raid1/home/kneel027/Second-Sight/models/" + self.hashNum + "_model_" + self.vector + ".pt", map_location=self.device))
            
    def predict(self, x, batch=False, batch_size=750):
        
        self.model.load_state_dict(torch.load("/export/raid1/home/kneel027/Second-Sight/models/" + self.hashNum + "_model_" + self.vector + ".pt", map_location=self.device))
        self.model.eval()
        self.model.to(self.device)
        out = self.model(x.to(self.device)).to(self.device)
            
        return out
                
    
    def benchmark(self, encodedPass=False):
        _, _, _, self.testLoader = load_nsd(vector=self.vector, 
                                                batch_size=self.batch_size, 
                                                num_workers=self.num_workers, 
                                                ae=True,
                                                encoderModel=self.encoderModel,
                                                average=True,
                                                old_norm=False)
        datasize = len(self.testLoader.dataset)
        out = torch.zeros((datasize,11838))
        target = torch.zeros((datasize, 11838))
        modelId = self.hashNum + "_model_" + self.vector + ".pt"
        print(modelId)
        self.model.load_state_dict(torch.load("/export/raid1/home/kneel027/Second-Sight/models/" + modelId, map_location=self.device))
        self.model.eval()
        self.model.to(self.device)

        loss = 0
        pearson_loss = 0
        
        criterion = nn.MSELoss()
        
        for index, data in enumerate(self.testLoader):
            
            x_test, y_test = data
            PeC = PearsonCorrCoef(num_outputs=x_test.shape[0]).to(self.device)
            y_test = y_test.to(self.device)
            x_test = x_test.to(self.device)
            # Generating predictions based on the current model
            if(encodedPass):
                pred_y = self.model(y_test).to(self.device)
            else:
                pred_y = self.model(x_test).to(self.device)
            
            
            out[index*self.batch_size:index*self.batch_size+pred_y.shape[0]] = pred_y
            target[index*self.batch_size:index*self.batch_size+pred_y.shape[0]] = y_test
            loss += criterion(pred_y, y_test)
            pred_y = pred_y.moveaxis(0,1)
            y_test = y_test.moveaxis(0,1)
            pearson_loss += torch.mean(PeC(pred_y, y_test))
            #print(pearson_corrcoef(out[index], target[index]))
            
            
        loss = loss / len(self.testLoader)
        
        # Vector correlation for that trial row wise
        pearson_loss = pearson_loss / len(self.testLoader)
        
        out = out.detach()
        PeC = PearsonCorrCoef()
        r = []
        for p in range(out.shape[1]):
            
            # Correlation across voxels for a sample (Taking a column)
            r.append(PeC(out[:,p], target[:,p]))
        r = np.array(r)
        
        print("Vector Correlation: ", float(pearson_loss))
        print("Mean Pearson: ", np.mean(r))
        print("Loss: ", float(loss))
        plt.hist(r, bins=40, log=True)
        plt.savefig("/export/raid1/home/kneel027/Second-Sight/charts/" + self.hashNum + "_" + self.vector + "_pearson_histogram_autoencoder.png")
        

    