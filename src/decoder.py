# Only GPU's in use
import os
import torch
from torch.autograd import Variable
from torch.optim import Adam
import numpy as np
import glob
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
# import bitsandbytes as bnb

# Pytorch model class for Linear regression layer Neural Network
class MLP(torch.nn.Module):
    def __init__(self, vector):
        super(MLP, self).__init__()
        self.vector=vector
        if(vector == "c_img_vd"):
            self.linear = nn.Linear(11838, 1000)
            # self.linear2 = nn.Linear(15000, 10000)
            self.outlayer = nn.Linear(1000, 197376)
        elif(vector == "c_text_vd"):
            self.linear = nn.Linear(11838, 8000)
            self.outlayer = nn.Linear(8000, 59136)
        self.relu = nn.Sigmoid()
        # self.half()
    def forward(self, x):
        if(self.vector == "c_img_vd" or self.vector=="c_text_vd"):
            y_pred = self.relu(self.linear(x))
            # y_pred = self.relu(self.linear2(y_pred))
            y_pred = self.outlayer(y_pred)#.to(torch.float32)
        return y_pred

    
# Main Class    
class Decoder():
    def __init__(self, 
                 hashNum,
                 vector, 
                 log, 
                 lr=0.00001,
                 batch_size=750,
                 device="cuda",
                 num_workers=4,
                 epochs=200
                 ):

        # Set the parameters for pytorch model training
        self.hashNum = hashNum
        self.vector = vector
        self.device = torch.device(device)
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = epochs
        self.num_workers = num_workers
        self.log = log

        # Initialize the Pytorch model class
        self.model = MLP(self.vector)

        # Send model to Pytorch Device 
        self.model.to(self.device)
        
        # Initialize the data loaders
        self.trainLoader, self.valLoader, self.testLoader = None, None, None
        
        # Initializes Weights and Biases to keep track of experiments and training runs
        if(self.log):
            wandb.init(
                # set the wandb project where this run will be logged
                project="decoder",
                # track hyperparameters and run metadata
                config={
                "hash": self.hashNum,
                "architecture": "MLP",
                # "architecture": "2 Convolutional Layers",
                "vector": self.vector,
                "dataset": "Z scored",
                "epochs": self.num_epochs,
                "learning_rate": self.lr,
                "batch_size:": self.batch_size,
                "num_workers": self.num_workers
                }
            )
        # print("INIT ")
        # print_gpu_utilization()
    

    def train(self):
        self.trainLoader, self.valLoader, _, _ = load_nsd(vector=self.vector, 
                                                        batch_size=self.batch_size, 
                                                        num_workers=self.num_workers, 
                                                        loader=True)
        # Set best loss to negative value so it always gets overwritten
        best_loss = -1.0
        loss_counter = 0
        
        # Configure the pytorch objects, loss function (criterion)
        criterion = nn.MSELoss(reduction='sum')
        # Set the optimizer to Adam
        optimizer = Adam(self.model.parameters(), lr = self.lr)
        # optimizer = bnb.optim.Adam8bit(self.model.parameters(), lr = self.lr)
        # scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        # print_gpu_utilization()
        # Begin training, iterates through epochs, and through the whole dataset for every epoch
        for epoch in tqdm(range(self.num_epochs), desc="epochs"):
            
            # For each epoch, do a training and a validation stage
            # Entering training stage
            self.model.train()
            
            
            # Keep track of running loss for this training epoch
            running_loss = 0.0
            for i, data in enumerate(self.trainLoader):
                # torch.cuda.empty_cache()
                # Load the data out of our dataloader by grabbing the next chunk
                # The chunk is the same size as the batch size
                # x_data = Brain Data
                # y_data = Clip/Z vector Data
                x_data, y_data = data
                # x_data = nn.functional.pad(input=x_data, pad=(0, 2, 0, 0), mode='constant', value=0)
                # print(i, " TRAIN 2 ")
                # print_gpu_utilization()
                # Moving the tensors to the GPU
                x_data = x_data.to(self.device)
                y_data = y_data.to(self.device)
                
                
                
                # Forward pass: Compute predicted y by passing x to the model
                # with torch.set_grad_enabled(True):
                # with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
                    # print("TRAIN 3 ")
                    # print_gpu_utilization()
                    # Train the x data in the model to get the predicted y value. 
                pred_y = self.model(x_data).to(self.device)
                    # assert pred_y.dtype is torch.float16
                    # print("TRAIN 4 ")
                    # print_gpu_utilization()
                    # Compute the loss between the predicted y and the y data. 
                loss = criterion(pred_y, y_data)
                    # assert loss.dtype is torch.float32
                    # print("TRAIN 5 ")
                    # print_gpu_utilization()
                # Perform weight updating
                # scaler.scale(loss).backward()
                loss.backward()
                # if ((i + 1) % gradient_accumulation == 0) or (i + 1 == len(self.trainLoader)):
                # scaler.step(optimizer)
                # scaler.update()
                    
                optimizer.step()
                # Zero gradients in the optimizer
                optimizer.zero_grad()
                # print("TRAIN 6 ")
                # print_gpu_utilization()
                # del x_data
                # del y_data
                # tqdm.write('train loss: %.3f' %(loss.item()))
                # Add up the loss for this training round
                running_loss += loss.item()
                # del loss
            tqdm.write('[%d] train loss: %.8f' %
                (epoch + 1, running_loss /len(self.trainLoader)))
                #     # wandb.log({'epoch': epoch+1, 'loss': running_loss/(50 * self.batch_size)})
            
                
            # Entering validation stage
            # Set model to evaluation mode
            self.model.eval()
            with torch.no_grad():
                running_test_loss = 0.0
                for i, data in enumerate(self.valLoader):
                    
                    # Loading in the test data
                    x_data, y_data = data
                    # x_data = nn.functional.pad(input=x_data, pad=(0, 2, 0, 0), mode='constant', value=0)
                    x_data = x_data.to(self.device)
                    y_data = y_data.to(self.device)
                    # with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
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
                    wandb.log({'epoch': epoch+1, 'test_loss': test_loss})
                        
                # Check if we need to save the model
                # Early stopping
                if(best_loss == -1.0 or test_loss < best_loss):
                    best_loss = test_loss
                    torch.save(self.model.state_dict(), "models/" + self.hashNum + "_model_" + self.vector + ".pt")
                    loss_counter = 0
                else:
                    loss_counter += 1
                    tqdm.write("loss counter: " + str(loss_counter))
                    if(loss_counter >= 5):
                        break
                
        # Load our best model into the class to be used for predictions
        self.model.load_state_dict(torch.load("models/" + self.hashNum + "_model_" + self.vector + ".pt", map_location=self.device))

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
        
        global_y_pred = pred_y.reshape((y_test.shape[0], 257,768))[:,0,:]
        global_y_test = y_test.reshape((y_test.shape[0], 257,768))[:,0,:]
        
        print(global_y_pred.shape, global_y_test.shape)
        
        global_pearson = torch.mean(PeC(global_y_pred.moveaxis(0,1), global_y_test.moveaxis(0,1)))
        global_loss = criterion(global_y_pred, global_y_test)
        
        print("Vector Correlation: ", float(pearson))
        print("Vector Correlation Global: ", float(global_pearson))
        print("Loss: ", float(loss))
        print("Loss Global: ", float(global_loss))