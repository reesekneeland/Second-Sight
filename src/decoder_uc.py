import torch
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torchmetrics import PearsonCorrCoef
from utils import *
import wandb
from tqdm import tqdm
import pickle as pk
import seaborn as sns
import matplotlib.pylab as plt
# import bitsandbytes as bnb

# Pytorch model class for Linear regression layer Neural Network
class MLP(torch.nn.Module):
    def __init__(self, vector, x_size):
        super(MLP, self,).__init__()
        self.vector = vector
        if(self.vector == "c_img_uc"):
            self.linear = nn.Linear(x_size, 15000)
            self.linear2 = nn.Linear(15000, 15000)
            self.linear3 = nn.Linear(15000, 15000)
            self.linear4 = nn.Linear(15000, 15000)
            self.outlayer = nn.Linear(15000, 1024)
        elif(self.vector == "c_text_uc"):
            self.linear = nn.Linear(x_size, 15000)
            self.linear2 = nn.Linear(15000, 15000)
            self.linear3 = nn.Linear(15000, 15000)
            self.linear4 = nn.Linear(15000, 15000)
            self.outlayer = nn.Linear(15000, 78848)
        elif(self.vector == "z_vdvae"):
            self.linear = nn.Linear(x_size, 12500)
            self.outlayer = nn.Linear(12500, 91168)
            # self.outlayer = nn.Linear(x_size, 91168)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.linear(x))
        if(self.vector == "c_img_uc" or self.vector == "c_text_uc"):
            # x = self.relu(self.linear(x))
            x = self.relu(self.linear2(x))
            x = self.relu(self.linear3(x))
            x = self.relu(self.linear4(x))
        y_pred = self.outlayer(x)
        return y_pred

    
# Main Class    
class Decoder_UC():
    def __init__(self, 
                 hashNum,
                 vector, 
                 log=False, 
                 subject=1,
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
        self.subject = subject
        
        
        # Initialize the data loaders
        self.trainLoader, self.valLoader, _ = load_nsd(vector=self.vector, 
                                                                    batch_size=self.batch_size, 
                                                                    num_workers=self.num_workers, 
                                                                    loader=True,
                                                                    average=False,
                                                                    subject=self.subject)
        x_size = self.trainLoader.dataset[0][0].shape[0]
        # Initialize the Pytorch model class
        self.model = MLP(self.vector, x_size)

        # Send model to Pytorch Device 
        self.model.to(self.device)
        
        # Initializes Weights and Biases to keep track of experiments and training runs
        if(self.log):
            wandb.init(
                # set the wandb project where this run will be logged
                project="decoder_uc",
                # track hyperparameters and run metadata
                config={
                "hash": self.hashNum,
                "architecture": "MLP unCLIP",
                "subject": self.subject,
                "vector": self.vector,
                "dataset": "Z scored",
                "epochs": self.num_epochs,
                "learning_rate": self.lr,
                "batch_size:": self.batch_size,
                "num_workers": self.num_workers
                }
            )
    

    def train(self):
        # Set best loss to negative value so it always gets overwritten
        best_loss = -1.0
        loss_counter = 0
        
        # Configure the pytorch objects, loss function (criterion)
        criterion = nn.MSELoss(reduction='sum')
        # Set the optimizer to Adam
        optimizer = Adam(self.model.parameters(), lr = self.lr)
        # Begin training, iterates through epochs, and through the whole dataset for every epoch
        for epoch in tqdm(range(self.num_epochs), desc="epochs for subject{}".format(self.subject)):
            # For each epoch, do a training and a validation stage
            # Entering training stage
            self.model.train()
            # Keep track of running loss for this training epoch
            running_loss = 0.0
            for i, data in enumerate(self.trainLoader):
                # x_data = Brain Data
                # y_data = Clip/Z vector Data
                x_data, y_data = data
                # Moving the tensors to the GPU
                
                x_data = x_data.to(self.device)
                y_data = y_data.to(self.device)
                # Forward pass: Compute predicted y by passing x to the model
                # Train the x data in the model to get the predicted y value. 
                pred_y = self.model(x_data).to(self.device)
                # scaled_pred_y /= torch.norm(y_data)
                
                # Compute the loss between the predicted y and the y data. 
                loss = criterion(pred_y, y_data)
                loss.backward()
                optimizer.step()
                # Zero gradients in the optimizer
                optimizer.zero_grad()
                # Add up the loss for this training round
                running_loss += loss.item()
            tqdm.write('[%d] train loss: %.8f' %
                (epoch + 1, running_loss /len(self.trainLoader.dataset)))
                #     # wandb.log({'epoch': epoch+1, 'loss': running_loss/(50 * self.batch_size)})
            
                
            # Entering validation stage
            # Set model to evaluation mode
            self.model.eval()
            running_test_loss = 0.0
            for i, data in enumerate(self.valLoader):
                
                # Loading in the test data
                x_data, y_data = data
                x_data = x_data.to(self.device)
                y_data = y_data.to(self.device)
                # Forward pass: Compute predicted y by passing x to the model
                # Train the x data in the model to get the predicted y value. 
                pred_y = self.model(x_data).to(self.device)
                # scaled_pred_y /= torch.norm(y_data)
                
                # Compute the loss between the predicted y and the y data. 
                loss = criterion(pred_y, y_data)
                running_test_loss += loss.item()
            test_loss = running_test_loss / len(self.valLoader.dataset)
                
            # Printing and logging loss so we can keep track of progress
            tqdm.write('[%d] test loss: %.8f' %
                        (epoch + 1, test_loss))
            if(self.log):
                wandb.log({'epoch': epoch+1, 'test_loss': test_loss})
                    
            # Check if we need to save the model
            # Early stopping
            if(best_loss == -1.0 or test_loss < best_loss):
                best_loss = test_loss
                torch.save(self.model.state_dict(), "models/subject{subject}/{hash}_model_{vec}.pt".format(subject=self.subject, hash=self.hashNum, vec=self.vector))
                loss_counter = 0
            else:
                loss_counter += 1
                tqdm.write("loss counter: " + str(loss_counter))
                if(loss_counter >= 5):
                    break
        if(self.log):
                wandb.finish()
                
        

    def predict(self, x):
        self.model.load_state_dict(torch.load("models/subject{subject}/{hash}_model_{vec}.pt".format(subject=self.subject, hash=self.hashNum, vec=self.vector), map_location=self.device))
        self.model.eval()
        self.model.to(self.device)
        out = self.model(x.to(self.device)).to(torch.float16)
        return out
    
    
    def benchmark(self, average=True):
        _, _, x_test, _, _, y_test, _ = load_nsd(vector=self.vector, 
                                                loader=False,
                                                average=average,
                                                subject=self.subject)
        # Load our best model into the class to be used for predictions
        modelId = "{hash}_model_{vec}.pt".format(hash=self.hashNum, vec=self.vector)
        self.model.load_state_dict(torch.load("models/subject{}/{}".format(self.subject, modelId)))
        self.model.eval()

        criterion = nn.MSELoss()
        PeC = PearsonCorrCoef(num_outputs=y_test.shape[0]).to(self.device)
        
        x_test = x_test.to(self.device)
        y_test = y_test.to(self.device)
        
        pred_y = self.model(x_test)
        pearson = torch.mean(PeC(pred_y.moveaxis(0,1), y_test.moveaxis(0,1)))
        loss = criterion(pred_y, y_test)

        
        print("Model ID: {}, Subject: {}, Averaged: {}".format(modelId, self.subject, average))
        print("Vector Correlation: ", float(pearson))
        print("Loss: ", float(loss))