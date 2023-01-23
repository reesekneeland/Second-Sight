# Only GPU's in use
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
import torch
from torch.autograd import Variable
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch.nn as nn
from pycocotools.coco import COCO
import h5py
from utils import *
import wandb
import copy
from tqdm import tqdm
from encoder import Encoder


# You decode brain data into clip then you encode the clip into an image. 


# target_c.pt (Ground Truth)
#   - Correct c vector made decoder of size 1x77x1024. (When put into stable diffusion gives you the correct image)

# target_z.pt (Ground Truth)
#   - Correct z vector made decoder of size 1x4x64x64. (When put into stable diffusion gives you the correct image)


# output_c.pt (We made)
#   - Wrong c vector made decoder of size 1x77x1024. (When put into stable diffusion gives you the wrong image)

# output_z.pt (We made)
#   - Wrong z vector made decoder of size 1x4x64x64. (When put into stable diffusion gives you the wrong image)


# Environments:
# 
#  Most environments are named the same folder they should be used in. 
#  
#  ldm (Main Environment): 
#   - Run's stable diffusion (Perfectly configured on this environment)
#   - NSD access (access to data)
#   - xformers (Library for memory purposes)
#       
#  ldm-custom-clip:
#    - stable diffusion environment configured with custom open-clip install from source
#
#  image-modification: 
#    - For the iamge modification folder
#    - Older statble diffusion without any special packages
#
#  mind-reader:
#    - For the mind reader folder which is the mind reader paper
#
#  ss:
#    - For the (deprecated) SelfSupervisedReconstr folder which is the self-supervised learning paper


# Pytorch model class for Linear regression layer Neural Network
class LinearRegression(torch.nn.Module):
    def __init__(self, vector):
        super(LinearRegression, self).__init__()
        if(vector == "c"):
            self.linear = nn.Linear(24948, 78848)
        elif(vector == "z"):
            self.linear = nn.Linear(4627,  16384)
    
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

<<<<<<< HEAD
#FOR C VECTOR ONLY *** TEST *** 
=======
# Pytorch model class for Convolutional Neural Network
>>>>>>> 735baf77643be759abb4bb9be25a46dfb9223242
class CNN(torch.nn.Module):
    def __init__(self, vector):
        super(CNN, self).__init__()
        self.conv_layer1 = self._conv_layer_set(1, 32)
        self.conv_layer2 = self._conv_layer_set(32, 64)
        self.flatten = nn.Flatten(start_dim=1)
        if(vector == "c"):
            self.fc1 = nn.Linear(64*9*4*5, 78848)
        elif(vector == "z"):
            self.fc1 = nn.Linear(64*9*4*5,  16384)
        # self.relu = nn.LeakyReLU()
        # self.batch=nn.BatchNorm1d(64*9*4*5)
        # self.drop=nn.Dropout(p=0.15)
        
            
    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0),
        nn.LeakyReLU(),
        nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer
    
    def forward(self, x):
        # print("size1: ", x.shape)
        out = self.conv_layer1(x)
        # print("size2 out size: ", out.shape)
        out = self.conv_layer2(out)
        # print("flattened out size: ", out.shape)
        out = self.flatten(out)
        # print("flattened out size: ", out.shape)
        # out = self.relu(out)
        # out = self.batch(out)
        out = self.fc1(out)
        return out

# Main Class    
class Decoder():
    def __init__(self, 
                 lr,
                 vector, 
                 log, 
                 batch_size,
                 parallel=True,
                 device="cuda",
                 num_workers=16,
                 epochs=200,
                 only_test=False
                 ):

        # Set the parameters for pytorch model training
        self.vector = vector
        self.device = torch.device(device)
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = epochs
        self.num_workers = num_workers
        self.log = log
        self.only_test = only_test
        self.parallel = parallel
        
        if(self.vector == "z"):
            self.datasize = (1, 16384)
        elif(self.vector == "c"):
            self.datasize = (1, 78848)

        # Initialize the Pytorch model class
        self.model = self.model_init()
        
        # Configure multi-gpu training
        if(self.parallel):
            self.model = nn.DataParallel(self.model)
        
        # Send model to Pytorch Device 
        self.model.to(self.device)
        
        # Initialize the data loaders
        self.trainloader, self.testloader = get_data(self.vector, self.batch_size, self.num_workers, self.only_test)
        
        # Initializes Weights and Biases to keep track of experiments and training runs
        if(self.log):
            wandb.init(
                # set the wandb project where this run will be logged
                project="decoder",
                
                # track hyperparameters and run metadata
                config={
                # "architecture": "Linear Regression",
                "architecture": "2 Convolutional Layers",
                "vector": self.vector,
                "dataset": "3D reduced ROI of NSD",
                "epochs": self.num_epochs,
                "learning_rate": self.lr,
                "batch_size:": self.batch_size,
                "num_workers": self.num_workers
                }
            )


    def model_init(self):
        # Initialize the Pytorch model class
        model = CNN(self.vector)
        # model = LinearRegression(self.vector)
        
        # Configure multi-gpu training
        if(self.parallel):
            model = nn.DataParallel(model)
        
        # Send model to Pytorch Device 
        model.to(self.device)
        return model
    

    def train(self):
        # If a previously trained model exists, load the best loss as the saved best model
        # try:
        #     best_loss = torch.load("best_loss_" + self.vector + ".pt")
        # except:
        self.model = self.model_init()
        best_loss = -1.0
        loss_counter = 0
        
        # Set best loss to negative value so it always gets overwritten
        
        # Configure the pytorch objects, loss function (criterion)
        criterion = nn.MSELoss(size_average = False)
        # criterion = nn.L1Loss()
        
        # Import gradients to wandb to track loss gradients
        if(self.log):
            wandb.watch(self.model, criterion, log="all")
        
        # Set the optimizer to Stochastic Gradient Descent
        optimizer = torch.optim.SGD(self.model.parameters(), lr = self.lr)
        
        # Set the learning rate scheduler to reduce the learning rate by 30% if the loss plateaus for 3 epochs
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=3)
        
        # Begin training, iterates through epochs, and through the whole dataset for every epoch
        for epoch in tqdm(range(self.num_epochs), desc="epochs"):
            
            # For each epoch, do a training and a validation stage
            # Entering training stage
            self.model.train()
            
            # Keep track of running loss for this training epoch
            running_loss = 0.0
            for i, data in enumerate(self.trainloader):
                
                # Load the data out of our dataloader by grabbing the next chunk
                # The chunk is the same size as the batch size
                # x_data = Brain Data
                # y_data = Clip Data
                x_data, y_data = data
                
                # Moving the tensors to the GPU
                x_data = x_data.to(self.device)
                y_data = y_data.to(self.device)
                
                # Zero gradients in the optimizer
                optimizer.zero_grad()
                
                # Forward pass: Compute predicted y by passing x to the model
                with torch.set_grad_enabled(True):
                    
                    # Train the x data in the model to get the predicted y value. 
                    pred_y = self.model(x_data).to(self.device)
                    
                    # Compute the loss between the predicted y and the y data. 
                    loss = criterion(pred_y, y_data)
                    
                    # Perform weight updating
                    loss.backward()
                    optimizer.step()

                # tqdm.write('train loss: %.3f' %(loss.item()))
                # Add up the loss for this training round
                running_loss += loss.item()
            tqdm.write('[%d] train loss: %.8f' %
                (epoch + 1, running_loss /self.batch_size))
                #     # wandb.log({'epoch': epoch+1, 'loss': running_loss/(50 * self.batch_size)})
                
            # Entering validation stage
            # Set model to evaluation mode
            self.model.eval()
            test_loss = 0.0
            for i, data in enumerate(self.testloader):
                
                # Loading in the test data
                x_data, y_data = data
                x_data = x_data.to(self.device)
                y_data = y_data.to(self.device)
                
                # Generating predictions based on the current model
                pred_y = self.model(x_data).to(self.device)
                
                # Compute the test loss 
                loss = criterion(pred_y, y_data)

                test_loss += loss.item()
                
            # Printing and logging loss so we can keep track of progress
            tqdm.write('[%d] test loss: %.8f' %
                        (epoch + 1, test_loss / self.batch_size))
            if(self.log):
                wandb.log({'epoch': epoch+1, 'test_loss': test_loss/self.batch_size})
            
            # Check if we need to drop the learning rate
            scheduler.step(test_loss / self.batch_size)
                    
            # Check if we need to save the model
            # Early stopping
            if(best_loss == -1.0 or test_loss < best_loss):
                best_loss = test_loss
                torch.save(best_loss, "best_loss_" + self.vector + ".pt")
                if(self.parallel):
                    torch.save(self.model.module.state_dict(), "models/model_" + self.vector + ".pt")
                else:
                    torch.save(self.model.state_dict(), "models/model_" + self.vector + ".pt")
                loss_counter = 0
            else:
                loss_counter += 1
                tqdm.write("loss counter: " + str(loss_counter))
                if(loss_counter >= 5):
                    break
                
        # Load our best model into the class to be used for predictions
        if(self.parallel):
            self.model.module.load_state_dict(torch.load("models/model_" + self.vector + ".pt"))
        else:
            self.model.load_state_dict(torch.load("models/model_" + self.vector + ".pt"))


    def predict(self, indices=[0], model="model_z.pt"):
        self.model = self.model_init()
        os.makedirs("../latent_vectors/" + model, exist_ok=True)
        # Load the model into the class to be used for predictions
        if(self.parallel):
            self.model.module.load_state_dict(torch.load("models/" + model))
        else:
            self.model.load_state_dict(torch.load("models/" + model))
        self.model.eval()
        outputs, targets = [], []
        x, y = next(iter(self.testloader))
        print(x.shape, y.shape)
        for i in indices:
            
            # Loading in the test data
            x_data = x[i]
            y_data = y[i]
            print(x_data.shape, y_data.shape)
            x_data = x_data[None,:].to(self.device)
            y_data = y_data[None,:].to(self.device)
            print(x_data.shape, y_data.shape)
            # Generating predictions based on the current model
            pred_y = self.model(x_data).to(self.device)
            outputs.append(pred_y)
            targets.append(y_data)
            torch.save(pred_y, "../latent_vectors/" + model + "/" + "output_" + self.vector + ".pt")
            torch.save(y_data, "../latent_vectors/" + model + "/" + "target_" + self.vector + ".pt")
        return outputs, targets