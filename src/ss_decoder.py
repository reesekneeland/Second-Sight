# Only GPU's in use
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
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
from pycocotools.coco import COCO
import h5py
from utils import *
import wandb
import copy
from tqdm import tqdm


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
class MLP(torch.nn.Module):
    def __init__(self, vector):
        super(MLP, self).__init__()
        self.relu = nn.ReLU()
        if(vector=="c_img_0" or vector=="c_text_0"):
            self.linear = nn.Linear(11838, 15000)
            self.linear2 = nn.Linear(15000, 5000)
            self.outlayer = nn.Linear(5000, 768)
        elif(vector == "z" or vector == "z_img_mixer"):
            self.linear = nn.Linear(11838, 15000)
            self.linear2 = nn.Linear(15000, 20000)
            self.outlayer = nn.Linear(20000, 16384)
    def forward(self, x):
        y_pred = self.relu(self.linear(x))
        y_pred = self.relu(self.linear2(y_pred))
        y_pred = self.outlayer(y_pred)
        return y_pred
    
# Main Class    
class SS_Decoder():
    def __init__(self, 
                hashNum,
                vector, 
                log, 
                encoderHash,
                lr=0.00001,
                batch_size=750,
                parallel=True,
                device="cuda",
                num_workers=16,
                epochs=200
                ):

        # Set the parameters for pytorch model training
        self.hashNum = hashNum
        self.vector = vector
        self.encModel = encoderHash + "_model_" + self.vector + ".pt"
        self.device = torch.device(device)
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = epochs
        self.num_workers = num_workers
        self.log = log
        self.parallel = parallel

        # Initialize the Pytorch model class
        self.model = MLP(self.vector)
        
        # Configure multi-gpu training
        if(self.parallel):
            self.model = nn.DataParallel(self.model)
        
        # Send model to Pytorch Device 
        self.model.to(self.device)
        
        # Initialize the data loaders
        self.trainLoader, self.valLoader, self.testLoader = load_cc3m(vector=self.vector,
                                                      modelId = self.encModel,
                                                      batch_size=self.batch_size, 
                                                      num_workers=self.num_workers)
        
        # Initializes Weights and Biases to keep track of experiments and training runs
        if(self.log):
            wandb.init(
                # set the wandb project where this run will be logged
                project="ss_decoder",
                # track hyperparameters and run metadata
                config={
                "hash": self.hashNum,
                "encoder model": self.encModel,
                "architecture": "MLP",
                "vector": self.vector,
                "dataset": "Z scored B', predicted brain data on CC3M",
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
        criterion = nn.MSELoss()
        
        # Import gradients to wandb to track loss gradients
        # if(self.log):
        #     wandb.watch(self.model, criterion, log="all")
        
        # Set the optimizer to Adam
        optimizer = Adam(self.model.parameters(), lr = self.lr)
        
        # Begin training, iterates through epochs, and through the whole dataset for every epoch
        for epoch in tqdm(range(self.num_epochs), desc="epochs"):
            
            # For each epoch, do a training and a validation stage
            # Entering training stage
            self.model.train()
            
            # Keep track of running loss for this training epoch
            running_loss = 0.0
            for i, data in enumerate(tqdm(self.trainLoader, desc="iter")):
                
                # Load the data out of our dataloader by grabbing the next chunk
                # The chunk is the same size as the batch size
                # x_data = Brain Data
                # y_data = Clip/Z vector Data
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
                if(self.log and i%100==0):
                    wandb.log({'train_loss': loss.item()})
                # tqdm.write('train loss: %.3f' %(loss.item()))
                # Add up the loss for this training round
                running_loss += loss.item()
            tqdm.write('[%d] train loss: %.8f' %
                (epoch + 1, running_loss /len(self.trainLoader)))
                #     # wandb.log({'epoch': epoch+1, 'loss': running_loss/(50 * self.batch_size)})
                
            # Entering validation stage
            # Set model to evaluation mode
            self.model.eval()
            running_val_loss = 0.0
            for i, data in enumerate(self.valLoader):
                
                # Loading in the test data
                x_data, y_data = data
                x_data = x_data.to(self.device)
                y_data = y_data.to(self.device)
                
                # Generating predictions based on the current model
                pred_y = self.model(x_data).to(self.device)
                
                # Compute the test loss 
                loss = criterion(pred_y, y_data)

                running_val_loss += loss.item()
                
            val_loss = running_val_loss / len(self.valLoader)
                
            # Printing and logging loss so we can keep track of progress
            tqdm.write('[%d] val loss: %.8f' %
                        (epoch + 1, val_loss))
            if(self.log):
                wandb.log({'validation_loss': val_loss, 'epoch': epoch})
                    
            # Check if we need to save the model
            # Early stopping
            if(best_loss == -1.0 or val_loss < best_loss):
                best_loss = val_loss
                if(self.parallel):
                    torch.save(self.model.module.state_dict(), "/export/raid1/home/kneel027/Second-Sight/models/" + self.hashNum + "_model_" + self.vector + ".pt")
                else:
                    torch.save(self.model.state_dict(), "/export/raid1/home/kneel027/Second-Sight/models/" + self.hashNum + "_model_" + self.vector + ".pt")
                loss_counter = 0
            else:
                loss_counter += 1
                tqdm.write("loss counter: " + str(loss_counter))
                if(loss_counter >= 5):
                    break
                
        # Load our best model into the class to be used for predictions
        if(self.parallel):
            self.model.module.load_state_dict(torch.load("/export/raid1/home/kneel027/Second-Sight/models/" + self.hashNum + "_model_" + self.vector + ".pt", map_location='cuda'))
        else:
            self.model.load_state_dict(torch.load("/export/raid1/home/kneel027/Second-Sight/models/" + self.hashNum + "_model_" + self.vector + ".pt", map_location='cuda'))

def benchmark(self):
        outSize = len(self.testLoader.dataset)
        if(vector=="c_img_0" or vector=="c_text_0"):
            vecSize = 768
        elif(vector == "z" or vector == "z_img_mixer"):
            vecSize = 16384
        out = torch.zeros((outSize, vecSize)).to(self.device)
        target = torch.zeros((outSize, vecSize)).to(self.device)
        self.model.load_state_dict(torch.load("/export/raid1/home/kneel027/Second-Sight/models/" + self.hashNum + "_model_" + self.vector + ".pt"))
        self.model.eval()
        self.model.to(self.device)

        loss = 0
        pearson_loss = 0
        
        criterion = nn.MSELoss()
        
        for index, data in enumerate(self.testLoader):
            
            y_test, x_test = data
            PeC = PearsonCorrCoef(num_outputs=x_test.shape[0]).to(self.device)
            y_test = y_test.to(self.device)
            x_test = x_test.to(self.device)
            # Generating predictions based on the current model
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
        target = target.detach()
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
        plt.savefig("/export/raid1/home/kneel027/Second-Sight/charts/" + self.hashNum + "_" + self.vector + "_pearson_histogram_encoder.png")