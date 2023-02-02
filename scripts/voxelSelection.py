# Only GPU's in use
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
import torch
from torchmetrics.functional import pearson_corrcoef
from torch.autograd import Variable
import numpy as np
from nsd_access import NSDAccess
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch.nn as nn
from pycocotools.coco import COCO
import h5py
sys.path.append('../src')
from utils import *
import wandb
import copy
from tqdm import tqdm
import nibabel as nib

# c
# Hash            = 182
# Mean            = 0.016873473
# Threshold       = 0.105578
# Number of voxels = 527

# z
# Hash             = 184
# Mean             = 0.0857764
# Threshold        = 0.07364
# Number of voxels = 5764

# c_prompt
# Hash             = 189
# Mean             = 
# Threshold        = 
# Number of voxels = 



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
        if(vector == "c_prompt"):
            self.linear = nn.Linear(78848,  11838)
        elif(vector == "z"):
            self.linear = nn.Linear(16384,  11838)
        elif(vector == "c"):
            self.linear = nn.Linear(1536,  11838)
    
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

# Main Class    
class VectorMapping():
    def __init__(self, vector):
        
        # Lock the seed to get ride of randomization. 
        seed = 42
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # First URL: This is the original read-only NSD file path (The actual data)
        # Second URL: Local files that we are adding to the dataset and need to access as part of the data
        # Object for the NSDAccess package
        self.nsda = NSDAccess('/home/naxos2-raid25/kneel027/home/surly/raid4/kendrick-data/nsd', '/home/naxos2-raid25/kneel027/home/kneel027/nsd_local')
        
        # Pytorch Device 
        self.device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
        
        # Condition to set layer size 
        self.vector = vector
        
        # self.hashNum = update_hash()
        # self.hashNum = "011"
        self.hashNum = "096"

        # Initializes the pytorch model class
        # self.model = model = Linear5Layer(self.vector)
        self.model = LinearRegression(self.vector).to(self.device)
        
        
        # Set the parameters for pytorch model training
        # 11.8 for z
        self.lr = 0.0005
        self.batch_size = 750
        self.num_epochs = 300
        self.num_workers = 4
        self.log = False
        
        # Initializes Weights and Biases to keep track of experiments and training runs
        if(self.log):
            wandb.init(
                # set the wandb project where this run will be logged
                project="voxelSelection clip encoder to brain vector",
                
                # track hyperparameters and run metadata
                config={
                "hash": self.hashNum,
                # "architecture": "Linear Regression",
                # "architecture": "Linear 2 Layer",
                # "architecture": "Linear 3 Layer",
                # "architecture": "Linear 4 Layer",
                "architecture": "Linear regression",
                "vector": self.vector,
                "dataset": "Entire visual cortex, clip vectors from image embedding",
                "epochs": self.num_epochs,
                "learning_rate": self.lr,
                "batch_size:": self.batch_size,
                "num_workers": self.num_workers
                }
            )

    
    def load_data_masked(self, vector):
        
        # Loads the preprocessed data
        prep_path = "/export/raid1/home/kneel027/nsd_local/preprocessed_data/"
        y = torch.load(prep_path + "x/whole_region_11838_unnormalized.pt").requires_grad_(False)
        x  = torch.load(prep_path + vector + "/vector.pt").requires_grad_(False)
        print(x.shape, y.shape)
        x_train = x[:25500]
        x_test = x[25500:27750]
        y_train = y[:25500]
        y_test = y[25500:27750]
        
        return x_train, x_test, y_train, y_test
    
    # Loads the data into an array of DataLoaders for each seperate C vector
    def get_data_masked(self):
        x, x_test, y, y_test = self.load_data_masked(self.vector)
        # Loads the raw tensors into Dataset objects
        trainset = torch.utils.data.TensorDataset(x, y) #.type(torch.LongTensor)
        testset = torch.utils.data.TensorDataset(x_test, y_test) #.type(torch.LongTensor)
        
        # Loads the Dataset into a DataLoader
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        return trainloader, testloader
    
        
    def train(self, trainLoader, testLoader):

        # Set best loss to negative value so it always gets overwritten
        best_loss = -1.0
        
        #check for how many iterations the loss has not been decreasing
        loss_counter = 0
        
        # Configure the pytorch objects, loss function (criterion)
        criterion = nn.MSELoss()
        
        # Import gradients to wandb to track loss gradients
        if(self.log):
            wandb.watch(self.model, criterion, log="all")
        
        # Set the optimizer to Stochastic Gradient Descent
        optimizer = torch.optim.SGD(self.model.parameters(), lr = self.lr)
        
        # Set the learning rate scheduler to reduce the learning rate if the loss plateaus for 3 epochs
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.4, patience=3)
        
        # Begin training, iterates through epochs, and through the whole dataset for every epoch
        for epoch in tqdm(range(self.num_epochs), desc="epochs"):
            
            #training process
            self.model.train()
            
            # Keep track of running loss for this training epoch
            running_loss = 0.0
            for i, data in enumerate(trainLoader):
                
                # Load the data out of our dataloader by grabbing the next chunk
                x_data, y_data = data
                
                # Moving the tensors to the GPU
                x_data = x_data.to(self.device)
                y_data = y_data.to(self.device)
                
                # Zero gradients in the optimizer
                optimizer.zero_grad()
                
                # Forward pass: Compute predicted y by passing x to the model
                with torch.set_grad_enabled(True):
                    pred_y = self.model(x_data).to(self.device)
                    # Compute and print loss
                    loss = criterion(pred_y, y_data)
                    
                    # Perform weight updating
                    loss.backward()
                    optimizer.step()
                        
                # Add up the loss for this training round
                running_loss += loss.item()
            tqdm.write('[%d, %5d] train loss: %.8f' %
                (epoch + 1, i + 1, running_loss /len(trainLoader)))
                    # wandb.log({'epoch': epoch+1, 'loss': running_loss/(50 * self.batch_size)})

        # Entering validation stage
        # Set model to evaluation mode
            self.model.eval()
            running_test_loss = 0.0
            for i, data in enumerate(testLoader):
                # Loading in the test data
                x_data, y_data = data
                x_data = x_data.to(self.device)
                y_data = y_data.to(self.device)
                
                # Generating predictions based on the current model
                pred_y = self.model(x_data).to(self.device)
                
                # Compute loss
                loss = criterion(pred_y, y_data)
                running_test_loss+=loss.item()
                
            test_loss = running_test_loss/len(testLoader)
            # Printing and logging loss so we can keep track of progress
            tqdm.write('[%d] test loss: %.8f' %
                        (epoch + 1, test_loss))
            if(self.log):
                wandb.log({'epoch': epoch+1, 'test_loss': test_loss})
            
            # Check if we need to drop the learning rate
            scheduler.step(test_loss)
                    
            # Check if we need to save the model
            if(best_loss == -1.0 or test_loss < best_loss):
                best_loss = test_loss
                torch.save(self.model.state_dict(), "/export/raid1/home/kneel027/Second-Sight/models/" + self.hashNum + "_" + self.vector + "2voxels.pt")
                loss_counter = 0
            else:
                loss_counter += 1
                tqdm.write("loss counter: " + str(loss_counter))
                if(loss_counter >= 8):
                    break
        
        # Load our best model and returning it
        self.model.load_state_dict(torch.load("/export/raid1/home/kneel027/Second-Sight/models/" + self.hashNum + "_" + self.vector + "2voxels.pt"))


    #reassemble an output c vector from the individual component models
    def predict(self, testLoader, hashNum):
        out = torch.zeros((2250,11838))
        target = torch.zeros((2250, 11838))
        self.model.load_state_dict(torch.load("/export/raid1/home/kneel027/Second-Sight/models/" + hashNum + "_" + self.vector + "2voxels.pt"))
        self.model.eval()
        self.model.to(self.device)

        for index, data in enumerate(testLoader):
            
            # Loading in the test data
            x_data, y_data = data
            x_data = x_data.to(self.device)
            y_data = y_data.to(self.device)
            # Generating predictions based on the current model
            pred_y = self.model(x_data).to(self.device)
            out[index*self.batch_size:index*self.batch_size+self.batch_size] = pred_y
            target[index*self.batch_size:index*self.batch_size+self.batch_size] = y_data
        
        out = out.detach()
        target = target.detach()
        # outPrev = torch.load("output_z_scalar.pt")
        # targetPrev = torch.load("target_z_scalar.pt")
        # print("out check", torch.eq(out, outPrev))
        # print("target check", torch.eq(target, targetPrev))
        
    
        # Pearson correlation
        r = []
        for p in range(out.shape[1]):
            r.append(pearson_corrcoef(out[:,p], target[:,p]))
        r = np.array(r)
        print(np.mean(r))
        threshold = round((min(r) * -1), 6)
        print(threshold)
        mask = np.array(len(r) * [True])
        # for threshold in [0.0, 0.05, 0.1, 0.2]:
        #     threshmask = np.where(np.array(r) > threshold, mask, False)
        #     print(threshmask.shape)
        #     np.save("/export/raid1/home/kneel027/Second-Sight/masks/" + hashNum + "_" + self.vector + "2voxels_pearson_thresh" + str(threshold), threshmask)
        
        threshmask = np.where(np.array(r) > threshold, mask, False)
        print(threshmask.sum())
        np.save("/export/raid1/home/kneel027/Second-Sight/masks/" + hashNum + "_" + self.vector + "2voxels_pearson_thresh" + str(threshold), threshmask)
            
        #print(r)
        #r = np.log(r)
        # plt.hist(r, bins=40, log=True)
        # #plt.yscale('log')
        # plt.savefig("/export/raid1/home/kneel027/Second-Sight/scripts/" + hashNum + "_" + self.vector + "2voxels_pearson_histogram_log_applied.png")
        
        
        # torch.save(out, "output_z_scalar.pt")
        # torch.save(target, "target_z_scalar.pt")
        


def main():
    vector = "z"
    VM = VectorMapping(vector)
    train, test = VM.get_data_masked()
    VM.train(train, test)
    
    VM.predict(test, VM.hashNum)

if __name__ == "__main__":
    main()
