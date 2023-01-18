import torch
from torch.autograd import Variable
import numpy as np
from nsd_access import NSDAccess

# Only GPU's in use
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch.nn as nn
from pycocotools.coco import COCO
import h5py
from file_utility import *
import wandb
import copy
from tqdm import tqdm
from reconstruct import reconstruct


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
            self.linear = nn.Linear(4627, 78848)
        elif(vector == "z"):
            self.linear = nn.Linear(4627,  16384)
    
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

# Pytorch model class for Linear regression layer Neural Network for individual c tokens
class LinearRegression_c_seperate(torch.nn.Module):
    def __init__(self, vector):
        super(LinearRegression_c_seperate, self).__init__()
        if(vector == "c"):
            self.linear = nn.Linear(4627, 1024)
    
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
    

# Pytorch model class for Linear 2 layer Neural Network
class Linear2Layer(torch.nn.Module):
    def __init__(self, vector):
        super(Linear2Layer, self).__init__()
        # Checking if our model is to map the C or the Z vector, and we set output size accordingly
        if(vector == "c"):
            self.linear1 = nn.Linear(4627, 10000)
            self.linear2 = nn.Linear(10000, 78848)
        elif(vector == "z"):
            self.linear1 = nn.Linear(4627, 10000)
            self.linear2 = nn.Linear(10000, 16384)
    
    def forward(self, x):
        y_1 = self.linear1(x)
        y_pred = self.linear2(y_1)
        return y_pred
    
# Pytorch model class for Linear 3 layer Neural Network
class Linear3Layer(torch.nn.Module):
    def __init__(self, vector):
        super(Linear3Layer, self).__init__()
        # Checking if our model is to map the C or the Z vector, and we set output size accordingly
        if(vector == "c"):
            self.linear1 = nn.Linear(4627, 10000)
            self.linear2 = nn.Linear(10000, 10000)
            self.linear3 = nn.Linear(10000, 78848)
        elif(vector == "z"):
            self.linear1 = nn.Linear(4627, 10000)
            self.linear2 = nn.Linear(10000, 10000)
            self.linear3 = nn.Linear(10000, 16384)
    
    def forward(self, x):
        y_1 = self.linear1(x)
        y_2 = self.linear2(y_1)
        y_pred = self.linear3(y_2)
        return y_pred
    
# Pytorch model class for Linear 4 layer Neural Network
class Linear4Layer(torch.nn.Module):
    def __init__(self, vector):
        super(Linear4Layer, self).__init__()
        # Checking if our model is to map the C or the Z vector, and we set output size accordingly
        if(vector == "c"):
            self.linear1 = nn.Linear(4627, 10000)
            self.linear2 = nn.Linear(10000, 10000)
            self.linear3 = nn.Linear(10000, 10000)
            self.linear4 = nn.Linear(10000, 78848)
        elif(vector == "z"):
            self.linear1 = nn.Linear(4627, 10000)
            self.linear2 = nn.Linear(10000, 10000)
            self.linear3 = nn.Linear(10000, 10000)
            self.linear4 = nn.Linear(10000, 16384)
    
    def forward(self, x):
        y_1 = self.linear1(x)
        y_2 = self.linear2(y_1)
        y_3 = self.linear3(y_2)
        y_pred = self.linear4(y_3)
        return y_pred
    
# Pytorch model class for Linear 5 layer Neural Network
class Linear5Layer(torch.nn.Module):
    def __init__(self, vector):
        super(Linear5Layer, self).__init__()
        # Checking if our model is to map the C or the Z vector, and we set output size accordingly
        if(vector == "c"):
            self.linear1 = nn.Linear(4627, 10000)
            self.linear2 = nn.Linear(10000, 10000)
            self.linear3 = nn.Linear(10000, 10000)
            self.linear4 = nn.Linear(10000, 10000)
            self.linear5 = nn.Linear(10000, 78848)
        elif(vector == "z"):
            self.linear1 = nn.Linear(4627, 10000)
            self.linear2 = nn.Linear(10000, 10000)
            self.linear3 = nn.Linear(10000, 10000)
            self.linear4 = nn.Linear(10000, 10000)
            self.linear5 = nn.Linear(10000, 16384)
    
    def forward(self, x):
        y_1 = self.linear1(x)
        y_2 = self.linear2(y_1)
        y_3 = self.linear3(y_2)
        y_4 = self.linear4(y_3)
        y_pred = self.linear5(y_4)
        return y_pred


# Main Class    
class VectorMapping():
    def __init__(self, vector):

        # First URL: This is the original read-only NSD file path (The actual data)
        # Second URL: Local files that we are adding to the dataset and need to access as part of the data
        # Object for the NSDAccess package
        self.nsda = NSDAccess('/home/naxos2-raid25/kneel027/home/surly/raid4/kendrick-data/nsd', '/home/naxos2-raid25/kneel027/home/kneel027/nsd_local')
        
        # Pytorch Device 
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
       
        # Segmented dataset paths and file loader
        # Specific regions for the voxels of the brain
        voxel_dir = "/export/raid1/home/styvesg/data/nsd/voxels/"
        voxel_data_set = h5py.File(voxel_dir+'voxel_data_V1_4_part1.h5py', 'r')
        
        # Main dictionary that contains the voxel activation data, as well as ROI maps and indices for the actual beta
        self.voxel_data_dict = embed_dict({k: np.copy(d) for k,d in voxel_data_set.items()})
        voxel_data_set.close()
        
        # Condition to set layer size 
        self.vector = vector
        if(self.vector == "z"):
            self.datasize = (1, 16384)
        elif(self.vector == "c"):
            self.datasize = (1, 78848)

        # Initializes the pytorch model class
        # self.model = model = Linear5Layer(self.vector)
        self.model = LinearRegression_c_seperate(self.vector)
        
        # Set the parameters for pytorch model training
        self.lr = 0.0000015
        self.batch_size = 1024
        self.num_epochs = 75
        self.num_workers = 16
        self.log = False
        
        # Initializes Weights and Biases to keep track of experiments and training runs
        if(self.log):
            wandb.init(
                # set the wandb project where this run will be logged
                project="vector_mapping",
                
                # track hyperparameters and run metadata
                config={
                # "architecture": "Linear Regression",
                # "architecture": "Linear 2 Layer",
                # "architecture": "Linear 3 Layer",
                # "architecture": "Linear 4 Layer",
                "architecture": "Linear regression individual c vector",
                "vector": self.vector,
                "dataset": "Ghislain reduced ROI of NSD",
                "epochs": self.num_epochs,
                "learning_rate": self.lr,
                "batch_size:": self.batch_size,
                "num_workers": self.num_workers
                }
            )

    # DEPRECATED
    # Loads the entire 699,192 size beta scan into memory so we can train on it
    def load_data_whole(self, only_test=False):
        # initialize some empty tensors to allocate memory
        y_train = torch.empty((29250, 16384))
        y_test = torch.empty((750, 16384))
        x_train = torch.empty((29250, 699192))
        x_test = torch.empty((750, 699192))
        
        #Checks if we are only loading the test data so we don't have to load all the training betas
        if(not only_test):
        # train_betas = []
            for i in range(1,40):
                #loads the full collection of beta sessions for subject 1
                beta = self.nsda.read_betas(subject='subj01', 
                                    session_index=i, 
                                    trial_index=[], # empty list as index means get all for this session
                                    data_type='betas_fithrf_GLMdenoise_RR',
                                    data_format='func1pt8mm')
                beta = np.moveaxis(beta, -1, 0)
                x_train[(i-1)*750:(i-1)*750+750] = torch.from_numpy(beta.reshape(750, 699192))
                print(i)

        # Loads the test beta and puts it into a tensor
        test_betas = self.nsda.read_betas(subject='subj01', 
                                    session_index=40, 
                                    trial_index=[], # empty list as index means get all for this session
                                    data_type='betas_fithrf_GLMdenoise_RR',
                                    data_format='func1pt8mm')
        test_betas = np.moveaxis(test_betas, -1, 0)
        x_test = torch.from_numpy(test_betas.reshape(750, 699192))
        
        #loads the description object for the stimulus images for subject1
        subj1 = self.nsda.stim_descriptions[self.nsda.stim_descriptions['subject1'] != 0]
        if(not only_test):
            #iterating through all the scan data in our X dataset
            for i in range(1,29251):
                if(i%1000==0):
                    print(i)
                #index is a 0-73000 index of the coco images, which is found by finding the line of the dataframe
                #that matches the scan ID of the current beta
                index = int(subj1.loc[(subj1['subject1_rep0'] == i) | (subj1['subject1_rep1'] == i) | (subj1['subject1_rep2'] == i)].nsdId)
                #load the saved Z tensor conversion of the image we found into the Y dataset
                #DEPRECATED: Doesn't convert between C and Z vectors
                y_train[i-1] = torch.reshape(torch.load("/home/naxos2-raid25/kneel027/home/kneel027/nsd_local/nsddata_stimuli/tensors/z/" + str(index) + ".pt"), (1, 16384))
        #do the same thing for test data
        for i in range(1,751):
            index = int(subj1.loc[(subj1['subject1_rep0'] == 29250 + i) | (subj1['subject1_rep1'] == 29250 + i) | (subj1['subject1_rep2'] == 29250 + i)].nsdId)
            y_test[i-1] = torch.reshape(torch.load("/home/naxos2-raid25/kneel027/home/kneel027/nsd_local/nsddata_stimuli/tensors/z/" + str(index) + ".pt"), (1, 16384))
            
        return x_train, x_test, y_train, y_test
    
    def load_data_roi(self, seperate_c=False):
        
        # Open the dictionary of refined voxel information from Ghislain, and pull out the data variable
        voxel_data = self.voxel_data_dict['voxel_data']
        
        datashape = self.datasize
        if(seperate_c and self.vector == "c"):
            datashape= (1, 77, 1024)
        
        # Index it to the first subject
        subj1x = voxel_data["1"]
        
        # Import the data into a tensor
        x_train = torch.tensor(subj1x[:25500])
        x_test = torch.tensor(subj1x[25500:27750])
        
        # Loading the description object for subejct1
        subj1y = self.nsda.stim_descriptions[self.nsda.stim_descriptions['subject1'] != 0]
        
        # Do the same annotation extraction process from load_data_whole()
        y_train = torch.empty((25500, self.datasize[-1]))
        y_test = torch.empty((2250, self.datasize[-1]))
        if(seperate_c and self.vector == "c"):
            datashape= (1, 77, 1024)
        
        
        for i in tqdm(range(0,25500), desc="train loader"):
            #flexible to both Z and C tensors depending on class configuration
            index = int(subj1y.loc[(subj1y['subject1_rep0'] == i+1) | (subj1y['subject1_rep1'] == i+1) | (subj1y['subject1_rep2'] == i+1)].nsdId)
            y_train[i] = torch.reshape(torch.load("/home/naxos2-raid25/kneel027/home/kneel027/nsd_local/nsddata_stimuli/tensors/" + self.vector + "/" + str(index) + ".pt"), datashape)
        for i in tqdm(range(0,2250), desc="test loader"):
            index = int(subj1y.loc[(subj1y['subject1_rep0'] == 25501 + i) | (subj1y['subject1_rep1'] == 25501 + i) | (subj1y['subject1_rep2'] == 25501 + i)].nsdId)
            y_test[i] = torch.reshape(torch.load("/home/naxos2-raid25/kneel027/home/kneel027/nsd_local/nsddata_stimuli/tensors/" + self.vector + "/" + str(index) + ".pt"), datashape)

        return x_train, x_test, y_train, y_test
    
    def load_data_c_seperate(self):
        
        # Open the dictionary of refined voxel information from Ghislain, and pull out the data variable
        voxel_data = self.voxel_data_dict['voxel_data']
        
        datashape= (1, 77, 1024)
        
        # Index it to the first subject
        subj1x = voxel_data["1"]
        
        # Import the data into a tensor
        x_train = torch.tensor(subj1x[:25500])
        x_test = torch.tensor(subj1x[25500:27750])
        
        # Loading the description object for subejct1
        subj1y = self.nsda.stim_descriptions[self.nsda.stim_descriptions['subject1'] != 0]
        
        # Do the same annotation extraction process from load_data_whole()
        y_train = torch.empty((25500, 77, 1024))
        y_test = torch.empty((2250, 77, 1024))
        
        
        for i in tqdm(range(0,25500), desc="train loader"):
            #flexible to both Z and C tensors depending on class configuration
            index = int(subj1y.loc[(subj1y['subject1_rep0'] == i+1) | (subj1y['subject1_rep1'] == i+1) | (subj1y['subject1_rep2'] == i+1)].nsdId)
            y_train[i] = torch.reshape(torch.load("/home/naxos2-raid25/kneel027/home/kneel027/nsd_local/nsddata_stimuli/tensors/" + self.vector + "/" + str(index) + ".pt"), datashape)
        for i in tqdm(range(0,2250), desc="test loader"):
            index = int(subj1y.loc[(subj1y['subject1_rep0'] == 25501 + i) | (subj1y['subject1_rep1'] == 25501 + i) | (subj1y['subject1_rep2'] == 25501 + i)].nsdId)
            y_test[i] = torch.reshape(torch.load("/home/naxos2-raid25/kneel027/home/kneel027/nsd_local/nsddata_stimuli/tensors/" + self.vector + "/" + str(index) + ".pt"), datashape)

        return x_train, x_test, y_train, y_test
    
    # Loads the data into an array of DataLoaders for each seperate C vector
    def get_data_c_seperate(self):
        x, x_test, y, y_test = self.load_data_c_seperate()
        print(y[:,0,:].shape)
        # Loads the raw tensors into an array of Dataset objects
        trainset, testset, trainloader, testloader = [], [], [], []
        for i in range(0, 77):
            trainset.append(torch.utils.data.TensorDataset(x, y[:,i,:])) #.type(torch.LongTensor)
            testset.append(torch.utils.data.TensorDataset(x_test, y_test[:,i,:])) #.type(torch.LongTensor)
        
        # Loads the Dataset into an array of DataLoaders
        for i in range(0, 77):
            trainloader.append(torch.utils.data.DataLoader(trainset[i], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True))
            testloader.append(torch.utils.data.DataLoader(testset[i], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False))
        print(len(trainloader))
        return trainloader, testloader
    
    # Loads the data and puts it into a DataLoader
    def get_data(self):
        x, x_test, y, y_test = self.load_data_roi()
        # Loads the raw tensors into a Dataset object
        trainset = torch.utils.data.TensorDataset(x, y) #.type(torch.LongTensor)
        testset = torch.utils.data.TensorDataset(x_test, y_test) #.type(torch.LongTensor)
        
        # Loads the Dataset into a DataLoader
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        return trainloader, testloader

    def train(self, trainLoader, testLoader):
        # If a previously trained model exists, load the best loss as the saved best model
        try:
            best_loss = torch.load("best_loss_" + self.vector + ".pt")
        except:
            best_loss = -1.0
        
        # Set best loss to negative value so it always gets overwritten
        
        # Configure the pytorch objects, loss function (criterion)
        criterion = nn.MSELoss(size_average = False)
        
        # Import gradients to wandb to track loss gradients
        if(self.log):
            wandb.watch(self.model, criterion, log="all")
        
        # Set the optimizer to Stochastic Gradient Descent
        optimizer = torch.optim.SGD(self.model.parameters(), lr = self.lr)
        
        # Set the learning rate scheduler to reduce the learning rate by 30% if the loss plateaus for 3 epochs
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.3, patience=1)
        
        # Begin training, iterates through epochs, and through the whole dataset for every epoch
        for epoch in tqdm(range(self.num_epochs), desc="epochs"):
            
            # For each epoch, do a training and a validation stage
            for phase in ['train', 'val']:
                # Entering training stage
                if phase == 'train':
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
                    tqdm.write('[%d, %5d] train loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 25500))
                        #     # wandb.log({'epoch': epoch+1, 'loss': running_loss/(50 * self.batch_size)})
                
                # Entering validation stage
                else:
                    # Set model to evaluation mode
                    self.model.eval()
                    test_loss = 0.0
                    for i, data in enumerate(testLoader):
                        # Loading in the test data
                        x_data, y_data = data
                        x_data = x_data.to(self.device)
                        y_data = y_data.to(self.device)
                        
                        # Generating predictions based on the current model
                        pred_y = self.model(x_data).to(self.device)
                        
                        # Compute loss
                        loss = criterion(pred_y, y_data)
                        test_loss+=loss.item()
                        
                    # Printing and logging loss so we can keep track of progress
                    tqdm.write('[%d] test loss: %.3f' %
                                (epoch + 1, test_loss / 2250))
                    if(self.log):
                        wandb.log({'epoch': epoch+1, 'test_loss': test_loss/2250})
                    
                    # Check if we need to drop the learning rate
                    scheduler.step(test_loss/2250)
                    
            # Check if we need to save the model
            if(best_loss == -1.0 or test_loss < best_loss):
                best_loss = test_loss
                torch.save(best_loss, "best_loss_" + self.vector + ".pt")
                torch.save(self.model.state_dict(), "model_" + self.vector + ".pt")
        
        # Load our best model and returning it
        self.model.load_state_dict(torch.load("model_" + self.vector + ".pt"))
        
    def train_c_seperate(self, trainLoader, testLoader):
        
        for m in tqdm(range(len(trainLoader)), desc="individual c vector training"):
            self.model = LinearRegression_c_seperate(self.vector)
            self.model.to(self.device)
            # If a previously trained model exists, load the best loss as the saved best model
            try:
                best_loss = torch.load("token_space_models/best_loss_c_" + str(m) + ".pt")
            except:
                best_loss = -1.0
            
            # Set best loss to negative value so it always gets overwritten
            
            # Configure the pytorch objects, loss function (criterion)
            criterion = nn.MSELoss(size_average = False)
            
            # Import gradients to wandb to track loss gradients
            if(self.log):
                wandb.watch(self.model, criterion, log="all")
            
            # Set the optimizer to Stochastic Gradient Descent
            optimizer = torch.optim.SGD(self.model.parameters(), lr = self.lr)
            
            # Set the learning rate scheduler to reduce the learning rate by 30% if the loss plateaus for 3 epochs
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.3, patience=1)
            
            # Begin training, iterates through epochs, and through the whole dataset for every epoch
            for epoch in tqdm(range(self.num_epochs), desc="epochs"):
                
                # For each epoch, do a training and a validation stage
                for phase in ['train', 'val']:
                    # Entering training stage
                    if phase == 'train':
                        self.model.train()
                        
                        # Keep track of running loss for this training epoch
                        running_loss = 0.0
                        for i, data in enumerate(trainLoader[m]):
                            
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
                        tqdm.write('[%d, %5d] train loss: %.3f' %
                            (epoch + 1, i + 1, running_loss / 25500))
                            #     # wandb.log({'epoch': epoch+1, 'loss': running_loss/(50 * self.batch_size)})
                    
                    # Entering validation stage
                    else:
                        # Set model to evaluation mode
                        self.model.eval()
                        test_loss = 0.0
                        for i, data in enumerate(testLoader[m]):
                            # Loading in the test data
                            x_data, y_data = data
                            x_data = x_data.to(self.device)
                            y_data = y_data.to(self.device)
                            
                            # Generating predictions based on the current model
                            pred_y = self.model(x_data).to(self.device)
                            
                            # Compute loss
                            loss = criterion(pred_y, y_data)
                            test_loss+=loss.item()
                            
                        # Printing and logging loss so we can keep track of progress
                        tqdm.write('[%d] test loss: %.3f' %
                                    (epoch + 1, test_loss / 2250))
                        if(self.log and m==0):
                            wandb.log({'epoch': epoch+1, 'test_loss': test_loss/2250})
                        
                        # Check if we need to drop the learning rate
                        scheduler.step(test_loss/2250)
                        
                # Check if we need to save the model
                if(best_loss == -1.0 or test_loss < best_loss):
                    best_loss = test_loss
                    torch.save(best_loss, "token_space_models/best_loss_c_" + str(m) + ".pt")
                    torch.save(self.model.state_dict(), "token_space_models/model_c_" + str(m) + ".pt")
            
            # Load our best model and returning it
            self.model.load_state_dict(torch.load("token_space_models/model_c_" + str(m) + ".pt"))

    #reassemble an output c vector from the individual component models
    def predict_c_seperate(self, testLoader_arr):
        models = []
        for i in tqdm(range(77), desc="loading models"):
            models.append(LinearRegression_c_seperate(self.vector))
            models[i].load_state_dict(torch.load("token_space_models/model_c_" + str(i) + ".pt"))
            models[i].to(self.device)
        x0, y0 = [], []
        for i in tqdm(range(77), desc="loading data"):
            x, y = next(iter(testLoader_arr[i]))
            x0.append(x[2].to(self.device))
            y0.append(y[2].to(self.device))
        
        out = torch.zeros(1, 77, 1024)
        target = torch.zeros(1, 77, 1024)
        for i in tqdm(range(len(x0)), desc="predicting"):
            out[:,i] = models[i](x0[i])
            target[:,i] = y0[i]
        torch.save(out, "output_c_seperate_" + self.vector + ".pt")
        torch.save(y0, "target_c_seperate_" + self.vector + ".pt")
        


def main():
    # vector = "c"
    # VM = VectorMapping(vector)
    # VM.model.to(VM.device)
    # train, test = VM.get_data_c_seperate()
    # VM.predict_c_seperate(test)
    # VM.train_c_seperate(train, test)
    # VM.model.load_state_dict(torch.load("model_" + vector + ".pt"))
    # VM.model.eval()
    # x, y = next(iter(test))
    # x0 = x[2].to(VM.device)
    # y0 = y[2].to(VM.device)
    # out = VM.model(x0)
    # cosSim = nn.CosineSimilarity(dim=0)
    # print(cosSim(out, y0))
    # print(cosSim(torch.randn_like(out), y0))
    # torch.save(out, "output_" + vector + ".pt")
    # torch.save(y0, "target_" + vector + ".pt")
    
    z = torch.load("target_z.pt")
    c = torch.load("target_c.pt")
    img = reconstruct(z, c, 0.99999999)
    print("reconstructed", img)

if __name__ == "__main__":
    main()
