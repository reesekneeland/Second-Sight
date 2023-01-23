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
from encoder import Encoder


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
            self.linear = nn.Linear(78848, 16000)
        elif(vector == "z"):
            self.linear = nn.Linear(16384,  16000)
    
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

        # Initializes the pytorch model class
        # self.model = model = Linear5Layer(self.vector)
        self.model = LinearRegression(self.vector).to(self.device)
        
        
        # Set the parameters for pytorch model training
        self.lr = 0.1
        self.batch_size = 750
        self.num_epochs = 100
        self.num_workers = 4
        self.log = True
        
        # Initializes Weights and Biases to keep track of experiments and training runs
        if(self.log):
            wandb.init(
                # set the wandb project where this run will be logged
                project="voxelSelection clip encoder to brain vector",
                
                # track hyperparameters and run metadata
                config={
                # "architecture": "Linear Regression",
                # "architecture": "Linear 2 Layer",
                # "architecture": "Linear 3 Layer",
                # "architecture": "Linear 4 Layer",
                "architecture": "Linear regression",
                "vector": self.vector,
                "dataset": "Entire visual cortex",
                "epochs": self.num_epochs,
                "learning_rate": self.lr,
                "batch_size:": self.batch_size,
                "num_workers": self.num_workers
                }
            )
    
    def load_data_masked(vector, only_test=False):
        
        # 750 trails per session 
        # 40 sessions per subject
        # initialize some empty tensors to allocate memory
        
        
        if(vector == "c"):
            y_train = torch.empty((25500, 77, 1024))
            y_test  = torch.empty((2250, 77, 1024))
        elif(vector == "z"):
            y_train = torch.empty((25500, 4, 64, 64))
            y_test  = torch.empty((2250, 4, 64, 64))
        
        
        # 34 * 750 = 25500
        x_train = torch.empty((25500, 42, 22, 27))
        
        # 3 * 750 = 2250
        x_test  = torch.empty((2250, 42, 22, 27))
        
        #grabbing voxel mask for subject 1
        voxel_mask = voxel_data_dict['voxel_mask']["1"]
        voxel_mask_reshape = voxel_mask.reshape((81, 104, 83, 1))

        slices = get_slices(voxel_mask_reshape)

        # Checks if we are only loading the test data so we don't have to load all the training betas
        if(not only_test):
            
            # Loads the full collection of beta sessions for subject 1
            for i in tqdm(range(1,35), desc="Loading Training Voxels"):
                beta = nsda.read_betas(subject='subj01', 
                                    session_index=i, 
                                    trial_index=[], # Empty list as index means get all 750 scans for this session
                                    data_type='betas_fithrf_GLMdenoise_RR',
                                    data_format='func1pt8mm')
                roi_beta = np.where((voxel_mask_reshape), beta, 0)
                beta_trimmed = roi_beta[slices] 
                beta_trimmed = np.moveaxis(beta_trimmed, -1, 0)
                x_train[(i-1)*750:(i-1)*750+750] = torch.from_numpy(beta_trimmed)
        
        for i in tqdm(range(35,38), desc="Loading Test Voxels"):
            
            # Loads the test betas and puts it into a tensor
            test_betas = nsda.read_betas(subject='subj01', 
                                        session_index=i, 
                                        trial_index=[], # Empty list as index means get all for this session
                                        data_type='betas_fithrf_GLMdenoise_RR',
                                        data_format='func1pt8mm')
            roi_beta = np.where((voxel_mask_reshape), test_betas, 0)
            beta_trimmed = roi_beta[slices] 
            beta_trimmed = np.moveaxis(beta_trimmed, -1, 0)
            x_test[(i-35)*750:(i-35)*750+750] = torch.from_numpy(beta_trimmed)

        # Loading the description object for subejct1
        subj1y = nsda.stim_descriptions[nsda.stim_descriptions['subject1'] != 0]

        for i in tqdm(range(0,25500), desc="Loading Training Vectors"):
            # Flexible to both Z and C tensors depending on class configuration
            index = int(subj1y.loc[(subj1y['subject1_rep0'] == i+1) | (subj1y['subject1_rep1'] == i+1) | (subj1y['subject1_rep2'] == i+1)].nsdId)
            y_train[i] = torch.load("/home/naxos2-raid25/kneel027/home/kneel027/nsd_local/nsddata_stimuli/tensors/" + vector + "/" + str(index) + ".pt")

        for i in tqdm(range(0,2250), desc="Loading Test Vectors"):
            index = int(subj1y.loc[(subj1y['subject1_rep0'] == 25501 + i) | (subj1y['subject1_rep1'] == 25501 + i) | (subj1y['subject1_rep2'] == 25501 + i)].nsdId)
            y_test[i] = torch.load("/home/naxos2-raid25/kneel027/home/kneel027/nsd_local/nsddata_stimuli/tensors/" + vector + "/" + str(index) + ".pt")

        if(vector == "c"):
            y_train = y_train.reshape((25500, 78848))
            y_test  = y_test.reshape((2250, 78848))
        elif(vector == "z"):
            y_train = y_train.reshape((25500, 16384))
            y_test  = y_test.reshape((2250, 16384))
            
        x_train = x_train.reshape((25500, 1, 42, 22, 27))
        x_test  = x_test.reshape((2250, 1, 42, 22, 27))

        print("3D STATS PRENORM", torch.max(x_train), torch.var(x_train))
        x_train_mean, x_train_std = x_train.mean(), x_train.std()
        x_test_mean, x_test_std = x_test.mean(), x_test.std()
        x_train = (x_train - x_train_mean) / x_train_std
        x_test = (x_test - x_test_mean) / x_test_std

        print("3D STATS NORMALIZED", torch.max(x_train), torch.var(x_train))
        return x_train, x_test, y_train, y_test
    
    # Loads the data into an array of DataLoaders for each seperate C vector
    def get_data_c_seperate(self):
        x, x_test, y, y_test = self.load_data_c_seperate()
        # Loads the raw tensors into Dataset objects
        trainset = torch.utils.data.TensorDataset(x, y) #.type(torch.LongTensor)
        testset = torch.utils.data.TensorDataset(x_test, y_test) #.type(torch.LongTensor)
        
        # Loads the Dataset into a DataLoader
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        return trainloader, testloader
    
        
    def train_c_seperate(self, trainLoader, testLoader):

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
        
        # Set the learning rate scheduler to reduce the learning rate by 30% if the loss plateaus for 3 epochs
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=1)
        
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
                (epoch + 1, i + 1, running_loss /self.batch_size))
                    # wandb.log({'epoch': epoch+1, 'loss': running_loss/(50 * self.batch_size)})

        # Entering validation stage
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
            tqdm.write('[%d] test loss: %.8f' %
                        (epoch + 1, test_loss /self.batch_size))
            if(self.log):
                wandb.log({'epoch': epoch+1, 'test_loss': test_loss/self.batch_size})
            
            # Check if we need to drop the learning rate
            scheduler.step(test_loss/self.batch_size)
                    
            # Check if we need to save the model
            if(best_loss == -1.0 or test_loss < best_loss):
                best_loss = test_loss
                torch.save(self.model.state_dict(), "scalar_space_models/model_z_100.pt")
                loss_counter = 0
            else:
                loss_counter += 1
                tqdm.write("loss counter: " + str(loss_counter))
                if(loss_counter >= 3):
                    break
        
        # Load our best model and returning it
        self.model.load_state_dict(torch.load("scalar_space_models/model_z_100.pt"))


    #reassemble an output c vector from the individual component models
    def predict_c_seperate(self, testLoader):
        out = torch.zeros((2250,100))
        target = torch.zeros((2250, 100))
        self.model.load_state_dict(torch.load("scalar_space_models/model_z_100.pt"))
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
        for p in range(100):
            x_bar = torch.mean(out[:,p])
            y_bar = torch.mean(target[:,p])
            numerator = torch.sum((out[:,p]-x_bar)*(target[:,p]-y_bar))
            denominator = torch.sqrt(torch.sum((out[:,p]-x_bar)**2)*torch.sum((target[:,p]-y_bar)**2))
            pearson = numerator/denominator
            r.append(pearson)
        # r = []
        # for p in range(100):
        #     r.append(pearson_corrcoef(out[:,p], target[:,p]))
        print(np.mean(r))
            
        plt.hist(r, bins=40)
        plt.savefig("pearson_scalar_100_histogram_z.png")
        # plt.plot(r)
        # plt.savefig("pearson_scalar_100_original2.png")
        
        
        torch.save(out, "output_z_scalar.pt")
        torch.save(target, "target_z_scalar.pt")
        


def main():
    vector = "z"
    VM = VectorMapping(vector)
    train, test = VM.get_data_c_seperate()
    # VM.train_c_seperate(train, test)
    VM.predict_c_seperate(test)

if __name__ == "__main__":
    main()
