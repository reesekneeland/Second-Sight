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
from torchmetrics.functional import pearson_corrcoef
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
class LinearRegression(torch.nn.Module):
    def __init__(self, vector, inpSize):
        super(LinearRegression, self).__init__()
        if(vector == "c_prompt"):
            outSize = 78848
        elif(vector == "c_combined" or vector == "c_img_mixer"):
            outSize = 3840
        elif(vector == "c_img_mixer_0" or vector=="c_img_0" or vector=="c_text_0"):
            outSize = 768
        elif(vector == "z" or vector == "z_img_mixer"):
            outSize = 16384
        elif(vector == "c_img"):
            outSize = 1536
        self.linear = nn.Linear(inpSize, 10000)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(10000, 12000)
        self.outlayer = nn.Linear(12000, outSize)
    def forward(self, x):
        y_pred = self.relu(self.linear(x))
        y_pred = self.relu(self.linear2(y_pred))
        y_pred = self.outlayer(y_pred)
        return y_pred
    
# Main Class    
class Decoder():
    def __init__(self, 
                 hashNum,
                 vector, 
                 log, 
                 lr=0.00001,
                 threshold=0.06734,
                 inpSize = 7372,
                 batch_size=750,
                 parallel=True,
                 device="cuda",
                 num_workers=16,
                 epochs=200
                 ):

        # Set the parameters for pytorch model training
        self.hashNum = hashNum
        self.vector = vector
        self.threshold = threshold
        self.device = torch.device(device)
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = epochs
        self.num_workers = num_workers
        self.log = log
        self.parallel = parallel
        self.inpSize = inpSize

        # Initialize the Pytorch model class
        self.model = LinearRegression(self.vector, self.inpSize)
        
        # Configure multi-gpu training
        if(self.parallel):
            self.model = nn.DataParallel(self.model)
        
        # Send model to Pytorch Device 
        self.model.to(self.device)
        
        # Initialize the data loaders
        self.trainloader, self.testloader = load_data(vector=self.vector, 
                                                      batch_size=self.batch_size, 
                                                      num_workers=self.num_workers, 
                                                      loader=True)
        
        # Initializes Weights and Biases to keep track of experiments and training runs
        if(self.log):
            wandb.init(
                # set the wandb project where this run will be logged
                project="decoder",
                # track hyperparameters and run metadata
                config={
                "hash": self.hashNum,
                "threshold": self.threshold,
                "architecture": "Linear Regression",
                # "architecture": "2 Convolutional Layers",
                "vector": self.vector,
                "dataset": "custom masked positive pearson correlation on c_combined data",
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
        criterion = nn.MSELoss(size_average = False)
        
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
            for i, data in enumerate(self.trainloader):
                
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
                    # loss = compound_loss(pred_y, y_data)
                    loss = criterion(pred_y, y_data)
                    
                    # Perform weight updating
                    loss.backward()
                    optimizer.step()

                # tqdm.write('train loss: %.3f' %(loss.item()))
                # Add up the loss for this training round
                running_loss += loss.item()
            tqdm.write('[%d] train loss: %.8f' %
                (epoch + 1, running_loss /len(self.trainloader)))
                #     # wandb.log({'epoch': epoch+1, 'loss': running_loss/(50 * self.batch_size)})
                
            # Entering validation stage
            # Set model to evaluation mode
            self.model.eval()
            running_test_loss = 0.0
            for i, data in enumerate(self.testloader):
                
                # Loading in the test data
                x_data, y_data = data
                x_data = x_data.to(self.device)
                y_data = y_data.to(self.device)
                
                # Generating predictions based on the current model
                pred_y = self.model(x_data).to(self.device)
                
                # Compute the test loss 
                # loss = compound_loss(pred_y, y_data)
                loss = criterion(pred_y, y_data)

                running_test_loss += loss.item()
                
            test_loss = running_test_loss / len(self.testloader)
                
            # Printing and logging loss so we can keep track of progress
            tqdm.write('[%d] test loss: %.8f' %
                        (epoch + 1, test_loss))
            if(self.log):
                wandb.log({'epoch': epoch+1, 'test_loss': test_loss})
                    
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
                if(loss_counter >= 5):
                    break
                
        # Load our best model into the class to be used for predictions
        if(self.parallel):
            self.model.module.load_state_dict(torch.load("/export/raid1/home/kneel027/Second-Sight/models/" + self.hashNum + "_model_" + self.vector + ".pt", map_location='cuda'))
        else:
            self.model.load_state_dict(torch.load("/export/raid1/home/kneel027/Second-Sight/models/" + self.hashNum + "_model_" + self.vector + ".pt", map_location='cuda'))


    # def predict(self, model, indices=[0]):
    #     print(model)
    #     os.makedirs("latent_vectors/" + model, exist_ok=True)
    #     # Load the model into the class to be used for predictions
    #     if(self.parallel):
    #         self.model.module.load_state_dict(torch.load("/export/raid1/home/kneel027/Second-Sight/models/" + model, map_location='cuda'))
    #     else:
    #         self.model.load_state_dict(torch.load("/export/raid1/home/kneel027/Second-Sight/models/" + model, map_location='cuda'))
    #     self.model.eval()
    #     outputs, targets = [], []
    #     x, y = next(iter(self.testloader))
    #     for x in self.testloader:
    #         for i in indices:
    #             # Loading in the test data
    #             x_data = x[i]
    #             y_data = y[i]
    #             x_data = x_data[None,:].to(self.device)
    #             y_data = y_data[None,:].to(self.device)
    #             # Generating predictions based on the current model
    #             pred_y = self.model(x_data).to(self.device)
    #             outputs.append(pred_y)
    #             targets.append(y_data)
    #             torch.save(pred_y, "/export/raid1/home/kneel027/Second-Sight/latent_vectors/" + model + "/" + "output_" + str(i) + "_" + self.vector + ".pt")
    #             torch.save(y_data, "/export/raid1/home/kneel027/Second-Sight/latent_vectors/" + model + "/" + "target_" + str(i) + "_" + self.vector + ".pt")
            
                
            
    #     # Pearson correlation
    #     out = torch.Tensor(outputs)
    #     target = torch.Tensor(target)
    #     r = []
    #     for p in range(out[0].shape[1]):
    #         r.append(pearson_corrcoef(out[:,p], target[:,p]))
    #     r = np.array(r)
    #     print(np.mean(r))
    #     plt.hist(r, bins=40, log=True)
    #     plt.savefig("/export/raid1/home/kneel027/Second-Sight/charts/" + self.hashNum + "_" + self.vector + "_pearson_histogram_log_applied_decoder.png")
    #     return outputs, targets
    
    def predict(self, model):
        
        if(self.vector == "c_prompt"):
            outSize = 78848
        elif(self.vector == "c_combined" or self.vector == "c_img_mixer"):
            outSize = 3840
        elif(self.vector == "c_img_mixer_0" or self.vector=="c_img_0" or self.vector=="c_text_0"):
            outSize = 768
        elif(self.vector == "z" or self.vector == "z_img_mixer"):
            outSize = 16384
        elif(self.vector == "c_img"):
            outSize = 1536
        
        out = torch.zeros((2770, outSize))
        target = torch.zeros((2770, outSize))
        self.model.load_state_dict(torch.load("/export/raid1/home/kneel027/Second-Sight/models/" + self.hashNum + "_model_" + self.vector + ".pt"))
        self.model.eval()
        self.model.to(self.device)
        
        _, _, x_test, _, _, y_test = load_data(vector=self.vector, 
                                                batch_size=self.batch_size, 
                                                num_workers=self.num_workers, 
                                                loader=False)
        
        y_test = y_test.to(self.device)
        x_test = x_test.to(self.device)

        
        loss = 0
        pearson_corrcoef_loss = 0
        criterion = nn.MSELoss(size_average = False)
        
        for index in range(y_test.shape[0]):
            
            # Generating predictions based on the current model
            pred_y = self.model(x_test[index]).to(self.device)
            loss += criterion(pred_y, y_test[index])
            
            out[index] = pred_y
            target[index] = y_test[index]
            
            pearson_corrcoef_loss += pearson_corrcoef(out[index], target[index])
            
        loss = loss / y_test.shape[0]
        
        # Vector correlation for that trial row wise
        pearson_corrcoef_loss = pearson_corrcoef_loss / y_test.shape[0]
        
        out = out.detach()
        target = target.detach()
        
        r = []
        for p in range(out.shape[1]):
            r.append(pearson_corrcoef(out[:,p], target[:,p]))
        r = np.array(r)
        
        print("Vector Correlation: ", float(pearson_corrcoef_loss)) 
        print("Mean Pearson: ", np.mean(r))
        print("Loss: ", float(loss))
        plt.hist(r, bins=40, log=True)
        plt.savefig("/export/raid1/home/kneel027/Second-Sight/charts/" + self.hashNum + "_" + self.vector + "_pearson_histogram_encoder.png")
        
        return out, target
    
    # def predict(self, model):
    #     if(self.vector == "c_prompt"):
    #         outSize = 78848
    #     elif(self.vector == "c_combined" or self.vector == "c_img_mixer"):
    #         outSize = 3840
    #     elif(self.vector == "c_img_mixer_0" or self.vector=="c_img_0" or self.vector=="c_text_0"):
    #         outSize = 768
    #     elif(self.vector == "z" or self.vector == "z_img_mixer"):
    #         outSize = 16384
    #     elif(self.vector == "c_img"):
    #         outSize = 1536
    #     out = torch.zeros((2250, outSize))
    #     target = torch.zeros((2250, outSize))
    #     print(model)
    #     os.makedirs("latent_vectors/" + model, exist_ok=True)
    #     # Load the model into the class to be used for predictions
    #     if(self.parallel):
    #         self.model.module.load_state_dict(torch.load("/export/raid1/home/kneel027/Second-Sight/models/" + model, map_location='cuda'))
    #     else:
    #         self.model.load_state_dict(torch.load("/export/raid1/home/kneel027/Second-Sight/models/" + model, map_location='cuda'))
    #     self.model.eval()

    #     for index, data in enumerate(self.testloader):
            
    #         # Loading in the test data
    #         x_data, y_data = data
    #         x_data = x_data.to(self.device)
    #         y_data = y_data.to(self.device)
    #         # Generating predictions based on the current model
    #         pred_y = self.model(x_data).to(self.device)
    #         out[index*self.batch_size:index*self.batch_size+self.batch_size] = pred_y
    #         target[index*self.batch_size:index*self.batch_size+self.batch_size] = y_data
        
    #     # out = out.detach()
    #     # target = target.detach()
        
    
    #     # Pearson correlation
    #     # r = []
    #     # for p in range(out.shape[1]):
    #     #     r.append(pearson_corrcoef(out[:,p], target[:,p]))
    #     # r = np.array(r)
    #     # print(np.mean(r))
            
    #     # plt.hist(r, bins=40, log=True)
    #     # plt.savefig("/export/raid1/home/kneel027/Second-Sight/charts/" + self.hashNum + "_" + self.vector + "2voxels_pearson_histogram_log_applied_decoder.png")
        
    #     return out, target
        
