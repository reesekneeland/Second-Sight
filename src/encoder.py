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
#
# 410_model_c_img_0.pt
#
# 411_model_c_text_0.pt
#
# 412_model_z_img_mixer.pt


    
# Pytorch model class for Linear regression layer Neural Network
class LinearRegression(torch.nn.Module):
    def __init__(self, vector, outputSize):
        super(LinearRegression, self).__init__()
        if(vector == "c_prompt"):
            inpSize = 78848
        elif(vector == "c_combined" or vector == "c_img_mixer"):
            inpSize = 3840
        elif(vector == "c_img_mixer_0" or vector=="c_img_0" or vector=="c_text_0"):
            inpSize = 768
        elif(vector == "z" or vector == "z_img_mixer"):
            inpSize = 16384
        elif(vector == "c_img"):
            inpSize = 1536
        self.linear = nn.Linear(inpSize, 10000)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(10000, 12000)
        self.outlayer = nn.Linear(12000, outputSize)
    def forward(self, x):
        y_pred = self.relu(self.linear(x))
        y_pred = self.relu(self.linear2(y_pred))
        y_pred = self.outlayer(y_pred)
        return y_pred
    
# Main Class    
class Encoder():
    def __init__(self, 
                 hashNum,
                 vector, 
                 log, 
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
        self.device      = torch.device(device)
        self.lr          = lr
        self.batch_size  = batch_size
        self.num_epochs  = epochs
        self.num_workers = num_workers
        self.log         = log
        self.parallel    = parallel
        self.outputSize  = 11838
    
        # Initialize the Pytorch model class
        self.model = LinearRegression(self.vector, self.outputSize)
        
        # Configure multi-gpu training
        if(self.parallel):
            self.model = nn.DataParallel(self.model)
        
        # Send model to Pytorch Device 
        self.model.to(self.device)
        
        # Initialize the data loaders
        self.trainloader, self.testloader = get_data_encoder(vector=self.vector, 
                                                             batch_size=self.batch_size, 
                                                             num_workers=self.num_workers)
        
        # Initializes Weights and Biases to keep track of experiments and training runs
        if(self.log):
            wandb.init(
                # set the wandb project where this run will be logged
                project="encoder",
                # track hyperparameters and run metadata
                config={
                "hash": self.hashNum,
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
        # criterion = nn.MSELoss(size_average = False)
        
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
                    loss = compound_loss(pred_y, y_data)
                    
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
                loss = compound_loss(pred_y, y_data)

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


    def predict(self, model):

        prep_path = "/export/raid1/home/kneel027/nsd_local/preprocessed_data/"

        # Save to latent vectors
        out = torch.zeros((73000, 11838))
        print(model)
        os.makedirs("latent_vectors/" + model, exist_ok=True)
        # Load the model into the class to be used for predictions
        if(self.parallel):
            self.model.module.load_state_dict(torch.load("/export/raid1/home/kneel027/Second-Sight/models/" + model, map_location='cuda'))
        else:
            self.model.load_state_dict(torch.load("/export/raid1/home/kneel027/Second-Sight/models/" + model, map_location='cuda'))
        self.model.eval()

        # preprocessed_data_c_img_0 = torch.load(prep_path + "c_img_0/vector_73k.pt")
        # preprocessed_data_c_text_0 = torch.load(prep_path + "c_text_0/vector_73k.pt")
        preprocessed_data_z_img_mixer = torch.load(prep_path + "z_img_mixer/vector_73k.pt")



        for index, data in enumerate(preprocessed_data_z_img_mixer):
            
            # Loading in the data
            x_data = data
            x_data = x_data.to(self.device)
            
            # Generating predictions based on the current model
            pred_y = self.model(x_data).to(self.device)
            # if(torch.max(pred_y) < 0.1):
            #     print(torch.max(pred_y))
            out[index] = pred_y
            
        torch.save(out, "/export/raid1/home/kneel027/Second-Sight/latent_vectors/" + model + "/" + "brain_preds.pt")
        
        return out
    
    def predict_cc3m(self, model):

        prep_path = "/export/raid1/home/kneel027/nsd_local/preprocessed_data/"

        # Save to latent vectors
        print(model)
        os.makedirs("latent_vectors/" + model + "/cc3m_batches/", exist_ok=True)
        # Load the model into the class to be used for predictions
        if(self.parallel):
            self.model.module.load_state_dict(torch.load("/export/raid1/home/kneel027/Second-Sight/models/" + model, map_location='cuda'))
        else:
            self.model.load_state_dict(torch.load("/export/raid1/home/kneel027/Second-Sight/models/" + model, map_location='cuda'))
        self.model.eval()



        for i in tqdm(range(124), desc="predicting brain batches"):
            out = torch.zeros((22735, 11838))
            preprocessed_data_batch = torch.load(prep_path + self.vector + "/cc3m_batches/" + str(i) + ".pt")

            for index, data in enumerate(preprocessed_data_batch):
                
                # Loading in the data
                x_data = data
                x_data = x_data.to(self.device)
                
                # Generating predictions based on the current model
                pred_y = self.model(x_data).to(self.device)
                # if(torch.max(pred_y) < 0.1):
                #     print(torch.max(pred_y))
                out[index] = pred_y
                
            torch.save(out, "/export/raid1/home/kneel027/Second-Sight/latent_vectors/" + model + "/cc3m_batches/" + str(i) + ".pt")
        
        return out
    