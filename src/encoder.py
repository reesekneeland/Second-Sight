# Only GPU's in use
import os
from torch.autograd import Variable
from torch.optim import Adam
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
from pearson import PearsonCorrCoef
    
# Pytorch model class for MLP Neural Network
class MLP(torch.nn.Module):
    def __init__(self, vector):
        super(MLP, self).__init__()
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
        self.outlayer = nn.Linear(12000, 11838)
        
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
                project="encoder",
                # track hyperparameters and run metadata
                config={
                "hash": self.hashNum,
                "architecture": "MLP, pearson loss whole",
                "vector": self.vector,
                "dataset": "Whole region visual cortex",
                "epochs": self.num_epochs,
                "learning_rate": self.lr,
                "batch_size:": self.batch_size,
                "num_workers": self.num_workers
                }
            )
    

    def train(self):
        self.trainLoader, self.valLoader, _, _, _ = load_nsd(vector=self.vector, 
                                                                    batch_size=self.batch_size, 
                                                                    num_workers=self.num_workers, 
                                                                    loader=True,
                                                                    average=False)
        # Set best loss to negative value so it always gets overwritten
        best_loss = -1.0
        loss_counter = 0
        
        # Configure the pytorch objects, loss function (criterion)
        criterion = nn.MSELoss(reduction='sum')
        
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
                # x_data = Clip vector Data
                # y_data = Brain Data
                y_data, x_data = data
                
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
                y_data, x_data = data
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
        
        # out = torch.zeros((x.shape[0],11838))
        self.model.load_state_dict(torch.load("models/" + self.hashNum + "_model_" + self.vector + ".pt"))
        self.model.eval()
        self.model.to(self.device)
        out = self.model(x.to(self.device))
        return out
                
    
    def benchmark(self):
        _, _, _, _, self.testLoader = load_nsd(vector=self.vector, 
                                                batch_size=self.batch_size, 
                                                num_workers=self.num_workers, 
                                                loader=True,
                                                average=True)
        out = torch.zeros((1656,11838))
        target = torch.zeros((1656, 11838))
        self.model.load_state_dict(torch.load("models/" + self.hashNum + "_model_" + self.vector + ".pt"))
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
            pearson_loss += torch.sum(PeC(pred_y, y_test))
            #print(pearson_corrcoef(out[index], target[index]))
            
            
        loss = loss / len(self.testLoader)
        
        # Vector correlation for that trial row wise
        pearson_loss = pearson_loss / len(self.testLoader.dataset)
        
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
        plt.savefig("charts/" + self.hashNum + "_" + self.vector + "_pearson_histogram_encoder.png")
        


    def predict_73K_coco(self, model, predict=True):
        
        if(predict):
            prep_path = "/export/raid1/home/kneel027/nsd_local/preprocessed_data/"

            # Save to latent vectors
            out = torch.zeros((73000, 11838))
            print(model)
            os.makedirs("latent_vectors/" + model, exist_ok=True)
            # Load the model into the class to be used for predictions
            self.model.load_state_dict(torch.load("/models/" + model, map_location=self.device))
            self.model.eval()

            # preprocessed_data = torch.zeros()
            # if(model == "z_img_mixer"):
            #     preprocessed_data = torch.load(prep_path + "z_img_mixer/vector.pt")
                
            # elif(model == "c_text_0"):
            #     preprocessed_data = torch.load(prep_path + "c_text_0/vector.pt")
                
            # elif(model == "c_img_0"):
            preprocessed_data = torch.load(prep_path + self.vector + "/vector_73k.pt")
            print(preprocessed_data.shape)

            for index, data in tqdm(enumerate(preprocessed_data), desc="predicting coco data"):
                
                # Loading in the data
                x_data = data
                x_data = x_data.to(self.device)
                
                # Generating predictions based on the current model
                pred_y = self.model(x_data).to(self.device)
                out[index] = pred_y
                
            torch.save(out, "latent_vectors/" + model + "/" + "coco_brain_preds.pt")
        
        
    
    def predict_cc3m(self, model):

        prep_path = "/export/raid1/home/kneel027/nsd_local/preprocessed_data/"

        # Save to latent vectors
        print(model)
        os.makedirs("latent_vectors/" + model + "/cc3m_batches/", exist_ok=True)
        # Load the model into the class to be used for predictions
        self.model.load_state_dict(torch.load("models/" + model, map_location=self.device))
        self.model.eval()



        for i in tqdm(range(124), desc="predicting brain batches"):
            out = torch.zeros((22735, 11838))
            preprocessed_data_batch = torch.load(prep_path + self.vector + "/cc3m_batches/" + str(i) + ".pt")

            # for index in range(len(preprocessed_data_batch)):
                
                # Loading in the data
            x_data = preprocessed_data_batch
            x_data = x_data.to(self.device)
                
                # Generating predictions based on the current model
            pred_y = self.model(x_data).to(self.device)
            # if(torch.max(pred_y) < 0.1):
            #     print(torch.max(pred_y))
            out = pred_y.to(self.device)
                
            torch.save(out, "latent_vectors/" + model + "/cc3m_batches/" + str(i) + ".pt")
        
        return out
    