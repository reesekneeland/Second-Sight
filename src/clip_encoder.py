from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from utils import *
import wandb
import yaml
from tqdm import tqdm
from torchmetrics import PearsonCorrCoef
from model_zoo import c_enc

    
# Main Class    
class CLIP_Encoder():
    def __init__(self,
                 inference=False, 
                 subject=1,
                 lr=0.00001,
                 batch_size=750,
                 device="cuda",
                 num_workers=4,
                 epochs=200
                 ):
    
        self.subject = subject
        self.device = torch.device(device)
        if inference:
            self.log = False
        else:
            self.log = True
            self.lr = lr
            self.batch_size = batch_size
            self.num_epochs = epochs
            self.num_workers = num_workers
            # Initialize the data loaders
            self.trainLoader, self.valLoader, _ = load_nsd(vector="c_img_uc", 
                                                            batch_size=self.batch_size, 
                                                            num_workers=self.num_workers, 
                                                            loader=True,
                                                            average=False,
                                                            subject=self.subject,
                                                            big=True)
             # Initializes Weights and Biases to keep track of experiments and training runs
            if(self.log):
                wandb.init(
                    # set the wandb project where this run will be logged
                    project="CLIP_Encoder",
                    # track hyperparameters and run metadata
                    config={
                    "subject": self.subject,
                    "vector": "c_img_uc",
                    "epochs": self.num_epochs,
                    "learning_rate": self.lr,
                    "batch_size:": self.batch_size,
                    "num_workers": self.num_workers
                    }
                )
        subject_sizes = [0, 15724, 14278, 0, 0, 13039, 0, 12682]
        # Initialize the Pytorch model class
        self.model = C_Enc(subject_sizes[self.subject])
        # Send model to Pytorch Device 
        self.model.to(self.device)
    

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
                
                # x_data = Clip vector Data
                # y_data = Brain Data
                y_data, x_data = data
                
                # Moving the tensors to the GPU
                x_data = x_data.to(self.device)
                y_data = y_data.to(self.device)
                
                # Forward pass: Compute predicted y by passing x to the model
                # Train the x data in the model to get the predicted y value. 
                pred_y = self.model(x_data).to(self.device)
                
                # Compute the loss between the predicted y and the y data. 
                loss = criterion(pred_y, y_data)
                loss.backward()
                optimizer.step()
                
                # Zero gradients in the optimizer
                optimizer.zero_grad()
                
                # Add up the loss for this training round
                running_loss += loss.item()
                
            tqdm.write('[%d] train loss: %.8f' %
                (epoch + 1, running_loss /len(self.trainLoader)))
                #     # wandb.log({'epoch': epoch+1, 'loss': running_loss/(50 * self.batch_size)})
            
                
            # Entering validation stage
            # Set model to evaluation mode
            self.model.eval()
            running_test_loss = 0.0
            for i, data in enumerate(self.valLoader):
                
                # Loading in the test data
                
                # x_data = Clip vector Data
                # y_data = Brain Data
                y_data, x_data = data
                x_data = x_data.to(self.device)
                y_data = y_data.to(self.device, torch.float64)
                
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
                torch.save(self.model.state_dict(), "models/sub0{subject}_clip_encoder.pt".format(subject=self.subject))
                loss_counter = 0
            else:
                loss_counter += 1
                tqdm.write("loss counter: {}".format(loss_counter))
                if(loss_counter >= 5):
                    break
        if(self.log):
                wandb.finish()
                
    def predict(self, x, mask=None):
        self.model.load_state_dict(torch.load("models/sub0{subject}_clip_encoder.pt".format(subject=self.subject), map_location=self.device))
        self.model.eval()
        self.model.to(self.device)
        out = self.model(x.to(self.device, torch.float32))
        if mask:
            out = out[mask]
        return out.to(self.device)
        
        
    def benchmark(self, average=True):
            
        # y_test = Brain data
        # x_test = clip data
        _, _, y_test, _, _, x_test, _ = load_nsd(vector=self.vector, 
                                                loader=False,
                                                average=average,
                                                subject=self.subject,
                                                big=True)
        self.model.load_state_dict(torch.load("models/sub0{subject}_clip_encoder.pt".format(subject=self.subject)))
        self.model.eval()

        criterion = nn.MSELoss()
        PeC = PearsonCorrCoef(num_outputs=y_test.shape[0]).to(self.device)
        
        x_test = x_test.to(self.device)
        y_test = y_test.to(self.device)
        
        pred_y = self.model(x_test)
        
        loss = criterion(pred_y, y_test)
              
        pearson = torch.mean(PeC(pred_y.moveaxis(0,1), y_test.moveaxis(0,1)))
        
        pred_y = pred_y.cpu().detach()
        y_test = y_test.cpu().detach()
        PeC = PearsonCorrCoef()
        r = []
        for voxel in range(pred_y.shape[1]):
            # Correlation across voxels for a sample (Taking a column)
            r.append(PeC(pred_y[:,voxel], y_test[:,voxel]))
        r = np.array(r)
        
        print("Model: CLIP Encoder, Subject: {}, Averaged: {}".format(self.subject, average))
        print("Vector Correlation: ", float(pearson))
        print("Mean Pearson: ", np.mean(r))
        print("Loss: ", float(loss))
        plt.hist(r, bins=50, log=True)
        plt.savefig("data/charts/subject{}_clip_encoder_pearson_correlation.png".format(self.subject))
    
    
    
    
    