import torch
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from pearson import PearsonCorrCoef
from utils import *
import wandb
from tqdm import tqdm
import pickle as pk

# Pytorch model class for Linear regression layer Neural Network
class MLP(torch.nn.Module):
    def __init__(self, vector):
        super(MLP, self).__init__()
        self.vector=vector
        if(vector == "c_img_vd"):
            self.linear = nn.Linear(11838, 35000)
            self.outlayer = nn.Linear(35000, 13875)
        elif(vector == "c_text_vd"):
            self.linear = nn.Linear(11838, 25000)
            self.outlayer = nn.Linear(25000, 13875)
        self.double()
        self.relu = nn.ReLU()
        
    def forward(self, x):
        if(self.vector == "c_img_vd" or self.vector=="c_text_vd"):
            y_pred = self.relu(self.linear(x))
            y_pred = self.outlayer(y_pred)
        return y_pred

    
# Main Class    
class Decoder_PCA():
    def __init__(self, 
                 hashNum,
                 vector, 
                 log, 
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
        self.pca = pk.load(open("masks/pca_" +self.vector + ".pkl",'rb'))

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
                project="decoder",
                # track hyperparameters and run metadata
                config={
                "hash": self.hashNum,
                "architecture": "MLP",
                # "architecture": "2 Convolutional Layers",
                "vector": self.vector,
                "dataset": "Z scored",
                "epochs": self.num_epochs,
                "learning_rate": self.lr,
                "batch_size:": self.batch_size,
                "num_workers": self.num_workers
                }
            )
        # print("INIT ")
        # print_gpu_utilization()
    

    def train(self):
        self.trainLoader, self.valLoader, _, _ = load_nsd(vector=self.vector, 
                                                        batch_size=self.batch_size, 
                                                        num_workers=self.num_workers, 
                                                        loader=True,
                                                        pca=True)
        # Set best loss to negative value so it always gets overwritten
        best_loss = -1.0
        loss_counter = 0
        
        # Configure the pytorch objects, loss function (criterion)
        criterion = nn.MSELoss(reduction='sum')
        # Set the optimizer to Adam
        optimizer = Adam(self.model.parameters(), lr = self.lr)
        # Begin training, iterates through epochs, and through the whole dataset for every epoch
        for epoch in tqdm(range(self.num_epochs), desc="epochs"):
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
                x_data = x_data.to(self.device, torch.float64)
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
                x_data, y_data = data
                x_data = x_data.to(self.device, torch.float64)
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
                wandb.log({'epoch': epoch+1, 'test_loss': test_loss})
                    
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
        self.model.load_state_dict(torch.load("models/" + self.hashNum + "_model_" + self.vector + ".pt", map_location=self.device))
        self.model.eval()
        self.model.to(self.device)
        out = self.model(x.to(self.device, torch.float64)).cpu.numpy() #.to(torch.float16)
        out = torch.from_numpy(self.pca.inverse_transform(out))
        return out.to(torch.float32)
    
    def benchmark(self, average=True):
        _, _, _, _, _, _, _, target, _, _ = load_nsd(vector=self.vector, 
                                                batch_size=self.batch_size, 
                                                num_workers=self.num_workers, 
                                                loader=False,
                                                average=average,
                                                pca=False)
        _, _, _, x_test, _, _, _, y_test, _, _ = load_nsd(vector=self.vector, 
                                                batch_size=self.batch_size, 
                                                num_workers=self.num_workers, 
                                                loader=False,
                                                average=average,
                                                pca=True)
        outSize = len(x_test)
        if(self.vector=="c_img_0" or self.vector=="c_text_0"):
            vecSize = 768
        elif(self.vector == "z" or self.vector == "z_img_mixer"):
            vecSize = 16384
        elif(self.vector == "c_img_vd"):
            vecSize = 13875
        elif(self.vector == "c_text_vd"):
            vecSize = 13875
        out = torch.zeros((outSize, vecSize))
        # target = torch.zeros((outSize, vecSize))
        self.model.load_state_dict(torch.load("models/" + self.hashNum + "_model_" + self.vector + ".pt"))
        self.model.eval()

        loss = 0
        pearson_loss = 0
        
        criterion = nn.MSELoss()
        
        
            # x_test = nn.functional.pad(input=x_test, pad=(0, 2, 0, 0), mode='constant', value=0)
        x_test = x_test.to(self.device, torch.float64)
            # Generating predictions based on the current model
                # print(torch.sum(torch.isnan(y_test)))
        pred_y = self.model(x_test).cpu()
            # print(torch.sum(torch.isnan(pred_y)))
        
        out = pred_y
        loss += criterion(pred_y.to(self.device), y_test.to(self.device))
                # print(pred_y, y_test)
        
        PeC = PearsonCorrCoef(num_outputs=out.shape[0])
        pearson_loss =torch.mean(PeC(out.moveaxis(0,1), y_test.moveaxis(0,1)))
        out = torch.from_numpy(self.pca.inverse_transform(out.detach().numpy()))
        pearson_loss_T = torch.mean(PeC(out.moveaxis(0,1), target.moveaxis(0,1)))
            #print(pearson_corrcoef(out[index], target[index]))
            
        # print(y_test.shape[0], target.shape[0])
        # print(y_test == target)
        loss = loss / len(target)
        
        out = out.detach()
        target = target.detach()
        PeC = PearsonCorrCoef()
        r = []
        for p in range(out.shape[1]):
            
            # Correlation across voxels for a sample (Taking a column)
            r.append(PeC(out[:,p], target[:,p]))
        r = np.array(r)
        
        print("Vector Correlation_PCA: ", float(pearson_loss))
        print("Vector Correlation: ", float(pearson_loss_T))
        print("Mean Pearson: ", np.mean(r))
        print("Loss: ", float(loss))
        plt.hist(r, bins=40, log=True)
        plt.savefig("charts/" + self.hashNum + "_" + self.vector + "_pearson_histogram_decoder.png")