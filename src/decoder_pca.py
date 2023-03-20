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
    def __init__(self, vector, numLayers):
        super(MLP, self,).__init__()
        # assert(vector == "c_img_vd" or vector=="c_text_vd")
        self.vector = vector
        if(self.vector == "c_img_vd"):
            self.linear = nn.Linear(11838, 15000)
            self.linear2 = nn.Linear(15000, 15000)
            self.linear3 = nn.Linear(15000, 15000)
        if(self.vector == "c_text_vd"):
            self.linear = nn.Linear(11838, 15000)
            self.linear2 = nn.Linear(15000, 15000)
        self.outlayer = nn.Linear(15000, 10000)
        # layers = [nn.Linear(11838, 15000),
        #           nn.ReLU()]
        # for i in range(numLayers-1):
        #     layers.append(nn.Linear(15000, 15000))
        #     layers.append(nn.ReLU())
        # layers.append(nn.Linear(15000, 10000))
        
        # self.layers = nn.Sequential(*layers)
        self.double()
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # y_pred = self.layers(x)
        if(self.vector == "c_img_vd"):
            y_pred = self.relu(self.linear(x))
            y_pred = self.relu(self.linear2(y_pred))
            y_pred = self.relu(self.linear3(y_pred))
            y_pred = self.outlayer(y_pred)
        if(self.vector=="c_text_vd"):
            y_pred = self.relu(self.linear(x))
            y_pred = self.relu(self.linear2(y_pred))
            y_pred = self.outlayer(y_pred)
        return y_pred

    
# Main Class    
class Decoder_PCA():
    def __init__(self, 
                 hashNum,
                 vector, 
                 log, 
                 lr=0.00001,
                 numLayers=4,
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
        self.pca = pk.load(open("masks/pca_" +self.vector + "_10k.pkl",'rb'))

        # Initialize the Pytorch model class
        self.model = MLP(self.vector, numLayers)

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
                "vector": self.vector,
                "dataset": "Z scored",
                "epochs": self.num_epochs,
                "learning_rate": self.lr,
                "batch_size:": self.batch_size,
                "num_workers": self.num_workers
                }
            )
    

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
        out = self.model(x.to(self.device, torch.float64)).cpu().detach().numpy() #.to(torch.float16)
        out = torch.from_numpy(self.pca.inverse_transform(out))
        return out.to(torch.float32)
    
    def benchmark(self, average=True):
        _, _, _, _, _, _, _, y_test, _, _ = load_nsd(vector=self.vector, 
                                                batch_size=self.batch_size, 
                                                num_workers=self.num_workers, 
                                                loader=False,
                                                average=average,
                                                pca=False)
        _, _, _, x_test_pca, _, _, _, y_test_pca, _, _ = load_nsd(vector=self.vector, 
                                                batch_size=self.batch_size, 
                                                num_workers=self.num_workers, 
                                                loader=False,
                                                average=average,
                                                pca=True)
        self.model.load_state_dict(torch.load("models/" + self.hashNum + "_model_" + self.vector + ".pt"))
        self.model.eval()

        criterion = nn.MSELoss()
        PeC = PearsonCorrCoef(num_outputs=y_test.shape[0]).to(self.device)
        
        x_test_pca = x_test_pca.to(self.device, torch.float64)
        y_test_pca = y_test_pca.to(self.device)
        y_test = y_test.to(self.device)
        
        pred_y_pca = self.model(x_test_pca)
        
        loss_pca = criterion(pred_y_pca, y_test_pca.to(self.device))
              
        pearson_pca =torch.mean(PeC(pred_y_pca.moveaxis(0,1), y_test_pca.moveaxis(0,1)))
        
        pred_y = torch.from_numpy(self.pca.inverse_transform(pred_y_pca.detach().numpy())).to(torch.float32)
        
        pearson = torch.mean(PeC(pred_y.moveaxis(0,1), y_test.moveaxis(0,1)))
        loss = criterion(pred_y, y_test)
        
        print("Vector Correlation_PCA: ", float(pearson_pca))
        print("Vector Correlation: ", float(pearson))
        print("Loss_PCA: ", float(loss_pca))
        print("Loss: ", float(loss))