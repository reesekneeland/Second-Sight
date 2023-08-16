import torch
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from utils import *
import wandb
import yaml
from tqdm import tqdm
from torchmetrics import PearsonCorrCoef
from model_zoo import AutoEncoderModel
import argparse

# Main Class    
class AutoEncoder():
    def __init__(self, 
                 config="hybrid", #hybrid, gnet, or clip
                 inference=False,
                 subject=1,
                 lr=0.00001,
                 log=False,
                 batch_size=64,
                 device="cuda",
                 num_workers=4,
                 epochs=200,
                 ):
        
        self.subject = subject
        self.config = config
        self.device = torch.device(device)
        if inference:
            self.log = False
        else:
            self.log = log
            self.lr = lr
            self.batch_size = batch_size
            self.num_epochs = epochs
            self.num_workers = num_workers
            if config == "clip":
                self.vector = "c"
            else:
                self.vector = "images"
            # Initialize the data loaders
            self.trainLoader, self.valLoader, _ = load_nsd(vector=self.vector, 
                                                                batch_size=self.batch_size, 
                                                                num_workers=self.num_workers, 
                                                                ae=True,
                                                                average=False,
                                                                encoderModel=self.config,
                                                                subject=self.subject)
             # Initializes Weights and Biases to keep track of experiments and training runs
            if(self.log):
                wandb.init(
                    # set the wandb project where this run will be logged
                    project="Autoencoder",
                    # track hyperparameters and run metadata
                    config={
                    "subject": self.subject,
                    "config": self.config,
                    "epochs": self.num_epochs,
                    "learning_rate": self.lr,
                    "batch_size:": self.batch_size,
                    "num_workers": self.num_workers
                    }
                )
        subject_sizes = [0, 15724, 14278, 0, 0, 13039, 0, 12682]
        self.x_size = subject_sizes[self.subject]
        # Initialize the Pytorch model class
        self.model = AutoEncoderModel(self.x_size)
        # Send model to Pytorch Device 
        self.model.to(self.device)
    

    def train(self):
        # Set best loss to negative value so it always gets overwritten
        best_loss = -1.0
        loss_counter = 0
        
        # Configure the pytorch objects, loss function (criterion)
        criterion = nn.MSELoss(reduction="sum")
        
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
                # x_data = Beta (Brain data)
                # y_data = Beta prime (Vector conditioned encoding)
                x_data, y_data = data
                
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
                
            train_loss = running_loss/len(self.trainLoader.dataset)
            tqdm.write('[%d] train loss: %.8f' %
                (epoch + 1, train_loss ))
                #     # wandb.log({'epoch': epoch+1, 'loss': running_loss/(50 * self.batch_size)})
                
            # Entering validation stage
            # Set model to evaluation mode
            self.model.eval()
            running_test_loss = 0.0
            for i, data in enumerate(self.valLoader):
                
                # Loading in the test data
                # x_data = Beta (Brain data)
                # y_data = Beta prime (Vector conditioned encoding)
                x_data, y_data = data
                x_data = x_data.to(self.device)
                y_data = y_data.to(self.device)
                # Generating predictions based on the current model
                pred_y = self.model(x_data).to(self.device)
                
                # Compute the test loss 
                loss = criterion(pred_y, y_data)

                running_test_loss += loss.item()
                
            test_loss = running_test_loss / len(self.valLoader.dataset)
                
            # Printing and logging loss so we can keep track of progress
            tqdm.write('[%d] test loss: %.8f' %
                        (epoch + 1, test_loss))
            if(self.log):
                wandb.log({'test_loss': test_loss})
                    
            # Check if we need to save the model
            # Early stopping
            if(best_loss == -1.0 or test_loss < best_loss):
                best_loss = test_loss
                torch.save(self.model.state_dict(), "models/sub0{subject}_{config}_autoencoder.pt".format(subject=self.subject, config=self.config))
                loss_counter = 0
            else:
                loss_counter += 1
                tqdm.write("loss counter: " + str(loss_counter))
                if(loss_counter >= 5):
                    break
        if(self.log):
                wandb.finish()
            
    def predict(self, x):
        self.model.load_state_dict(torch.load("models/sub0{subject}_{config}_autoencoder.pt".format(subject=self.subject, config=self.config), map_location=self.device))
        self.model.eval()
        self.model.to(self.device)
        out = self.model(x.to(self.device)).to(self.device)
        return out
                
    
    def benchmark(self, encodedPass=False, average=True):
        _, _, self.testLoader = load_nsd(vector=self.vector, 
                                        batch_size=self.batch_size, 
                                        num_workers=self.num_workers, 
                                        ae=True,
                                        encoderModel=self.config,
                                        average=average,
                                        subject=self.subject)
        datasize = len(self.testLoader.dataset)
        out = torch.zeros((datasize,self.x_size))
        target = torch.zeros((datasize, self.x_size))
        self.model.load_state_dict(torch.load("models/sub0{subject}_{config}_autoencoder.pt".format(subject=self.subject, config=self.config)))
        self.model.eval()
        self.model.to(self.device)

        loss = 0
        pearson_loss = 0
        
        criterion = nn.MSELoss()
        
        for index, data in enumerate(self.testLoader):
            
            x_test, y_test = data
            PeC = PearsonCorrCoef(num_outputs=x_test.shape[0]).to(self.device)
            y_test = y_test.to(self.device)
            x_test = x_test.to(self.device)
            # Generating predictions based on the current model
            if(encodedPass):
                pred_y = self.model(y_test).to(self.device)
            else:
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
        PeC = PearsonCorrCoef()
        r = []
        for p in range(out.shape[1]):
            
            # Correlation across voxels for a sample (Taking a column)
            r.append(PeC(out[:,p], target[:,p]))
        r = np.array(r)
        
        print("Model: {} Autoencoder, Subject: {}, Averaged: {}, Encoded Pass: {}".format(self.config, self.subject, average, encodedPass))
        print("Vector Correlation: ", float(pearson_loss))
        print("Mean Pearson: ", np.mean(r))
        print("Loss: ", float(loss))
        plt.hist(r, bins=40, log=True)
        plt.savefig("data/charts/subject{}_{}_autoencoder_pearson_correlation.png".format(self.subject, self.config))
        

if __name__ == "__main__":
     # Create the parser and add arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument(
        '--subjects', 
        help="list of subjects to train models for, if not specified, will run on all subjects",
        type=str,
        default="1,2,5,7")
    parser.add_argument(
        '--configs', 
        help="list of autoencoder configs to train models for, if not specified, will run on all configs",
        type=str,
        default="gnet,clip,hybrid")
    
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Batch size for training",
    )
    parser.add_argument(
        "--log",
        help="whether to log to wandb",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--num_epochs",
        help="number of epochs of training",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )
    parser.add_argument(
        "--benchmark",
        help="run benchmark on each autoencoder model after it finishes training.",
        type=bool,
        default=True,
    )
    args = parser.parse_args()
    subject_list = [int(sub) for sub in args.subjects.strip().split(",")]
    configs_list = [int(sub) for sub in args.configs.strip().split(",")]
    
    for sub in args.subjects:
        for config in args.configs:
            AE = AutoEncoder(config=config,
                        inference=False,
                        subject=sub,
                        lr=args.lr,
                        log=args.log,
                        batch_size=args.batch_size,
                        device=args.device,
                        num_workers=args.num_workers,
                        epochs=args.num_epochs)
            AE.train()
            if(args.benchmark):
                AE.benchmark(encodedPass=False, average=False)
                AE.benchmark(encodedPass=False, average=True)