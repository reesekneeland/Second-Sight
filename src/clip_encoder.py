from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from utils import *
import wandb
from tqdm import tqdm
from torchmetrics import PearsonCorrCoef
from model_zoo import CLIPEncoderModel
import argparse

    
# Main Class    
class CLIPEncoder():
    def __init__(self,
                 inference=False, 
                 subject=1,
                 lr=0.00001,
                 log=False,
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
            self.log = log
            self.lr = lr
            self.batch_size = batch_size
            self.num_epochs = epochs
            self.num_workers = num_workers
            # Initialize the data loaders
            self.trainLoader, self.valLoader, _ = load_nsd(vector="c", 
                                                            batch_size=self.batch_size, 
                                                            num_workers=self.num_workers, 
                                                            loader=True,
                                                            average=False,
                                                            subject=self.subject)
             # Initializes Weights and Biases to keep track of experiments and training runs
            if(self.log):
                wandb.init(
                    # set the wandb project where this run will be logged
                    project="CLIPEncoder",
                    # track hyperparameters and run metadata
                    config={
                    "subject": self.subject,
                    "vector": "c",
                    "epochs": self.num_epochs,
                    "learning_rate": self.lr,
                    "batch_size:": self.batch_size,
                    "num_workers": self.num_workers
                    }
                )
        subject_sizes = [0, 15724, 14278, 0, 0, 13039, 0, 12682]
        # Initialize the Pytorch model class
        self.model = CLIPEncoderModel(subject_sizes[self.subject])
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
        _, _, y_test, _, _, x_test, _ = load_nsd(vector="c", 
                                                loader=False,
                                                average=average,
                                                subject=self.subject)
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
    
if __name__ == "__main__":
     # Create the parser and add arguments
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--subjects', 
        help="list of subjects to train models for, if not specified, will run on all subjects",
        type=str,
        default="1,2,5,7")
    
    parser.add_argument(
        "--batch_size", type=int, default=750,
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
        default=0.00001,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )
    parser.add_argument(
        "--benchmark",
        help="run benchmark on each encoder model after it finishes training.",
        type=bool,
        default=True,
    )
    args = parser.parse_args()
    subject_list = [int(sub) for sub in args.subjects.split(",")]
    
    for sub in subject_list:
        E = CLIPEncoder(
                    inference=False,
                    subject=sub,
                    lr=args.lr,
                    batch_size=args.batch_size,
                    log=args.log,
                    device=args.device,
                    num_workers=args.num_workers,
                    epochs=args.num_epochs)
        E.train()
        if(args.benchmark):
            E.benchmark(average=False)
            E.benchmark(average=True)