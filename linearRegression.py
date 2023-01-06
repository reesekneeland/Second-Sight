import torch
from torch.autograd import Variable
import numpy as np
from nsd_access import NSDAccess
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch.nn as nn
from pycocotools.coco import COCO

nsda = NSDAccess('/home/naxos2-raid25/kneel027/home/surly/raid4/kendrick-data/nsd', '/home/naxos2-raid25/kneel027/home/kneel027/nsd_local')
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def load_data():
    y_train = torch.empty((29250, 1, 4, 64, 64))
    y_test = torch.empty((750, 1, 4, 64, 64))
    x_train = torch.empty((29250, 81, 104, 83))
    x_test = torch.empty((750, 81, 104, 83))
    
    # train_betas = []
    for i in range(1,40):
        beta = nsda.read_betas(subject='subj01', 
                            session_index=i, 
                            trial_index=[], # empty list as index means get all for this session
                            data_type='betas_fithrf_GLMdenoise_RR',
                            data_format='func1pt8mm')
        beta = np.moveaxis(beta, -1, 0)
        x_train[(i-1)*750:(i-1)*750+750] = torch.from_numpy(beta)
        print(i)
        
    # train_betas_np = np.concatenate(train_betas, axis=3)
    # train_betas_np.reshape(29250, 81, 104, 83)
    # x_train = torch.from_numpy(train_betas_np)
    print(x_train.shape)
    
    test_betas = nsda.read_betas(subject='subj01', 
                                session_index=40, 
                                trial_index=[], # empty list as index means get all for this session
                                data_type='betas_fithrf_GLMdenoise_RR',
                                data_format='func1pt8mm')
    test_betas = np.moveaxis(test_betas, -1, 0)
    x_test = torch.from_numpy(test_betas)
    print(x_test.shape)
    
    subj1 = nsda.stim_descriptions[nsda.stim_descriptions['subject1'] != 0]
    for i in range(1,29251):
        if(i%1000==0):
            print(i)
        index = int(subj1.loc[(subj1['subject1_rep0'] == i) | (subj1['subject1_rep1'] == i) | (subj1['subject1_rep2'] == i)].nsdId)
        y_train[i-1] = torch.load("/home/naxos2-raid25/kneel027/home/kneel027/nsd_local/nsddata_stimuli/tensors/z/" + str(index) + ".pt")
    for i in range(1,751):
        index = int(subj1.loc[(subj1['subject1_rep0'] == i) | (subj1['subject1_rep1'] == i) | (subj1['subject1_rep2'] == i)].nsdId)
        y_test[i-1] = torch.load("/home/naxos2-raid25/kneel027/home/kneel027/nsd_local/nsddata_stimuli/tensors/z/" + str(index) + ".pt")
    print(y_train.shape)
    print(y_test.shape)
    return x_train, x_test, y_train, y_test
    



class LinearRegressionModel(torch.nn.Module):

	def __init__(self):
		super(LinearRegressionModel, self).__init__()
		self.linear = nn.Linear(81*104*83, 1*4*64*64)

	def forward(self, x):
		y_pred = self.linear(x)
		return y_pred

def train(dataLoader):
    model = LinearRegressionModel()

    criterion = nn.MSELoss(size_average = False)
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

    for i, data in enumerate(dataLoader):
        data
        x_data, y_data = data
        print(x_data.shape, y_data.shape)
        #THIS IS BROKEN RIGHT NOW !!!!!!!!!
        x_data.flatten(start_dim=1).to(device)
        y_data.flatten(start_dim=1).to(device)
        print(x_data.shape, y_data.shape)
        # Forward pass: Compute predicted y by passing
        # x to the model
        pred_y = model(x_data)

        # Compute and print loss
        loss = criterion(pred_y, y_data)

        # Zero gradients, perform a backward pass,
        # and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('iter {}, loss {}'.format(i, loss.item()))
    return model

def main():
    x, x_test, y, y_test = load_data()
    
    trainset = torch.utils.data.TensorDataset(x, y)
    testset = torch.utils.data.TensorDataset(x_test, y_test)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True)
    
    
    model = train(trainloader)
    torch.save(model(x_test[0]), "output.pt")
    torch.save(y_test[0], "target.pt")

if __name__ == "__main__":
    main()
