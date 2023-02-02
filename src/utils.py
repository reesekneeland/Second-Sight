import sys
import os
import struct
import time
import numpy as np
import scipy.io as sio
from scipy import ndimage as nd
from scipy import misc
from glob import glob
import h5py
import pickle
import math
import matplotlib.pyplot as plt
import PIL.Image as pim
import nibabel as nib
from nsd_access import NSDAccess
import torch
from tqdm import tqdm


        
def get_hash():
    with open('/export/raid1/home/kneel027/Second-Sight/hash','r') as file:
        h = file.read()
    file.close()
    return str(h)

def update_hash():
    with open('/export/raid1/home/kneel027/Second-Sight/hash','r+') as file:
        h = int(file.read())
        new_h = f'{h+1:03d}'
        file.seek(0)
        file.write(new_h)
        file.truncate()      
    file.close()
    return str(new_h)

# Loads the data and puts it into a DataLoader
def get_data(vector, threshold=0.2, batch_size=375, num_workers=16):
        
    prep_path = "/export/raid1/home/kneel027/nsd_local/preprocessed_data/"
    y = torch.load(prep_path + vector + "/vector.pt").requires_grad_(False)
    x  = torch.load(prep_path + "x/" + vector + "_2voxels_pearson_thresh" + str(threshold) + "_normalization_test.pt").requires_grad_(False)
    x_train = x[:25500]
    x_test = x[25500:27750]
    y_train = y[:25500]
    y_test = y[25500:27750]
    print("shapes", x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    
    # Loads the raw tensors into a Dataset object
    
    # TensorDataset takes in two tensors of equal size and then maps 
    # them to one dataset. 
    # x is the brain data 
    # y are the captions
    trainset = torch.utils.data.TensorDataset(x_train, y_train)
    testset = torch.utils.data.TensorDataset(x_test, y_test)
    
    # Loads the Dataset into a DataLoader
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return trainloader, testloader

# First URL: This is the original read-only NSD file path (The actual data)
# Second URL: Local files that we are adding to the dataset and need to access as part of the data
# Object for the NSDAccess package
nsda = NSDAccess('/home/naxos2-raid25/kneel027/home/surly/raid4/kendrick-data/nsd', '/home/naxos2-raid25/kneel027/home/kneel027/nsd_local')
