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


prep_path = "/export/raid1/home/kneel027/nsd_local/preprocessed_data/"

# First URL: This is the original read-only NSD file path (The actual data)
# Second URL: Local files that we are adding to the dataset and need to access as part of the data
# Object for the NSDAccess package
nsda = NSDAccess('/home/naxos2-raid25/kneel027/home/surly/raid4/kendrick-data/nsd', '/home/naxos2-raid25/kneel027/home/kneel027/nsd_local')


        
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
        
    y = torch.load(prep_path + vector + "/vector.pt").requires_grad_(False)
    x  = torch.load(prep_path + "x/" + vector + "_2voxels_pearson_thresh" + str(threshold) + ".pt").requires_grad_(False)
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


    
def create_whole_region_unnormalized():
    
    whole_region = torch.load(prep_path + "x/whole_region_11838_unnormalized.pt")
    nsd_general = nib.load("/export/raid1/home/kneel027/Second-Sight/masks/brainmask_nsdgeneral_1.0.nii").get_data()
    print(nsd_general.shape)

    nsd_general_mask = np.nan_to_num(nsd_general)
    nsd_mask = np.array(nsd_general_mask.reshape((699192,)), dtype=bool)
        
    # Loads the full collection of beta sessions for subject 1
    for i in tqdm(range(1,38), desc="Loading Voxels"):
        beta = nsda.read_betas(subject='subj01', 
                            session_index=i, 
                            trial_index=[], # Empty list as index means get all 750 scans for this session (trial --> scan)
                            data_type='betas_fithrf_GLMdenoise_RR',
                            data_format='func1pt8mm')

        # Reshape the beta trails to be flattened. 
        beta = beta.reshape((699192, 750))

        for j in range(750):

            # Grab the current beta trail. 
            curScan = beta[:, j]
            
            # Normalizing the scan.  
            single_scan = torch.from_numpy(curScan)

            # Discard the unmasked values and keeps the masked values. 
            whole_region[j + (i-1)*750] = single_scan[nsd_mask]
            
    # Save the tensor
    torch.save(whole_region, prep_path + "x/whole_region_11838_unnormalized.pt")
    
    
def create_whole_region_normalized():
    
    whole_region = torch.zeros((27750, 11838))
    whole_region = torch.load(prep_path + "x/whole_region_11838_unnormalized.pt")
            
    # Normalize the data
    for i in range(11838):
        
        whole_region_mean, whole_region_std = whole_region[:, i].mean(), whole_region[:, i].std()
        whole_region[:, i] = (whole_region[:, i] - whole_region_mean) / whole_region_std

    # Save the tensor
    torch.save(whole_region, prep_path + "x/whole_region_11838.pt")
    
    
def process_data(vector):
    
    if(vector == "z"):
        vec_target = torch.zeros((27750, 16384))
        datashape = (1, 16384)
    elif(vector == "c"):
        vec_target = torch.zeros((27750, 1536))
        datashape = (1, 1536)
    elif(vector == "c_prompt"):
        vec_target = torch.zeros((27750, 78848))
        datashape = (1, 78848)
    elif(vector == "c_combined"):
        vec_target = torch.zeros((27750, 3840))
        datashape = (1, 3840)

    # Loading the description object for subejct1
    
    subj1x = nsda.stim_descriptions[nsda.stim_descriptions['subject1'] != 0]

    for i in tqdm(range(0,27750), desc="vector loader"):
        
        # Flexible to both Z and C tensors depending on class configuration
        
        # TODO: index the column of this table that is apart of the 1000 test set. 
        # Do a check here. Do this in get_data
        # If the sample is part of the held out 1000 put it in the test set otherwise put it in the training set. 
        index = int(subj1x.loc[(subj1x['subject1_rep0'] == i+1) | (subj1x['subject1_rep1'] == i+1) | (subj1x['subject1_rep2'] == i+1)].nsdId)
        vec_target[i] = torch.reshape(torch.load("/export/raid1/home/kneel027/nsd_local/nsddata_stimuli/tensors/" + vector + "/" + str(index) + ".pt"), datashape)

    torch.save(vec_target, prep_path + vector + "/vector.pt")
    
    
    
def grab_samples(vector, threshold, hashNum):
    
    whole_region = torch.load(prep_path + "x/whole_region_11838.pt") 
    mask = np.load("/export/raid1/home/kneel027/Second-Sight/masks/" + hashNum + "_" + vector + "2voxels_pearson_thresh" + threshold + ".npy")
    new_len = np.count_nonzero(mask)
    target = torch.zeros((27750, new_len))
    for i in tqdm(range(27750), desc=(vector + " masking")):
       
        # Indexing into the sample and then using y_mask to grab the correct samples. 
        target[i] = whole_region[i][torch.from_numpy(mask)]
    torch.save(target, prep_path + "x/" + vector + "_2voxels_pearson_thresh" + threshold + ".pt")

