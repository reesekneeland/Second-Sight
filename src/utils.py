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

        
def get_last_token(s, tokens={'@': list, '.': dict}):
    l,name,entry,t = 2**31,'','',None
    for tok,toktype in tokens.items():
        ss = s.split(tok)
        if len(ss)>1 and len(ss[-1])<l:
            l = len(ss[-1])
            entry = ss[-1]
            name = tok.join(ss[:-1])
            t = toktype
    return name, entry, t


def has_token(s, tokens=['@', '.']):
    isin = False
    for tok in tokens:
        if tok in s:
            isin = True
    return isin

def extend_list(l, i, v):
    if len(l)<i+1:
        l += [None,]*(i+1-len(l))
    l[i] = v
    return l


def embed_dict(fd):
    d = {}
    for k,v in fd.items():
        name, entry, ty = get_last_token(k, {'@': list, '.': dict})
        if ty==list:
            if name in d.keys():
                d[name] = extend_list(d[name], int(entry), v)
            else:
                d[name] = extend_list([], int(entry), v)
        elif ty==dict:
            if name in d.keys():
                d[name].update({entry: v})
            else:
                d[name] = {entry: v}
        else:
            if k in d.keys():
                d[k].update(v)
            else:
                d[k] = v   
    return embed_dict(d) if has_token(''.join(d.keys()), tokens=['@', '.']) else d

def get_slices(voxel_mask_reshape):
        
        print("Calculating slices....")
        single_trial = nsda.read_betas(subject='subj01', 
                                        session_index=1, 
                                        trial_index=[], # Empty list as index means get all for this session
                                        data_type='betas_fithrf_GLMdenoise_RR',
                                        data_format='func1pt8mm')
        roi_beta = np.where((voxel_mask_reshape), single_trial, 0)

        slices = tuple(slice(idx.min(), idx.max() + 1) for idx in np.nonzero(roi_beta))
        return slices
    
def load_data_3D(vector, only_test=False):
        
        # 750 trails per session 
        # 40 sessions per subject
        # initialize some empty tensors to allocate memory
        
        
        if(vector == "c"):
            y_train = torch.empty((25500, 77, 1024))
            y_test  = torch.empty((2250, 77, 1024))
        elif(vector == "z"):
            y_train = torch.empty((25500, 4, 64, 64))
            y_test  = torch.empty((2250, 4, 64, 64))
        
        
        # 34 * 750 = 25500
        x_train = torch.empty((25500, 42, 22, 27))
        
        # 3 * 750 = 2250
        x_test  = torch.empty((2250, 42, 22, 27))
        
        #grabbing voxel mask for subject 1
        voxel_mask = voxel_data_dict['voxel_mask']["1"]
        voxel_mask_reshape = voxel_mask.reshape((81, 104, 83, 1))

        slices = get_slices(voxel_mask_reshape)

        # Checks if we are only loading the test data so we don't have to load all the training betas
        if(not only_test):
            
            # Loads the full collection of beta sessions for subject 1
            for i in tqdm(range(1,35), desc="Loading Training Voxels"):
                beta = nsda.read_betas(subject='subj01', 
                                    session_index=i, 
                                    trial_index=[], # Empty list as index means get all 750 scans for this session
                                    data_type='betas_fithrf_GLMdenoise_RR',
                                    data_format='func1pt8mm')
                roi_beta = np.where((voxel_mask_reshape), beta, 0)
                beta_trimmed = roi_beta[slices] 
                beta_trimmed = np.moveaxis(beta_trimmed, -1, 0)
                x_train[(i-1)*750:(i-1)*750+750] = torch.from_numpy(beta_trimmed)
        
        for i in tqdm(range(35,38), desc="Loading Test Voxels"):
            
            # Loads the test betas and puts it into a tensor
            test_betas = nsda.read_betas(subject='subj01', 
                                        session_index=i, 
                                        trial_index=[], # Empty list as index means get all for this session
                                        data_type='betas_fithrf_GLMdenoise_RR',
                                        data_format='func1pt8mm')
            roi_beta = np.where((voxel_mask_reshape), test_betas, 0)
            beta_trimmed = roi_beta[slices] 
            beta_trimmed = np.moveaxis(beta_trimmed, -1, 0)
            x_test[(i-35)*750:(i-35)*750+750] = torch.from_numpy(beta_trimmed)

        # Loading the description object for subejct1
        subj1y = nsda.stim_descriptions[nsda.stim_descriptions['subject1'] != 0]

        for i in tqdm(range(0,25500), desc="Loading Training Vectors"):
            # Flexible to both Z and C tensors depending on class configuration
            index = int(subj1y.loc[(subj1y['subject1_rep0'] == i+1) | (subj1y['subject1_rep1'] == i+1) | (subj1y['subject1_rep2'] == i+1)].nsdId)
            y_train[i] = torch.load("/home/naxos2-raid25/kneel027/home/kneel027/nsd_local/nsddata_stimuli/tensors/" + vector + "/" + str(index) + ".pt")

        for i in tqdm(range(0,2250), desc="Loading Test Vectors"):
            index = int(subj1y.loc[(subj1y['subject1_rep0'] == 25501 + i) | (subj1y['subject1_rep1'] == 25501 + i) | (subj1y['subject1_rep2'] == 25501 + i)].nsdId)
            y_test[i] = torch.load("/home/naxos2-raid25/kneel027/home/kneel027/nsd_local/nsddata_stimuli/tensors/" + vector + "/" + str(index) + ".pt")

        if(vector == "c"):
            y_train = y_train.reshape((25500, 78848))
            y_test  = y_test.reshape((2250, 78848))
        elif(vector == "z"):
            y_train = y_train.reshape((25500, 16384))
            y_test  = y_test.reshape((2250, 16384))
            
        x_train = x_train.reshape((25500, 1, 42, 22, 27))
        x_test  = x_test.reshape((2250, 1, 42, 22, 27))

        print("3D STATS PRENORM", torch.max(x_train), torch.var(x_train))
        x_train_mean, x_train_std = x_train.mean(), x_train.std()
        x_test_mean, x_test_std = x_test.mean(), x_test.std()
        x_train = (x_train - x_train_mean) / x_train_std
        x_test = (x_test - x_test_mean) / x_test_std

        print("3D STATS NORMALIZED", torch.max(x_train), torch.var(x_train))
        return x_train, x_test, y_train, y_test
    
def load_data_roi(vector):
    
    # Open the dictionary of refined voxel information from Ghislain, and pull out the data variable
    voxel_data = voxel_data_dict['voxel_data']
    if(vector == "z"):
        datashape = (1, 16384)
    elif(vector == "c"):
        datashape = (1, 78848)
    
    # Index it to the first subject
    subj1x = voxel_data["1"]
    
    print("ROI STATS", np.max(subj1x), np.var(subj1x))
    # Import the data into a tensor
    x_train = torch.tensor(subj1x[:25500])
    x_test = torch.tensor(subj1x[25500:27750])
    
    # Loading the description object for subejct1
    subj1y = nsda.stim_descriptions.nsda.stim_descriptions['subject1'] != 0
    
    # Do the same annotation extraction process from load_data_whole()
    y_train = torch.empty((25500, datashape[-1]))
    y_test = torch.empty((2250, datashape[-1]))
    
    for i in tqdm(range(0,25500), desc="train loader"):
        
        # Flexible to both Z and C tensors depending on class configuration
        index = int(subj1y.loc[(subj1y['subject1_rep0'] == i+1) | (subj1y['subject1_rep1'] == i+1) | (subj1y['subject1_rep2'] == i+1)].nsdId)
        y_train[i] = torch.reshape(torch.load("/home/naxos2-raid25/kneel027/home/kneel027/nsd_local/nsddata_stimuli/tensors/" + vector + "/" + str(index) + ".pt"), datashape)
    
    for i in tqdm(range(0,2250), desc="test loader"):
        index = int(subj1y.loc[(subj1y['subject1_rep0'] == 25501 + i) | (subj1y['subject1_rep1'] == 25501 + i) | (subj1y['subject1_rep2'] == 25501 + i)].nsdId)
        y_test[i] = torch.reshape(torch.load("/home/naxos2-raid25/kneel027/home/kneel027/nsd_local/nsddata_stimuli/tensors/" + vector + "/" + str(index) + ".pt"), datashape)

    print("STATS", torch.max(x_train), torch.var(x_train))
    return x_train, x_test, y_train, y_test

# Loads the data and puts it into a DataLoader
def get_data(vector, batch_size=375, num_workers=16, only_test=False):
        
    x, x_test, y, y_test = load_data_3D(vector, only_test)
    print("shapes", x.shape, x_test.shape, y.shape, y_test.shape)
    
    # Loads the raw tensors into a Dataset object
    
    # TensorDataset takes in two tensors of equal size and then maps 
    # them to one dataset. 
    # x is the brain data 
    # y are the captions
    trainset = torch.utils.data.TensorDataset(x, y) #.type(torch.LongTensor)
    testset = torch.utils.data.TensorDataset(x_test, y_test) #.type(torch.LongTensor)
    
    # Loads the Dataset into a DataLoader
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return trainloader, testloader

# First URL: This is the original read-only NSD file path (The actual data)
# Second URL: Local files that we are adding to the dataset and need to access as part of the data
# Object for the NSDAccess package
nsda = NSDAccess('/home/naxos2-raid25/kneel027/home/surly/raid4/kendrick-data/nsd', '/home/naxos2-raid25/kneel027/home/kneel027/nsd_local')

# Segmented dataset paths and file loader
# Specific regions for the voxels of the brain      
voxel_dir = "/export/raid1/home/styvesg/data/nsd/voxels/"
voxel_data_set = h5py.File(voxel_dir+'voxel_data_V1_4_part1.h5py', 'r')

# Main dictionary that contains the voxel activation data, as well as ROI maps and indices for the actual beta
voxel_data_dict = embed_dict({k: np.copy(d) for k,d in voxel_data_set.items()})
voxel_data_set.close()