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



def load_mask_from_nii(mask_nii_file):
    return nib.load(mask_nii_file).get_data()



def load_data_masked(vector, only_test=False):

    # 750 trails per session 
    # 40 sessions per subject
    # initialize some empty tensors to allocate memory
    
    if(vector == "z"):
        datashape = (1, 16384)
    elif(vector == "c"):
        datashape = (1, 78848)
    
    
    # Clip Vectors (y data)

    # 34 * 750 = 25500
    y_train = torch.empty((25500, 11838))
    
    # 3 * 750 = 2250
    y_test  = torch.empty((2250, 11838))

    nsd_general = load_mask_from_nii("brainmask_nsdgeneral_1.0.nii")
    # print(nsd_general.shape)
    
    nsd_general_mask = np.nan_to_num(nsd_general)
    nsd_mask = np.array(nsd_general_mask.reshape((699192,)), dtype=bool)
    print(nsd_mask)

    if(not only_test):
            
            # Loads the full collection of beta sessions for subject 1
            for i in tqdm(range(1,35), desc="Loading Training Voxels"):
                beta = nsda.read_betas(subject='subj01', 
                                    session_index=i, 
                                    trial_index=[], # Empty list as index means get all 750 scans for this session
                                    data_type='betas_fithrf_GLMdenoise_RR',
                                    data_format='func1pt8mm')
                beta = beta.reshape((699192, 750))
                print("got beta")
                # roi_beta = np.where((nsd_general_mask), beta, 0)
                for j in range(750):

                    curScan = beta[:, j]

                    # Discard the unmasked values and keeps the masked values. 
                    y_train[j + (i-1)*750] = torch.from_numpy(curScan[nsd_mask])
                print("reshaped scan")


            for i in tqdm(range(35,38), desc="Loading Training Voxels"):
                beta = nsda.read_betas(subject='subj01', 
                                    session_index=i, 
                                    trial_index=[], # Empty list as index means get all 750 scans for this session
                                    data_type='betas_fithrf_GLMdenoise_RR',
                                    data_format='func1pt8mm')
                beta = beta.reshape((699192, 750))
                print("got beta")
                # roi_beta = np.where((nsd_general_mask), beta, 0)
                for j in range(750):
                    
                    curScan = beta[:, j]

                    # Discard the unmasked values and keeps the masked values. 
                    y_test[j + (i-1)*750] = torch.from_numpy(curScan[nsd_mask])
                print("reshaped scan")

    
    # Loading the description object for subejct1
    subj1y = nsda.stim_descriptions.nsda.stim_descriptions['subject1'] != 0

    # Do the same annotation extraction process from load_data_whole()
    # Brain scan data (x data)
    x_train = torch.empty((25500, datashape[-1]))
    x_test = torch.empty((2250, datashape[-1]))
    
    for i in tqdm(range(0,25500), desc="train loader"):
        
        # Flexible to both Z and C tensors depending on class configuration
        index = int(subj1y.loc[(subj1y['subject1_rep0'] == i+1) | (subj1y['subject1_rep1'] == i+1) | (subj1y['subject1_rep2'] == i+1)].nsdId)
        x_train[i] = torch.reshape(torch.load("/home/naxos2-raid25/kneel027/home/kneel027/nsd_local/nsddata_stimuli/tensors/" + vector + "/" + str(index) + ".pt"), datashape)
    
    for i in tqdm(range(0,2250), desc="test loader"):
        index = int(subj1y.loc[(subj1y['subject1_rep0'] == 25501 + i) | (subj1y['subject1_rep1'] == 25501 + i) | (subj1y['subject1_rep2'] == 25501 + i)].nsdId)
        x_test[i] = torch.reshape(torch.load("/home/naxos2-raid25/kneel027/home/kneel027/nsd_local/nsddata_stimuli/tensors/" + vector + "/" + str(index) + ".pt"), datashape)

    print("STATS", torch.max(x_train), torch.var(x_train))
    return x_train, x_test, y_train, y_test




def main():
    
    # First URL: This is the original read-only NSD file path (The actual data)
    # Second URL: Local files that we are adding to the dataset and need to access as part of the data
    # Object for the NSDAccess package
    nsda = NSDAccess('/home/naxos2-raid25/kneel027/home/surly/raid4/kendrick-data/nsd', '/home/naxos2-raid25/kneel027/home/kneel027/nsd_local')
    
    load_data_masked("z")

if __name__ == "__main__":

    main()
