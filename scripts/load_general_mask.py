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

# First URL: This is the original read-only NSD file path (The actual data)
# Second URL: Local files that we are adding to the dataset and need to access as part of the data
# Object for the NSDAccess package
nsda = NSDAccess('/home/naxos2-raid25/kneel027/home/surly/raid4/kendrick-data/nsd', '/home/naxos2-raid25/kneel027/home/kneel027/nsd_local')
    
# def load_data_3D(vector, only_test=False):
        
        # 750 trails per session 
        # 40 sessions per subject
        # initialize some empty tensors to allocate memory
        
        
        # if(vector == "c"):
        #     y_train = torch.empty((25500, 77, 1024))
        #     y_test  = torch.empty((2250, 77, 1024))
        # elif(vector == "z"):
        #     y_train = torch.empty((25500, 4, 64, 64))
        #     y_test  = torch.empty((2250, 4, 64, 64))
        
        
        # # 34 * 750 = 25500
        # x_train = torch.empty((25500, 42, 22, 27))
        
        # # 3 * 750 = 2250
        # x_test  = torch.empty((2250, 42, 22, 27))
        
        # #grabbing voxel mask for subject 1
        # voxel_mask = voxel_data_dict['voxel_mask']["1"]
        # voxel_mask_reshape = voxel_mask.reshape((81, 104, 83, 1))

        # slices = get_slices(voxel_mask_reshape)

        # # Checks if we are only loading the test data so we don't have to load all the training betas
        # if(not only_test):
            
        #     # Loads the full collection of beta sessions for subject 1
        #     for i in tqdm(range(1,35), desc="Loading Training Voxels"):
        #         beta = nsda.read_betas(subject='subj01', 
        #                             session_index=i, 
        #                             trial_index=[], # Empty list as index means get all 750 scans for this session
        #                             data_type='betas_fithrf_GLMdenoise_RR',
        #                             data_format='func1pt8mm')
        #         roi_beta = np.where((voxel_mask_reshape), beta, 0)
        #         beta_trimmed = roi_beta[slices] 
        #         beta_trimmed = np.moveaxis(beta_trimmed, -1, 0)
        #         x_train[(i-1)*750:(i-1)*750+750] = torch.from_numpy(beta_trimmed)


def get_slices(voxel_mask_reshape):
        
    print("Calculating slices....")
    single_trial = nsda.read_betas(subject='subj01', 
                                    session_index=1, 
                                    trial_index=[0], # Empty list as index means get all for this session
                                    data_type='betas_fithrf_GLMdenoise_RR',
                                    data_format='func1pt8mm')
    
    print(single_trial.shape)
    roi_beta = np.where((voxel_mask_reshape), single_trial, 0)
    print(roi_beta.shape)

    slices = tuple(slice(idx.min(), idx.max() + 1) for idx in np.nonzero(roi_beta))
    return slices

def load_mask_from_nii(mask_nii_file):
    return nib.load(mask_nii_file).get_fdata()

    

def load_general_mask(vector, only_test=False):
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

    nsd_general = load_mask_from_nii("brainmask_nsdgeneral_1.0.nii")
    # print(nsd_general.shape)
    
    nsd_general_mask = np.nan_to_num(nsd_general)
    nsd_general_mask_reshape = nsd_general_mask.reshape((81, 104, 83, 1))

    slices = get_slices(nsd_general_mask_reshape)
    
    # print(nsd_general.shape)
    # print(nsd_general[0])

    if(not only_test):
            
            # Loads the full collection of beta sessions for subject 1
            for i in tqdm(range(1,35), desc="Loading Training Voxels"):
                beta = nsda.read_betas(subject='subj01', 
                                    session_index=i, 
                                    trial_index=[], # Empty list as index means get all 750 scans for this session
                                    data_type='betas_fithrf_GLMdenoise_RR',
                                    data_format='func1pt8mm')
                roi_beta = np.where((nsd_general_mask), beta, 0)
                beta_trimmed = roi_beta[slices] 
                print(beta_trimmed.shape)
                beta_trimmed = np.moveaxis(beta_trimmed, -1, 0)
                x_train[(i-1)*750:(i-1)*750+750] = torch.from_numpy(beta_trimmed)




def main():

    load_general_mask("z")

if __name__ == "__main__":

    main()
