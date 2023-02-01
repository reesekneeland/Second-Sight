# Only GPU's in use
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
import torch
from torchmetrics.functional import pearson_corrcoef
from torch.autograd import Variable
import numpy as np
from nsd_access import NSDAccess
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch.nn as nn
from pycocotools.coco import COCO
sys.path.append('../src')
from utils import *
import wandb
import copy
from tqdm import tqdm
import nibabel as nib

nsda = NSDAccess('/home/naxos2-raid25/kneel027/home/surly/raid4/kendrick-data/nsd', '/home/naxos2-raid25/kneel027/home/kneel027/nsd_local')
# output filepath
prep_path = "/export/raid1/home/kneel027/nsd_local/preprocessed_data/"
thresholds = {"z": "0.07", "c": "0.06734", "c_prompt": "0.08"}
hashes = {"z": "096", "c": "113", "c_prompt": "011"}
vectors = ["z", "c", "c_prompt"]
whole_region = torch.zeros((27750, 11838))
# whole_region = torch.load(prep_path + "x/whole_region_11838.pt")
nsd_general = nib.load("/export/raid1/home/kneel027/Second-Sight/masks/brainmask_nsdgeneral_1.0.nii").get_data()
# print(nsd_general.shape)

nsd_general_mask = np.nan_to_num(nsd_general)
nsd_mask = np.array(nsd_general_mask.reshape((699192,)), dtype=bool)

# if(not only_test):
    
#     # Loads the full collection of beta sessions for subject 1
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
        

# Normalized whole region. 
whole_region = whole_region / whole_region.max(0, keepdim=True)[0]

#Save the tensor
torch.save(whole_region, prep_path + "x/whole_region_11838.pt")

for vector in vectors:
    if(vector == "z"):
        vec_target = torch.zeros((27750, 16384))
        datashape = (1, 16384)
    elif(vector == "c"):
        vec_target = torch.zeros((27750, 1536))
        datashape = (1, 1536)
    elif(vector == "c_prompt"):
        vec_target = torch.zeros((27750, 78848))
        datashape = (1, 78848)

    # Loading the description object for subejct1
    subj1x = nsda.stim_descriptions[nsda.stim_descriptions['subject1'] != 0]

    for i in tqdm(range(0,27750), desc="vector loader"):
        
        # Flexible to both Z and C tensors depending on class configuration
        index = int(subj1x.loc[(subj1x['subject1_rep0'] == i+1) | (subj1x['subject1_rep1'] == i+1) | (subj1x['subject1_rep2'] == i+1)].nsdId)
        vec_target[i] = torch.reshape(torch.load("/export/raid1/home/kneel027/nsd_local/nsddata_stimuli/tensors/" + vector + "/" + str(index) + ".pt"), datashape)

    torch.save(vec_target, prep_path + vector + "/vector.pt")



for vector in vectors:
    mask = np.load("/export/raid1/home/kneel027/Second-Sight/masks/" + hashes[vector] + "_" + vector + "2voxels_pearson_thresh" + thresholds[vector] + ".npy")
    new_len = np.count_nonzero(mask)
    target = torch.zeros((27750, new_len))
    for i in tqdm(range(27750), desc=(vector + " masking")):
        # Indexing into the sample and then using y_mask to grab the correct samples of a correlation of 0.1 or higher. 
        target[i] = whole_region[i][torch.from_numpy(mask)]
    torch.save(target, prep_path + "x/" + vector + "_2voxels_pearson_thresh" + thresholds[vector] + ".pt")

