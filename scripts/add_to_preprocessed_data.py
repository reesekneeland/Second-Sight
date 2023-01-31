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
threshold = "0.08"
vector = "c"
hashNum = "011"
whole_region = torch.zeros((27750, 11838))
whole_region = torch.load(prep_path + "x/whole_region_11838.pt")
nsd_general = nib.load("/export/raid1/home/kneel027/Second-Sight/masks/brainmask_nsdgeneral_1.0.nii").get_data()
# print(nsd_general.shape)

nsd_general_mask = np.nan_to_num(nsd_general)
nsd_mask = np.array(nsd_general_mask.reshape((699192,)), dtype=bool)


if(vector == "z"):
    vec_target = torch.zeros((27750, 16384))
    datashape = (1, 16384)
elif(vector == "c"):
    vec_target = torch.zeros((27750, 78848))
    datashape = (1, 78848)

# Loading the description object for subejct1
subj1x = nsda.stim_descriptions[nsda.stim_descriptions['subject1'] != 0]

for i in tqdm(range(0,27750), desc="vector loader"):
    
    # Flexible to both Z and C tensors depending on class configuration
    index = int(subj1x.loc[(subj1x['subject1_rep0'] == i+1) | (subj1x['subject1_rep1'] == i+1) | (subj1x['subject1_rep2'] == i+1)].nsdId)
    vec_target[i] = torch.reshape(torch.load("/export/raid1/home/kneel027/nsd_local/nsddata_stimuli/tensors/" + vector + "/" + str(index) + ".pt"), datashape)

torch.save(vec_target, prep_path + vector + "/vector_" + str(datashape[1]) + ".pt")


mask = np.load("/export/raid1/home/kneel027/Second-Sight/masks/" + hashNum + "_" + vector + "2voxels_pearson_thresh" + threshold + ".npy")
new_len = np.count_nonzero(mask)
target = torch.zeros((27750, new_len))
for i in range(27750):
    # Indexing into the sample and then using y_mask to grab the correct samples of a correlation of 0.1 or higher. 
    target[i] = whole_region[i][torch.from_numpy(mask)]
torch.save(target, prep_path + "x/" + vector + "_2voxels_pearson_thresh" + threshold + ".pt")