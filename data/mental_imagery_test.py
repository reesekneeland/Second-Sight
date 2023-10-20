import os
import sys
import torch
import numpy as np
import argparse
import torchvision.transforms as T
from data_utils import *
from torch.utils.data import DataLoader, Dataset

# nsd_root = '/export/raid1/home/surly/raid1/kendrick-data/nsd/'
# stim_dir = nsd_root + 'nsddata_stimuli/stimuli/nsd/'
# beta_dir = nsd_root + 'nsddata_betas/ppdata/'
# mask_dir= nsd_root + 'nsddata/ppdata/'
# img_stim_file = stim_dir + "nsdimagery_stimuli.pkl3"
# beta_subj = beta_dir + "subj%02d/func1pt8mm/nsdimagerybetas_fithrf/betas_nsdimagery.nii.gz"%subj

## LOAD THE STIM IMAGES AND SEQUENCE ALIGNMENT DESCRIPTORS
stim_dir = "data/nsddata_stimuli/stimuli/nsd/"
img_stim_file = stim_dir + "nsdimagery_stimuli.pkl3"
ex_file = open(img_stim_file, 'rb')
imagery_dict = pickle.load(ex_file)
print(imagery_dict.keys())
ex_file.close()
from src.load_nsd import image_feature_fn
exps = imagery_dict['exps']
cues = imagery_dict['cues']
image_map  = imagery_dict['image_map']
image_data = image_feature_fn(imagery_dict['image_data'])

## LOAD THE DATA WITHOUT GLOBAL ZSCORING (EVEN IF IT IS ZSCORED WE ARE GONNA RE-ZSCORE LATER)
beta_dir =    "/export/raid1/home/surly/raid1/kendrick-data/nsd/nsddata_betas/ppdata/"
from src.load_nsd import load_beta_file
voxel_data_raw = {}
for k,s in enumerate(subjects):
    print ('--------  subject %d  -------' % s)
    beta_subj = beta_dir + "subj%02d/func1pt8mm/nsdimagerybetas_fithrf/betas_nsdimagery.nii.gz"%(s)
    voxel_data_raw[s] = load_beta_file(beta_subj, zscore=False, voxel_mask=voxel_mask[s])
    print (voxel_data_raw[s].shape)
def zscore(x, mean=None, stddev=None, return_stats=False):
    if mean is not None:
        m = mean
    else:
        m = np.mean(x, axis=0, keepdims=True)
    if stddev is not None:
        s = stddev
    else:
        s = np.std(x, axis=0, keepdims=True)
    if return_stats:
        return (x - m)/(s+1e-6), m, s
    else:
        return (x - m)/(s+1e-6)
meta_cond_idx = {
    'visA': np.arange(len(exps))[exps=='visA'],
    'visB': np.arange(len(exps))[exps=='visB'],
    'imgA_1': np.arange(len(exps))[exps=='imgA_1'],
    'imgA_2': np.arange(len(exps))[exps=='imgA_2'],
    'imgB_1': np.arange(len(exps))[exps=='imgB_1'],
    'imgB_2': np.arange(len(exps))[exps=='imgB_2']
}
## NORMALIZATION OF THE DATA FOR EACH INDIVIDUAL TRIAL
voxel_data_n = {}
for s in voxel_data_raw.keys():
    voxel_data_n[s] = np.copy(voxel_data_raw[s])
    for c,idx in meta_cond_idx.items():
        voxel_data_n[s][idx] = zscore(voxel_data_raw[s][idx])
## EXAMPLE CONDITION AVERAGED RESPONSES
def condition_average(data, cond):
    idx, idx_count = np.unique(cond, return_counts=True)
    idx_list = [cond==i for i in np.sort(idx)]
    avg_data = np.zeros(shape=(len(idx),)+data.shape[1:], dtype=np.float32)
    for i,m in enumerate(idx_list):
        avg_data[i] = np.mean(data[m], axis=0)
    return avg_data
cond_idx = {
    'visA': np.arange(len(exps))[exps=='visA'],
    'visB': np.arange(len(exps))[exps=='visB'],
    'imgA': np.arange(len(exps))[np.logical_or(exps=='imgA_1', exps=='imgA_2')],
    'imgB': np.arange(len(exps))[np.logical_or(exps=='imgB_1', exps=='imgB_2')]
}
cond_im_idx = {n: [image_map[c] for c in cues[idx]] for n,idx in cond_idx.items()}
## EXAMPLE USE
for c, idx, im_idx in zip_dict(cond_idx, cond_im_idx): # loop conditions
    data_single = voxel_data_n[s][idx]
    data = condition_average(data_single, im_idx)