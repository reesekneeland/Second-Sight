# Only GPU's in use
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
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
sys.path.append('src')
from utils import *
import copy
from tqdm import tqdm
import nibabel as nib

_, _, _, _, _, _, _ = load_nsd(vector="c_img_uc", subject=1, loader=False, average=False)
_, _, _, _, _, _, _ = load_nsd(vector="c_img_uc", subject=2, loader=False, average=False)
_, _, _, _, _, _, _ = load_nsd(vector="c_img_uc", subject=5, loader=False, average=False)
_, _, _, _, _, _, _ = load_nsd(vector="c_img_uc", subject=7, loader=False, average=False)
# vdvae_73k = torch.load("/home/naxos2-raid25/kneel027/home/kneel027/nsd_local/preprocessed_data/z_vdvae_73k.pt")
# torch.save(torch.mean(vdvae_73k, dim=0), "/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/vdvae/train_mean.pt")
# torch.save(torch.std(vdvae_73k, dim=0), "/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/vdvae/train_std.pt")
# vdvae_27k = torch.load("/home/naxos2-raid25/kneel027/home/kneel027/nsd_local/preprocessed_data/subject1/z_vdvae.pt")
# for i in range(vdvae_73k.shape[0]):
#     print(i, torch.sum(torch.count_nonzero(vdvae_73k[i])), torch.sum(torch.count_nonzero(vdvae_27k[i])))
# process_raw_tensors(vector="z_vdvae")
# process_masks(subject=1, big=True)
# subjects = [7]
# for subject in subjects:
#     # create_whole_region_unnormalized(subject=subject, big=True)
#     # create_whole_region_normalized(subject=subject, big=True)
#     # process_data(subject=subject, vector="c_img_uc")
#     # process_data(subject=subject, vector="images")
#     # process_data(subject=subject, vector="z_vdvae")
#     process_masks(subject=subject, big=False)
#     process_masks(subject=subject, big=True)

# mask_path = "masks/subject1/nsdgeneral_big.nii.gz"
# # mask_path = "/export/raid1/home/styvesg/data/nsd/masks/subj01/func1pt8mm/brainmask_inflated_1.0.nii"
# # mask_path_v = "/home/naxos2-raid25/kneel027/home/kneel027/home/styvesg/data/nsd/masks/subj01/func1pt8mm/roi/prf-visualrois.nii.gz"
# nsd_general = nib.load(mask_path).get_fdata()
# nsd_general = np.nan_to_num(nsd_general)#.astype(bool)
# nsd_general = np.where(nsd_general==1.0, True, False)
# # nsd_general = np.where(nsd_general==1.0, True, False)
# print("NSD_GENERAL S1: ", np.unique(nsd_general, return_counts=True))

# visual_rois = nib.load(mask_path_v).get_fdata()
# V1L = np.where(visual_rois==1.0, True, False)
# V1R = np.where(visual_rois==2.0, True, False)
# V1 = torch.from_numpy(V1L[nsd_general] + V1R[nsd_general])
# print("V1 S1: ", np.unique(V1, return_counts=True))

# encoderWeights = torch.load("masks/subject{}/{}_encoder_prediction_weights.pt".format(1, "gnetEncoder_clipEncoder"))
# V1 = torch.load("masks/subject1/V1.pt")
# early_vis = torch.load("masks/subject1/early_vis.pt")
# higher_vis = torch.load("masks/subject1/higher_vis.pt")
# print(torch.mean(encoderWeights[0, V1], dim=0), torch.mean(encoderWeights[1, V1], dim=0))
# print(torch.mean(encoderWeights[0, early_vis], dim=0), torch.mean(encoderWeights[1, early_vis], dim=0))
# print(torch.mean(encoderWeights[0, higher_vis], dim=0), torch.mean(encoderWeights[1, higher_vis], dim=0))
# thresh = encoderWeights[0] > 0.5
# print(torch.sum(thresh))