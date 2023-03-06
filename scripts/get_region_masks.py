import os
import sys
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

visual_rois_path = "/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/masks/prf-visualrois.nii.gz"
nsd_general_path = "/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/masks/brainmask_nsdgeneral_1.0.nii"
nsd_general = nib.load(nsd_general_path).get_fdata()
nsd_general = np.nan_to_num(nsd_general).astype(bool)
visual_rois = nib.load(visual_rois_path).get_fdata()
empty_brain = np.full(visual_rois.shape, False)
V1 = np.where(visual_rois==1.0, True, False)
V1 = torch.from_numpy(V1[nsd_general])
V2 = np.where(visual_rois==2.0, True, False)
V2 = torch.from_numpy(V2[nsd_general])
V3 = np.where(visual_rois==3.0, True, False)
V3 = torch.from_numpy(V3[nsd_general])
V4 = np.where(visual_rois==4.0, True, False)
V4 = torch.from_numpy(V4[nsd_general])
V5 = np.where(visual_rois==5.0, True, False)
V5 = torch.from_numpy(V5[nsd_general])
V6 = np.where(visual_rois==6.0, True, False)
V6 = torch.from_numpy(V6[nsd_general])
V7 = np.where(visual_rois==7.0, True, False)
V7 = torch.from_numpy(V7[nsd_general])

torch.save(V1, "/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/masks/V1.pt")
torch.save(V2, "/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/masks/V2.pt")
torch.save(V3, "/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/masks/V3.pt")
torch.save(V4, "/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/masks/V4.pt")
torch.save(V5, "/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/masks/V5.pt")
torch.save(V6, "/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/masks/V6.pt")
torch.save(V7, "/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/masks/V7.pt")
print(np.unique(V1, return_counts=True))
print(np.unique(V2, return_counts=True))
print(np.unique(V3, return_counts=True))
print(np.unique(V4, return_counts=True))
print(np.unique(V5, return_counts=True))
print(np.unique(V6, return_counts=True))
print(np.unique(V7, return_counts=True))
# print(empty_brain.shape, empty_brain)
# print(mask.shape, mask)
# # mask = np.nan_to_num(mask)
print(np.unique(visual_rois, return_counts=True))
