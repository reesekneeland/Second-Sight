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
# for subject in range(1, 9):
#     # create_whole_region_unnormalized(subject=subject)
#     # create_whole_region_normalized(subject=subject)
#     process_data(subject=subject, vector="c_img_uc")
#     process_data(subject=subject, vector="images")
#     # process_data(subject=subject, vector="z_vdvae")
#     # process_masks(subject=subject)
