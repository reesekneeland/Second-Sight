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

# z
# Hash             = 206
# Mean             = 0.08684097
# Threshold        = 0.063478
# Number of voxels = 6378


# Create the whole region of the visual cortex with 11838 voxels. 
#utils.create_whole_region_unnormalized()

# Create the whole region and normalize it by subtracting
# the meand and diving by the standard deveiation. 
# normalization_test()

# Call process data 
# Input: The vector you want processed as a string
# process_data("z_img_mixer")

# Call to Index into the sample and then using y_mask to grab the correct samples. 
# Input: 
#   - Parameter #1: The vector you want grabbed as a string
#   - Parameter #2: The vector threshold as a string
#   - Parameter #3: The vector hash number as a string
# create_whole_region_normalized()
# grab_samples("z_img_mixer", "0.08283", "395")

# extract_dim("c_img_mixer", 0)

# process_data_full("z_img_mixer")
process_data_full("c_img_0")
process_data_full("c_text_0")
# process_data_full("c_combined")

# load_data("z_img_mixer", loader=False)

# voxel_dir = "/export/raid1/home/styvesg/data/nsd/voxels/"
# voxel_data_set = h5py.File(voxel_dir+'voxel_data_V1_4_part1.h5py', 'r')
# load_cc3m()