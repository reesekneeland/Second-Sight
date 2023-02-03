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
#create_whole_region_normalized()

# Call process data 
# Input: The vector you want processed as a string
#process_data("c_combined")

# Call to Index into the sample and then using y_mask to grab the correct samples. 
# Input: 
#   - Parameter #1: The vector you want grabbed as a string
#   - Parameter #2: The vector threshold as a string
#   - Parameter #3: The vector hash number as a string
grab_samples("z", "0.063478", "206")



