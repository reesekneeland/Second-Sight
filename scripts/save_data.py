# Only GPU's in use
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
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

_, _, _, _, _, _, _ = load_nsd(vector="c_img_uc", subject=1, loader=False, average=False, nest=True)

# for subject in range(1, 9):
#     create_whole_region_unnormalized(subject=subject)
#     create_whole_region_normalized(subject=subject)
#     process_data(subject=subject, vector="c_img_uc")
#     process_data(subject=subject, vector="images")
#     # process_masks(subject=subject)