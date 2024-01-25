import os, sys, shutil
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib as plt
from PIL import Image
sys.path.append("src")
from utils import *
from data_utils import *
from autoencoder import AutoEncoder
from gnet8_encoder import GNet8_Encoder
from matplotlib.lines import Line2D
import matplotlib as mpl
import math
import matplotlib.image as mpimg
import random


create_whole_region_normalized(subject=3, include_heldout=True)
for subject in tqdm(range(4,9)):
    create_whole_region_unnormalized(subject, include_heldout=True)
    create_whole_region_normalized(subject, include_heldout=True)
    
# for subject in tqdm([1,2,5,7]):
#     create_whole_region_normalized(subject, include_heldout=False)
#     create_whole_region_normalized(subject, include_heldout=True)