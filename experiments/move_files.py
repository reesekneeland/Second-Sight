import os, sys, shutil
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib as plt
from PIL import Image
sys.path.append('../src')
from utils import *
sys.path.append('../data')
from data_utils import *
from autoencoder import AutoEncoder
from matplotlib.lines import Line2D
import matplotlib as mpl
import math
import matplotlib.image as mpimg
import random
from nsd_access import NSDAccess
nsda = NSDAccess('/home/naxos2-raid25/kneel027/home/surly/kendrick-data/nsd', '/home/naxos2-raid25/kneel027/home/kneel027/nsd_local')
os.chdir("..")

source_path = "/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/output/brain-optimized-inference-v2.3/"
gt_source_path = "/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/output/brain-optimized-inference-v2.3_fixed/"
dest_path = "/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/output/brain-optimized-inference-v2.3_fixed3/"

def convert_indices(idx, sub=1, reverse=False, held_out=False):
    df = nsda.stim_descriptions[(nsda.stim_descriptions['subject{}'.format(sub)]) & (nsda.stim_descriptions['shared1000'] == True)]
    if reverse:
        sorted_df = df.copy()
        df = df.sort_values(by='subject{}_rep0'.format(sub))
    else:
        sorted_df = df.sort_values(by='subject{}_rep0'.format(sub))
    nsdIds = sorted_df["nsdId"].tolist()
    # print(len(sorted_df))
    output_idx = []
    for i in idx:
        nsd = df.iloc[i].nsdId
        output_idx.append(nsdIds.index(nsd))
    return output_idx

converted_indices = convert_indices(range(1000), reverse=True)
for s, subject in enumerate([1,2,5,7]):
    for i in tqdm(range(1, 1000)):
        source = f"{source_path}subject{subject}/{i}/"
        dest = f"{dest_path}subject{subject}/{converted_indices[i]}/"
        gt_source = f"{gt_source_path}subject{subject}/{converted_indices[i]}/"
        # os.makedirs(dest, exist_ok=True)
        # shutil.move(dest, source)
        shutil.copytree(source, dest, symlinks=True)
        # shutil.copy(f"{source}low_level.png", f"{dest}low_level.png")
        # shutil.copy(f"{gt_source}ground_truth.png", f"{dest}ground_truth.png")
        # shutil.copy(f"{gt_source}basemodel_recon.png", f"{dest}basemodel_recon.png")
