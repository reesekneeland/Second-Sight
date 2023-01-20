import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
import torch
from torch.autograd import Variable
import numpy as np
from nsd_access import NSDAccess
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch.nn as nn
from pycocotools.coco import COCO
import h5py
from utils import *
import wandb
import copy
from tqdm import tqdm
from encoder import Encoder
from decoder import Decoder

def main():
    D = Decoder(vector="c", 
                 device="cuda",
                 lr=0.2,
                 batch_size=375,
                 num_workers=16,
                 epochs=200,
                 log=True,
                 parallel=False,
                 only_test=False
                 )
    D.train()
    outputs, targets = D.predict([0])
    cosSim = nn.CosineSimilarity(dim=0)
    print(cosSim(outputs[0], targets[0]))
    print(cosSim(torch.randn_like(outputs[0]), targets[0]))
    
    E = Encoder()
    z = torch.load("target_z.pt")
    img = E.reconstruct(z, outputs[0], 0.999999999)
    img2 = E.reconstruct(z, targets[0], 0.999999999)
    print("reconstructed", img)

if __name__ == "__main__":
    main()
