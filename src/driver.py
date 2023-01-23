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
    D = Decoder(lr=0.00001,
                 vector="z", 
                 log=True, 
                 batch_size=375,
                 parallel=True,
                 device="cuda",
                 num_workers=16,
                 epochs=300,
                 only_test=False
                 )
    D.train()
    outputs, targets = D.predict(indices=[0], model="model_z.pt")
    cosSim = nn.CosineSimilarity(dim=0)
    print(cosSim(outputs[0], targets[0]))
    print(cosSim(torch.randn_like(outputs[0]), targets[0]))
    
    E = Encoder()
    c = torch.load("latent_vectors/target_c.pt")
    img = E.reconstruct(outputs[0], c, 0.00000000001)
    img2 = E.reconstruct(targets[0], c, 0.00000000001)

if __name__ == "__main__":
    main()
