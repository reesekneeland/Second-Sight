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
    hashNum = update_hash()
    D = Decoder(hash = hashNum,
                 lr=0.0000005,
                 vector="c", 
                 threshold=0.2,
                 log=False, 
                 batch_size=375,
                 parallel=True,
                 device="cuda",
                 num_workers=16,
                 epochs=300,
                 only_test=False
                 )
    # D.train()
    modelId = hashNum + "_model_" + D.vector + ".pt"
    outputs, targets = D.predict(model=modelId, indices=[1, 2, 3])
    cosSim = nn.CosineSimilarity(dim=0)
    print(cosSim(outputs[0], targets[0]))
    print(cosSim(torch.randn_like(outputs[0]), targets[0]))
    
    E = Encoder()
    z = torch.load("latent_vectors/target_z.pt")
    img = E.reconstruct(z, outputs[0], 0.9999999999)
    img2 = E.reconstruct(z, targets[0], 0.9999999999)
    img = E.reconstruct(z, outputs[1], 0.9999999999)
    img2 = E.reconstruct(z, targets[1], 0.9999999999)
    img = E.reconstruct(z, outputs[2], 0.9999999999)
    img2 = E.reconstruct(z, targets[2], 0.9999999999)
    
    reconstruction_outputs = [outputs[0], outputs[1], outputs[2], outputs[3]]
    reconstruction_targets = [targets[0], targets[1], targets[2], targets[3]]
    # c = torch.load("latent_vectors/target_c.pt")
    # img = E.reconstruct(outputs[0], c, 0.00000000001)
    # img2 = E.reconstruct(targets[0], c, 0.00000000001)

if __name__ == "__main__":
    main()
