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
    os.chdir("/export/raid1/home/kneel027/Second-Sight/")
    hashNum = update_hash()
    hashNum = "022"
    D = Decoder(hashNum = hashNum,
                 lr=0.00001,
                 vector="c", 
                 threshold=0.2,
                 log=False, 
                 batch_size=750,
                 parallel=False,
                 device="cuda",
                 num_workers=16,
                 epochs=300
                 )
    # D.train()
    modelId = D.hashNum + "_model_" + D.vector + ".pt"
    outputs_c, targets_c = D.predict(model=modelId, indices=[1, 2, 3])
    # Test
    # modelId_z = "044" + "_model_" + "z" + ".pt"
    # outputs_z, targets_z = D.predict(model=modelId_z, indices=[1, 2, 3])
    cosSim = nn.CosineSimilarity(dim=0)
    print(cosSim(outputs_c[0], targets_c[0]))
    print(cosSim(torch.randn_like(outputs_c[0]), targets_c[0]))
    
    E = Encoder()
    z = torch.load("/export/raid1/home/kneel027/Second-Sight/latent_vectors/" + "044_model_z.pt" + "/target_z.pt")
    c = torch.load("/export/raid1/home/kneel027/Second-Sight/latent_vectors/" + "022_model_c.pt" + "/target_c.pt")
    outputs_z = torch.load("/export/raid1/home/kneel027/Second-Sight/latent_vectors/" + "044_model_z.pt" + "/output_z.pt")
    E.reconstructNImages(z, outputs_c[0], targets_c[0],  0.9999999999,
                         c, outputs_z[0], z,  0.00000000001)

    # img = E.reconstruct(z, outputs[0], 0.9999999999)
    # img2 = E.reconstruct(z, targets[0], 0.9999999999)
    # img = E.reconstruct(z, outputs[1], 0.9999999999)
    # img2 = E.reconstruct(z, targets[1], 0.9999999999)
    # img = E.reconstruct(z, outputs[2], 0.9999999999)
    # img2 = E.reconstruct(z, targets[2], 0.9999999999)
    
    # reconstruction_outputs = [outputs[0], outputs[1], outputs[2], outputs[3]]
    # reconstruction_targets = [targets[0], targets[1], targets[2], targets[3]]
    # c = torch.load("/export/raid1/home/kneel027/Second-Sight/latent_vectors/" + "022_model_c.pt" + "/target_c.pt")
    # img = E.reconstruct(outputs[0], c, 0.00000000001)
    # img2 = E.reconstruct(targets[0], c, 0.00000000001)
    # img = E.reconstruct(outputs[1], c, 0.00000000001)
    # img2 = E.reconstruct(targets[1], c, 0.00000000001)
    # img = E.reconstruct(outputs[2], c, 0.00000000001)
    # img2 = E.reconstruct(targets[2], c, 0.00000000001)

if __name__ == "__main__":
    main()
