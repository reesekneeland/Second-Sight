import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
import torch
from torch.autograd import Variable
import numpy as np
from PIL import Image
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

    
    reconstructNImages(z_model_hash="044",
                         c_model_hash="022",
                         idx=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])


def train_decoder():
    hashNum = update_hash()
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
    D.train()
    modelId = D.hashNum + "_model_" + D.vector + ".pt"
    outputs_c, targets_c = D.predict(model=modelId, indices=[1, 2, 3])
    # Test
    # modelId_z = "044" + "_model_" + "z" + ".pt"
    # outputs_z, targets_z = D.predict(model=modelId_z, indices=[1, 2, 3])
    cosSim = nn.CosineSimilarity(dim=0)
    print(cosSim(outputs_c[0], targets_c[0]))
    print(cosSim(torch.randn_like(outputs_c[0]), targets_c[0]))

# Encode latent z (1x4x64x64) and condition c (1x77x1024) tensors into an image
# Strength parameter controls the weighting between the two tensors
def reconstructNImages(z_model_hash, c_model_hash, idx):
    Dz = Decoder(hashNum = z_model_hash,
                 vector="z", 
                 log=False, 
                 device="cuda",
                 )
    Dc = Decoder(hashNum = c_model_hash,
                 vector="c", 
                 log=False, 
                 device="cuda",
                 )
    
    # First URL: This is the original read-only NSD file path (The actual data)
    # Second URL: Local files that we are adding to the dataset and need to access as part of the data
    # Object for the NSDAccess package
    nsda = NSDAccess('/home/naxos2-raid25/kneel027/home/surly/raid4/kendrick-data/nsd', '/home/naxos2-raid25/kneel027/home/kneel027/nsd_local')
    
    # Retriving the ground truth image. 
    subj1 = nsda.stim_descriptions[nsda.stim_descriptions['subject1'] != 0]
        

    # Grabbing the models for a hash
    z_modelId = z_model_hash + "_model_" + Dz.vector + ".pt"
    c_modelId = c_model_hash + "_model_" + Dc.vector + ".pt"
    
    # Generating predicted and target vectors
    outputs_c, targets_c = Dc.predict(model=c_modelId, indices=idx)
    outputs_z, targets_z = Dz.predict(model=z_modelId, indices=idx)
    strength_c = 0.999999999999
    strength_z = 0.000000000001
    
    E = Encoder()
    
    for i in range(len(idx)):
        print(i)
        test_i = idx[i] + 25501
        
        # Make the c reconstrution images. 
        print(len(targets_z), targets_z[i].shape, len(targets_c), targets_c[i].shape)
        
        reconstructed_output_c = E.reconstruct(targets_z[i], outputs_c[i], strength_c)
        reconstructed_target_c = E.reconstruct(targets_z[i], targets_c[i], strength_c)
        
        # Make the z reconstrution images. 
        reconstructed_output_z = E.reconstruct(outputs_z[i], targets_c[i], strength_z)
        reconstructed_target_z = E.reconstruct(targets_z[i], targets_c[i], strength_z)
        
        # Make the z and c reconstrution images. 
        z_c_reconstruction = E.reconstruct(outputs_z[i], outputs_c[i], 0.9)
        
        index = int(subj1.loc[(subj1['subject1_rep0'] == test_i) | (subj1['subject1_rep1'] == test_i) | (subj1['subject1_rep2'] == test_i)].nsdId)

        # returns a numpy array 
        ground_truth_np_array = nsda.read_images([index], show=True)
        ground_truth = Image.fromarray(ground_truth_np_array[0])
        
        # Create figure
        fig = plt.figure(figsize=(10, 7))
        plt.title(str(i) + ": Reconstruction")
        
        # Setting values to rows and column variables
        rows = 3
        columns = 2
        
        # Adds a subplot at the 1st position
        fig.add_subplot(rows, columns, 1)
        
        # Showing image
        plt.imshow(ground_truth)
        plt.axis('off')
        plt.title("Ground Truth")
        
        # Adds a subplot at the 2nd position
        fig.add_subplot(rows, columns, 2)
        
        # Showing image
        plt.imshow(z_c_reconstruction)
        plt.axis('off')
        plt.title("Z and C Reconstructed")
        
        # Adds a subplot at the 1st position
        fig.add_subplot(rows, columns, 3)
        
        # Showing image
        plt.imshow(reconstructed_output_c)
        plt.axis('off')
        plt.title("Reconstructed Output C")
        
        # Adds a subplot at the 2nd position
        fig.add_subplot(rows, columns, 4)
        
        # Showing image
        plt.imshow(reconstructed_target_c)
        plt.axis('off')
        plt.title("Reconstructed Target C")
        
        # Adds a subplot at the 3rd position
        fig.add_subplot(rows, columns, 5)
        
        # Showing image
        plt.imshow(reconstructed_output_z)
        plt.axis('off')
        plt.title("Reconstructed Output Z")
        
        # Adds a subplot at the 4th position
        fig.add_subplot(rows, columns, 6)
        
        # Showing image
        plt.imshow(reconstructed_target_z)
        plt.axis('off')
        plt.title("Reconstructed Target Z")
        
        plt.savefig('reconstructions/' + str(i) + '_reconstruction.png')
    
if __name__ == "__main__":
    main()
