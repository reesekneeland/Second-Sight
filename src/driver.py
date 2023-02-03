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
from decoder import Decoder
from diffusers import StableDiffusionImageEncodingPipeline


# Good models: 
#
#
#  Masked voxels:
#    096_z2voxels.pt
#       - Model of 5051 voxels out of 11838 with a threshold of 0.07     (Used for training the driver)
#    
#    173_c_prompt2voxels.pt 
#       - Model of 7569 voxels out of 11838 with a threshold of 0.070519 (Used for training the driver)
#
#    174_c2voxels.pt 
#       - Model of 8483 voxels out of 11838 with a threshold of 0.063672 (Used for training the driver)
#
# 141_model_z.pt 
#      - Model of 5051 voxels out of 11383 with a learning rate of 0.000003 and a threshold of 0.07
#
# 148_model_c.pt 
#      - Model of 1729 voxels out of 11383 with a learning rate of 0.00001 and a threshold of 0.08
#
# 126_model_c_img.pt
#      - Model of 7372 voxels out of 11383 with a learning rate of 0.0000025 and a threshold of 0.06734
# 
# 155_model_z_normalization_test.pt (Normalization Test)
#      - Model of 5051 voxels out of 11383 with a learning rate of 0.0000001 and a threshold of 0.06734
#
# 

def main():
    os.chdir("/export/raid1/home/kneel027/Second-Sight/")
    train_decoder()

    # z = torch.load("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/latent_vectors/044_model_z.pt/output_1_z.pt")
    # c = torch.load("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/image_c_features2.pt")
    # E = Encoder()
    # E.reconstruct(z, c, 0.999999999999)
    reconstructNImages(z_model_hash="141",
                         c_model_hash="126",
                         c_thresh = 0.06734,
                         z_thresh = 0.07,
                         idx=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])#, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])


def train_decoder():
    hashNum = update_hash()
    #hashNum = "179"
    D = Decoder(hashNum = hashNum,
                 lr=0.00000001,
                 vector="c", #c, z, c_prompt
                 threshold=0.063672, #0.063672 for c #174, 0.07 for z #141
                 inpSize=8483, # 8483 for c with thresh 0.063672, 5051 for z with thresh 0.07
                 log=True, 
                 batch_size=750,
                 parallel=False,
                 device="cuda:1",
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
def reconstructNImages(z_model_hash, c_model_hash, c_thresh, z_thresh, idx):
    Dz = Decoder(hashNum = z_model_hash,
                 vector="z", 
                 threshold=z_thresh,
                 inpSize = 5051,
                 log=False, 
                 device="cuda",
                 parallel=False
                 )
    Dc = Decoder(hashNum = c_model_hash,
                 vector="c", 
                 threshold=c_thresh,
                 inpSize = 7372,
                 log=False, 
                 device="cuda",
                 parallel=False
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
    strength_c = 1
    strength_z = 0
    E = StableDiffusionImageEncodingPipeline.from_pretrained("lambdalabs/sd-image-variations-diffusers",revision="v2.0")
    E = E.to("cuda")
    
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
        z_c_reconstruction = E.reconstruct(outputs_z[i], outputs_c[i], 0.8)
        
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
        plt.imshow(reconstructed_target_c)
        plt.axis('off')
        plt.title("Reconstructed Target C")
        
        
        # Adds a subplot at the 2nd position
        fig.add_subplot(rows, columns, 4)
        
       # Showing image
        plt.imshow(reconstructed_output_c)
        plt.axis('off')
        plt.title("Reconstructed Output C")
        
        # Adds a subplot at the 3rd position
        fig.add_subplot(rows, columns, 5)
        
        # Showing image
        plt.imshow(reconstructed_target_z)
        plt.axis('off')
        plt.title("Reconstructed Target Z")
    
        # Adds a subplot at the 4th position
        fig.add_subplot(rows, columns, 6)
        
        # Showing image
        plt.imshow(reconstructed_output_z)
        plt.axis('off')
        plt.title("Reconstructed Output Z")
        
        
        plt.savefig('reconstructions/' + str(i) + '_reconstruction_test.png')
    
if __name__ == "__main__":
    main()
