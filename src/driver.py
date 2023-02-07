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
from encoder import Encoder
from fracridge_decoder import RidgeDecoder
# from diffusers import StableDiffusionImageEncodingPipeline


# Good models: 
#
#
#  Masked voxels:
#    096_z2voxels.pt
#       - Model of 5051 voxels out of 11838 with a threshold of 0.07     (Used for training the decoder)
#    
#    173_c_prompt2voxels.pt 
#       - Model of 7569 voxels out of 11838 with a threshold of 0.070519 (Used for training the decoder)
#
#    174_c2voxels.pt 
#       - Model of 8483 voxels out of 11838 with a threshold of 0.063672 (Used for training the decoder)
#
#    206_z2voxels.pt (BEST Z MAP)
#      - Model of 6378 voxels out of 11838 with a threshold of 0.063478 (Used for training the decoder)
#
#    211_c_combined2voxels.pt (BEST C MAP)
#       - Model of 8144 voxels out of 11838 with a threshold of 0.058954 (Used for training the decoder)
#
#    224_c_img_mixer2voxels.pt
#       - Model of 8112 voxels out of 11838 with a threshold of 0.060343 (Used for training the decoder)
#
#    229_c_img_mixer_02voxels.pt
#       - Model of 8095 voxels out of 11838 with a threshold of 0.060507 (Used for training the decoder)
#
#    231_c_img2voxels.pt
#       - Model of 7690 voxels out of 11838 with a threshold of 0.070246 (Used for training the decoder)    
#
#    247_c_combined2voxels.pt
#       - Model of 7976 voxels out of 11838 with a threshold of 0.065142 (Used for training the decoder)
#
#    265_c_combined2voxels.pt
#       - Model of 7240 voxels out of 11838 with a threshold of 0.06398 (Used for training the decoder)
#
#    280_c_combined2voxels_pearson_thresh0.063339
#       - fracridge Mask of 7348 voxels with a threshold of 0.063339, calculated on old_normalized x
#
#    283_c_combined2voxels_pearson_thresh0.06397
#       - fracridge Mask of 7322 voxels with a threshold of 0.06397, calculated on Z scored X (DOESNT WORK IN FRACRIDGE, SCIPY ERROR)
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
# 218_model_c_combined.pt
#      - Model of 8144 voxels out of 11838 with a learning rate of 0.000002 and a threshold of 0.058954
#
# 221_model_z.pt (BEST Z MODEL)
#      - Model of 6378 voxels out of 11838 with a learning rate of 0.00001 and a threshold of 0.063478
#
# 227_model_c_img_mixer.pt
#      - Model of 8112 voxels out of 11838 with a learning rate of 0.000002 and a threshold of 0.060343
#
# 232_model_c_img.pt
#      - Model of 7690 voxels out of 11838 with a learning rate of 0.000002 and a threshold of 0.070246
#
# 266_model_c_combined.pt (BEST C MODEL)
#      - Model of 7240 voxels out of 11838 on old normalization method with a learning rate of 0.00005 and a threshold of 0.06398
def main():
    os.chdir("/export/raid1/home/kneel027/Second-Sight/")
    # train_hash = train_decoder()
    c_hash,_,_ = run_fr_decoder()
    # z = torch.load("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/latent_vectors/044_model_z.pt/output_1_z.pt")
    # c = torch.load("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/image_c_features2.pt")
    # E = Encoder()
    # E.reconstruct(z, c, 0.999999999999)
    reconstructNImages(z_model_hash="221",
                         c_model_hash="294",
                         c_thresh = 0.063339,
                         z_thresh = 0.063478,
                         idx=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])


def train_decoder():
    hashNum = update_hash()
    #hashNum = "179"
    D = Decoder(hashNum = hashNum,
                 lr=0.00005,
                 vector="c_combined", #c, z, c_prompt
                 threshold=0.066412, #0.063672 for c #174, 0.07 for z #141
                 inpSize=7920, # 8483 for c with thresh 0.063672, 5051 for z with thresh 0.07, #8144 for c_combined with thresh 0.058954, 6378 for z with thresh 0.063478
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
    return hashNum

def run_fr_decoder():
    hashNum = update_hash()
    #hashNum = "179"
    D = RidgeDecoder(hashNum = hashNum,
                vector="c_combined", 
                 log=True, 
                 threshold=0.06397,
                 device="cuda",
                 n_alphas=20
                 )
    hashNum, outputs, target = D.train()
    return hashNum, outputs, target

# Encode latent z (1x4x64x64) and condition c (1x77x1024) tensors into an image
# Strength parameter controls the weighting between the two tensors
def reconstructNImages(z_model_hash, c_model_hash, c_thresh, z_thresh, idx):
    Dz = Decoder(hashNum = z_model_hash,
                 vector="z", 
                 threshold=z_thresh,
                 inpSize = 6378,
                 log=False, 
                 device="cuda",
                 parallel=False
                 )
    # Dc = Decoder(hashNum = c_model_hash,
    #              vector="c_combined", 
    #              threshold=c_thresh,
    #              inpSize = 7240,
    #              log=False, 
    #              device="cuda",
    #              parallel=False
    #              )
    Dc = RidgeDecoder(hashNum = c_model_hash,
                vector="c_combined", 
                 log=False, 
                 threshold=c_thresh,
                 device="cuda",
                 n_alphas=20
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
    outputs_c, targets_c = Dc.predict(hashNum=c_model_hash, indices=idx)
    # outputs_c, targets_c = Dc.predict(model=c_modelId, indices=idx)
    outputs_z, targets_z = Dz.predict(model=z_modelId, indices=idx)
    strength_c = 1
    strength_z = 0
    E = Encoder()
    # E = StableDiffusionImageEncodingPipeline.from_pretrained(
    # "lambdalabs/sd-image-variations-diffusers",
    # revision="v2.0",
    # ).to("cuda")
    for i in range(len(idx)):
        print(i)
        test_i = idx[i] + 25501
        
        # Make the c reconstrution images. 
        print(len(targets_z), targets_z[i].shape, len(targets_c), targets_c[i].shape)
        # outputs_z[i] = torch.load("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/latent_vectors/044_model_z.pt/output_1_z.pt")
        # target_c = []
        # out_c = []
        # out_c.append(outputs_c[i])
        # target_c.append(targets_c[i])
        # for j in range(0,4):
        #     out_c.append(torch.zeros((1, 768), device="cuda"))
        #     target_c.append(torch.zeros((1, 768), device="cuda"))
        
        # out_c = torch.cat(out_c, dim=0).unsqueeze(0)
        # out_c = out_c.tile(1, 1, 1)
        # target_c = torch.cat(target_c, dim=0).unsqueeze(0)
        # target_c = target_c.tile(1, 1, 1)
        
        reconstructed_output_c = E.reconstruct(c=outputs_c[i], strength=strength_c)
        reconstructed_target_c = E.reconstruct(c=targets_c[i], strength=strength_c)
        
        # # Make the z reconstrution images. 
        reconstructed_output_z = E.reconstruct(z=outputs_z[i], strength=strength_z)
        reconstructed_target_z = E.reconstruct(z=targets_z[i], strength=strength_z)
        
        # # Make the z and c reconstrution images. 
        z_c_reconstruction = E.reconstruct(z=outputs_z[i], c=outputs_c[i], strength=0.75)
        # reconstructed_output_c = E.reconstruct(c=outputs_c[i].reshape((2,1,768)), strength=strength_c)
        # reconstructed_target_c = E.reconstruct(c=targets_c[i].reshape((2,1,768)), strength=strength_c)
        
        # # # Make the z reconstrution images. 
        # reconstructed_output_z = E.reconstruct(z=outputs_z[i].reshape((1,4,64,64)), strength=strength_z)
        # reconstructed_target_z = E.reconstruct(z=targets_z[i].reshape((1,4,64,64)), strength=strength_z)
        
        # # # Make the z and c reconstrution images. 
        # z_c_reconstruction = E.reconstruct(z=outputs_z[i].reshape((1,4,64,64)), c=outputs_c[i].reshape((2,1,768)), strength=0.75)
        
        # reconstructed_output_c = E.reconstruct(c=out_c, strength=strength_c)
        # reconstructed_target_c = E.reconstruct(c=target_c, strength=strength_c)
        
        # # Make the z and c reconstrution images. 
        # z_c_reconstruction = E.reconstruct(z=outputs_z[i], c=out_c, strength=0.75)
        
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
        
        
        plt.savefig('reconstructions/' + str(i) + '_reconstruction_c_combined_9.png')
    
if __name__ == "__main__":
    main()
