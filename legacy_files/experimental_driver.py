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
from encoder import Encoder, load_img
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
    # c_hash,_,_ = run_fr_decoder()
    # z = torch.load("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/latent_vectors/044_model_z.pt/output_1_z.pt")
    # c = torch.load("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/image_c_features2.pt")
    # E = Encoder()
    # E.reconstruct(z, c, 0.999999999999)
    reconstructNImages(z_model_hash="322",
                         c_model_hash="294",
                         c_thresh = 0.063339,
                         z_thresh = 0.064564,
                         idx=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])


def train_decoder():
    hashNum = update_hash()
    #hashNum = "179"
    D = Decoder(hashNum = hashNum,
                 lr=0.00005,
                 vector="z_img_mixer", #c, z, c_prompt
                 threshold=0.064564, #0.063672 for c #174, 0.07 for z #141
                 inpSize=5615, # 8483 for c with thresh 0.063672, 5051 for z with thresh 0.07, #8144 for c_combined with thresh 0.058954, 6378 for z with thresh 0.063478
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
                 vector="z_img_mixer", 
                 threshold=z_thresh,
                 inpSize = 5615,
                 log=False, 
                 device="cuda",
                 parallel=False
                 )
    Dc_i = Decoder(hashNum = "318",
                 vector="c_img_0", 
                 threshold=0.062136,
                 inpSize = 7643,
                 log=False, 
                 device="cuda",
                 parallel=False
                 )
    Dc_t = Decoder(hashNum = "319",
                 vector="c_text_0", 
                 threshold=0.067784,
                 inpSize = 6650,
                 log=False, 
                 device="cuda",
                 parallel=False
                 )
    Dc = Decoder(hashNum = "266",
                 vector="c_combined", 
                 threshold=0.06398,
                 inpSize = 7240,
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
    c_img_modelId = "318_model_c_img_0.pt"
    c_text_modelId = "319_model_c_text_0.pt"
    c_modelId = "266_model_c_combined.pt"
    
    # Generating predicted and target vectors
    outputs_c_i, targets_c_i = Dc_i.predict(model=c_img_modelId, indices=idx)
    outputs_c_t, targets_c_t = Dc_t.predict(model=c_text_modelId, indices=idx)
    outputs_c, targets_c = Dc.predict(model=c_modelId, indices=idx)
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
        # print(len(targets_z), targets_z[i].shape, len(targets_c), targets_c[i].shape)
        # outputs_z[i] = torch.load("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/latent_vectors/044_model_z.pt/output_1_z.pt")
        c_combined = []
        c_combined_target = []
        c_img = []
        c_img_target = []
        c_added = []
        c_added_target = []
        c_added.append(outputs_c_i[i] + outputs_c_t[i])
        c_added_target.append(targets_c_i[i] + targets_c_t[i])
        c_added.append(torch.zeros((1, 768), device="cuda"))
        c_added_target.append(torch.zeros((1, 768), device="cuda"))
        c_combined.append(outputs_c_i[i])
        c_img.append(outputs_c_i[i])
        c_combined_target.append(targets_c_i[i])
        c_img_target.append(targets_c_i[i])
        c_combined.append(outputs_c_t[i])
        c_img.append(torch.zeros((1, 768), device="cuda"))
        c_combined_target.append(targets_c_t[i])
        c_img_target.append(torch.zeros((1, 768), device="cuda"))
        for j in range(0,3):
            c_combined.append(torch.zeros((1, 768), device="cuda"))
            c_combined_target.append(torch.zeros((1, 768), device="cuda"))
            c_img.append(torch.zeros((1, 768), device="cuda"))
            c_img_target.append(torch.zeros((1, 768), device="cuda"))
            c_added.append(torch.zeros((1, 768), device="cuda"))
            c_added_target.append(torch.zeros((1, 768), device="cuda"))
        
        c_combined = torch.cat(c_combined, dim=0).unsqueeze(0)
        c_combined = c_combined.tile(1, 1, 1)
        c_combined_target = torch.cat(c_combined_target, dim=0).unsqueeze(0)
        c_combined_target = c_combined_target.tile(1, 1, 1)
        c_img = torch.cat(c_img, dim=0).unsqueeze(0)
        c_img = c_img.tile(1, 1, 1)
        c_img_target = torch.cat(c_img_target, dim=0).unsqueeze(0)
        c_img_target = c_img_target.tile(1, 1, 1)
        c_added = torch.cat(c_added, dim=0).unsqueeze(0)
        c_added = c_added.tile(1, 1, 1)
        c_added_target = torch.cat(c_added_target, dim=0).unsqueeze(0)
        c_added_target = c_added_target.tile(1, 1, 1)
        
        reconstructed_output_c_combined = E.reconstruct(c=c_combined, strength=strength_c)
        reconstructed_target_c_combined = E.reconstruct(c=c_combined_target, strength=strength_c)
        
        # # Make the z reconstrution images. 
        reconstructed_output_c_img = E.reconstruct(c=c_img, strength=strength_c)
        reconstructed_target_c_img = E.reconstruct(c=c_img_target, strength=strength_c)

        reconstructed_output_c_trained_whole = E.reconstruct(c=outputs_c[i], strength=strength_c)
        reconstructed_target_c_trained_whole = E.reconstruct(c=targets_c[i], strength=strength_c)
        
        reconstructed_output_c_added = E.reconstruct(c=c_added, strength=strength_c)
        reconstructed_target_c_added = E.reconstruct(c=c_added_target, strength=strength_c)
        
        reconstructed_output_z = E.reconstruct(z=outputs_z[i], strength=strength_z)
        reconstructed_output_z_and_c = E.reconstruct(z=outputs_z[i], c=c_combined, strength=0.75)
        
        index = int(subj1.loc[(subj1['subject1_rep0'] == test_i) | (subj1['subject1_rep1'] == test_i) | (subj1['subject1_rep2'] == test_i)].nsdId)

        # returns a numpy array 
        ground_truth_np_array = nsda.read_images([index], show=True)
        ground_truth = Image.fromarray(ground_truth_np_array[0])
        
        # Create figure
        fig = plt.figure(figsize=(10, 7))
        plt.title(str(i) + ": Reconstruction")
        
        # Setting values to rows and column variables
        rows = 6
        columns = 2
        
        fig.add_subplot(rows, columns, 1)
        # Showing image
        plt.imshow(ground_truth)
        plt.axis('off')
        plt.title("Ground Truth")

        fig.add_subplot(rows, columns, 3)
        
         # Showing image
        plt.imshow(reconstructed_target_c_combined)
        plt.axis('off')
        plt.title("Target C_combined")
        
        
        fig.add_subplot(rows, columns, 4)
        
       # Showing image
        plt.imshow(reconstructed_output_c_combined)
        plt.axis('off')
        plt.title("Output C_combined")
        
        fig.add_subplot(rows, columns, 5)
        
        # Showing image
        plt.imshow(reconstructed_target_c_img)
        plt.axis('off')
        plt.title("Target C_img")
    
        fig.add_subplot(rows, columns, 6)
        
        # Showing image
        plt.imshow(reconstructed_output_c_img)
        plt.axis('off')
        plt.title("Output C_img")
        
        fig.add_subplot(rows, columns, 7)
        
        # Showing image
        plt.imshow(reconstructed_target_c_trained_whole)
        plt.axis('off')
        plt.title("Target c_combined trained whole")
        
        fig.add_subplot(rows, columns, 8)
        
        # Showing image
        plt.imshow(reconstructed_output_c_trained_whole)
        plt.axis('off')
        plt.title("Output c_combined trained whole")
        
        fig.add_subplot(rows, columns, 9)
        
        # Showing image
        plt.imshow(reconstructed_target_c_added)
        plt.axis('off')
        plt.title("Target c added")
        
        fig.add_subplot(rows, columns, 10)
        
        # Showing image
        plt.imshow(reconstructed_output_c_added)
        plt.axis('off')
        plt.title("Output c added")
        
        fig.add_subplot(rows, columns, 11)
        
        # Showing image
        plt.imshow(reconstructed_output_z)
        plt.axis('off')
        plt.title("z only")
        
        fig.add_subplot(rows, columns, 12)
        
        # Showing image
        plt.imshow(reconstructed_output_z_and_c)
        plt.axis('off')
        plt.title("c_combined and z")

        
        
        experiment_name = "individual_encoder_test"
        os.makedirs("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/reconstructions/" + experiment_name + "/", exist_ok=True)
        plt.savefig('reconstructions/' + experiment_name + '/' + str(i) + '.png', dpi=400)
    
if __name__ == "__main__":
    main()
