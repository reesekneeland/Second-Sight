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
from reconstructor import Reconstructor



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
#    313_c_img_02voxels.pt
#       - 7643
#       - old normalization method
#       - 0.062136
#
#    316_c_text_02voxels.pt
#       - 6650
#       - old normalization method
#       - 0.067784
#
#    320_z_img_mixer2voxels.pt
#       - Model of 5615 voxels out of 11838 with a threshold of 0.064564 (Used for training the decoder)
#       - compound loss: 0.11211
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
# 221_model_z.pt 
#      - Model of 6378 voxels out of 11838 with a learning rate of 0.00001 and a threshold of 0.063478
#
# 227_model_c_img_mixer.pt
#      - Model of 8112 voxels out of 11838 with a learning rate of 0.000002 and a threshold of 0.060343
#
# 232_model_c_img.pt
#      - Model of 7690 voxels out of 11838 with a learning rate of 0.000002 and a threshold of  0.070246
#
# 266_model_c_combined.pt 
#      - Model of 7240 voxels out of 11838 on old normalization method with a learning rate of 0.00005 and a threshold of 0.06398
#
# 318_model_c_img_0.pt 
#    - 7643
#    - old normalization method
#    - 0.062136
#
# 319_model_c_text_0.pt 
#    - 6650
#    - old normalization method
#    - 0.067784
#
# 322_z_img_mixer2voxels.pt 
#    - 5615
#    - old normalization method
#    - 0.064564 
#    - compound_loss: 0.6940
#
# 373_model_c_img_0.pt (BEST C MODEL PART 1)
#    - 7643
#    - trained on new MLP
#    - old normalization method
#    - 0.062136
# 375_model_c_text_0.pt (BEST C MODEL PART 2)
#    - 6650
#    - trained on new MLP
#    - old normalization method
#    - 0.067784
# 377_model_z_img_mixer.pt (BEST Z MODEL)
#    - 5615
#    - trained on new MLP
#    - old normalization method
#    - 0.064564 
#    - compound_loss: 0.6940
#
#
#   Encoders:
#      - 374_model_c_img_0.pt
#           - old normalization method
#           - Mean: 0.13217218
#
#      - 376_model_c_text_0.pt
#           - old normalization
#           - Mean: 0.09641416
#
#      - 378_model_z_img_mixer.pt
#           - old normalization
#           - Mean: 0.07613872


def main(decode, encode):
    os.chdir("/export/raid1/home/kneel027/Second-Sight/")
    
    if(decode):
        train_hash = train_decoder()
    elif(encode):
        encoder_hash = train_encoder()
    else:
        reconstructNImages(experiment_title="73k COCO Library Decoder",
                       idx=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])


def train_encoder():
    #hashNum = update_hash()
    hashNum = "378"
    E = Encoder(hashNum = hashNum,
                 lr=0.0001,
                 vector="z_img_mixer", #c, z, c_prompt
                 log=True, 
                 batch_size=750,
                 parallel=False,
                 device="cuda:0",
                 num_workers=16,
                 epochs=300
                )
    #E.train()
    modelId = E.hashNum + "_model_" + E.vector + ".pt"
    
    outputs_c = E.predict(model=modelId)
    # Test
    # modelId_z = "044" + "_model_" + "z" + ".pt"
    # outputs_z, targets_z = D.predict(model=modelId_z, indices=[1, 2, 3])
    # cosSim = nn.CosineSimilarity(dim=0)
    return hashNum

def train_decoder():
    hashNum = update_hash()
    # hashNum = "361"
    D = Decoder(hashNum = hashNum,
                 lr=0.00005,
                 vector="z_img_mixer", #c, z, c_prompt
                 log=True, 
                 threshold=0.064564,
                 inpSize = 5615,
                 batch_size=750,
                 parallel=False,
                 device="cuda:0",
                 num_workers=16,
                 epochs=300
                )
    D.train()
    modelId = D.hashNum + "_model_" + D.vector + ".pt"
    
    outputs_c, targets_c = D.predict(model=modelId)
    # Test
    # modelId_z = "044" + "_model_" + "z" + ".pt"
    # outputs_z, targets_z = D.predict(model=modelId_z, indices=[1, 2, 3])
    cosSim = nn.CosineSimilarity(dim=0)
    print(cosSim(outputs_c[0], targets_c[0]))
    print(cosSim(torch.randn_like(outputs_c[0]), targets_c[0]))
    return hashNum

# Encode latent z (1x4x64x64) and condition c (1x77x1024) tensors into an image
# Strength parameter controls the weighting between the two tensors
def reconstructNImages(experiment_title, idx):
    # First URL: This is the original read-only NSD file path (The actual data)
    # Second URL: Local files that we are adding to the dataset and need to access as part of the data
    # Object for the NSDAccess package
    nsda = NSDAccess('/home/naxos2-raid25/kneel027/home/surly/raid4/kendrick-data/nsd', '/home/naxos2-raid25/kneel027/home/kneel027/nsd_local')
    
    # Retriving the ground truth image. 
    subj1 = nsda.stim_descriptions[nsda.stim_descriptions['subject1'] != 0]
        

    # Grabbing the models for a hash
    # z_modelId = Dz.hashNum + "_model_" + Dz.vector + ".pt"
    # c_img_modelId = Dc_i.hashNum + "_model_" + Dc_i.vector + ".pt"
    # c_text_modelId = Dc_t.hashNum + "_model_" + Dc_t.vector + ".pt"
    
    # Generating predicted and target vectors
    # outputs_c, targets_c = Dc.predict(hashNum=Dc.hashNum, indices=idx)
    # outputs_c_i, targets_c_i = Dc_i.predict(model=c_img_modelId)
    # outputs_c_i = [outputs_c_i[i] for i in idx]
    _, x_test, _, targets_c_i = load_data_masked("c_img_0")
    
    _, _, _, targets_c_t = load_data_masked("c_text_0")
    _, _, _, targets_z = load_data_masked("z_img_mixer")
    x_test = x_test[idx]
    targets_c_i = targets_c_i[idx]
    targets_c_t = targets_c_t[idx]
    targets_z = targets_z[idx]
    outputs_c_i = predictVector(model="374_model_c_img_0.pt", vector="c_img_0", x=x_test)
    outputs_c_t = predictVector(model="376_model_c_text_0.pt", vector="c_text_0", x=x_test)
    outputs_z = predictVector(model="378_model_z_img_mixer.pt", vector="z_img_mixer", x=x_test)
    # outputs_c_i = torch.load("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/latent_vectors/374_model_c_img_0.pt/c_img_0_library_preds.pt")
    # outputs_c_t = torch.load("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/latent_vectors/376_model_c_text_0.pt/c_text_0_library_preds.pt")
    # outputs_z = torch.load("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/latent_vectors/378_model_z_img_mixer.pt/z_img_mixer_library_preds.pt")
    strength_c = 1
    strength_z = 0
    R = Reconstructor()
    for i in range(len(idx)):
        test_i = idx[i] + 25501
        brain_scan = x_test[idx[i]]
        # index = int(subj1x.loc[(subj1x['subject1_rep0'] == test_i) | (subj1x['subject1_rep1'] == test_i) | (subj1x['subject1_rep2'] == test_i)].nsdId)
        
        print(i)
        
        print("shape: ", outputs_c_i[i].shape)
        c_combined, c_combined_target, c_img, c_img_target = [], [], [], []
        c_combined.append(outputs_c_i[i].reshape((1,768)).to("cuda"))
        c_img.append(outputs_c_i[i].reshape((1,768)).to("cuda"))
        c_img.append(torch.zeros((1, 768), device="cuda"))
        c_img_target.append(targets_c_i[i].reshape((1,768)).to("cuda"))
        c_img_target.append(torch.zeros((1, 768), device="cuda"))
        c_combined_target.append(targets_c_i[i].reshape((1,768)).to("cuda"))
        c_combined.append(outputs_c_t[i].reshape((1,768)).to("cuda"))
        c_combined_target.append(targets_c_t[i].reshape((1,768)).to("cuda"))
        for j in range(0,3):
            c_combined.append(torch.zeros((1, 768), device="cuda"))
            c_combined_target.append(torch.zeros((1, 768), device="cuda"))
            c_img.append(torch.zeros((1, 768), device="cuda"))
            c_img_target.append(torch.zeros((1, 768), device="cuda"))
        
        c_combined = torch.cat(c_combined, dim=0).unsqueeze(0)
        c_combined = c_combined.tile(1, 1, 1)
        c_img = torch.cat(c_img, dim=0).unsqueeze(0)
        c_img = c_img.tile(1, 1, 1)
        c_combined_target = torch.cat(c_combined_target, dim=0).unsqueeze(0)
        c_combined_target = c_combined_target.tile(1, 1, 1)
        c_img_target = torch.cat(c_img_target, dim=0).unsqueeze(0)
        c_img_target = c_img_target.tile(1, 1, 1)
    
        # Make the c reconstrution images. 
        
        reconstructed_output_c = R.reconstruct(c=c_combined, strength=strength_c)
        reconstructed_target_c = R.reconstruct(c=c_combined_target, strength=strength_c)
        
        reconstructed_output_c_img = R.reconstruct(c=c_img, strength=strength_c)
        reconstructed_target_c_img = R.reconstruct(c=c_img_target, strength=strength_c)
        
        # # Make the z reconstrution images. 
        reconstructed_output_z = R.reconstruct(z=outputs_z[i], strength=strength_z)
        reconstructed_target_z = R.reconstruct(z=targets_z[i], strength=strength_z)
        
        # # Make the z and c reconstrution images. 
        z_c_reconstruction = R.reconstruct(z=outputs_z[i], c=c_combined, strength=0.8)
        
        index = int(subj1.loc[(subj1['subject1_rep0'] == test_i) | (subj1['subject1_rep1'] == test_i) | (subj1['subject1_rep2'] == test_i)].nsdId)

        # returns a numpy array 
        ground_truth_np_array = nsda.read_images([index], show=True)
        ground_truth = Image.fromarray(ground_truth_np_array[0])
        
        # Create figure
        fig = plt.figure(figsize=(10, 7))
        plt.title(str(i) + ": " + experiment_title)
        
        # Setting values to rows and column variables
        rows = 4
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
        
         # Adds a subplot at the 1st position
        fig.add_subplot(rows, columns, 5)
        
         # Showing image
        plt.imshow(reconstructed_target_c_img)
        plt.axis('off')
        plt.title("Reconstructed Target C_img")
        
        
        # Adds a subplot at the 2nd position
        fig.add_subplot(rows, columns, 6)
        
       # Showing image
        plt.imshow(reconstructed_output_c_img)
        plt.axis('off')
        plt.title("Reconstructed Output C_img")
        
        # Adds a subplot at the 3rd position
        fig.add_subplot(rows, columns, 7)
        
        # Showing image
        plt.imshow(reconstructed_target_z)
        plt.axis('off')
        plt.title("Reconstructed Target Z")
    
        # Adds a subplot at the 4th position
        fig.add_subplot(rows, columns, 8)
        
        # Showing image
        plt.imshow(reconstructed_output_z)
        plt.axis('off')
        plt.title("Reconstructed Output Z")
        
        
        os.makedirs("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/reconstructions/" + experiment_title + "/", exist_ok=True)
        plt.savefig('reconstructions/' + experiment_title + '/' + str(i) + '.png', dpi=400)
    
if __name__ == "__main__":
    main(decode=False, encode=False)