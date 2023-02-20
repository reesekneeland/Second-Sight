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


#   Encoders:
#      424_model_c_img_0.pt
# 
#      425_model_c_text_0.pt
#      
#      426_model_z_img_mixer.pt


def main():
    os.chdir("/export/raid1/home/kneel027/Second-Sight/")
    
    reconstructNImages(experiment_title="cc3m top 5 comparison new split",
                       idx=[i for i in range(22)])


def predictVector_cc3m(model, vector, x, device="cuda:1"):
        
        if(vector == "c_img_0" or vector == "c_text_0"):
            datasize = 768
        elif(vector == "z_img_mixer"):
            datasize = 16384
            
        prep_path = "/export/raid1/home/kneel027/nsd_local/preprocessed_data/"
        latent_path = "/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/latent_vectors/"
        
        PeC = PearsonCorrCoef(num_outputs=22735)
        outputPeC = PearsonCorrCoef(num_outputs=620)
        
        out = torch.zeros((x.shape[0], 5, datasize))
        average_pearson = 0
        
        for i in tqdm(range(x.shape[0]), desc="scanning library for " + vector):
            xDup = x[i].repeat(22735, 1).moveaxis(0, 1)
            batch_max_x = torch.zeros((620, x.shape[1]))
            batch_max_y = torch.zeros((620, datasize))
            for batch in tqdm(range(124), desc="batching sample"):
                y = torch.load(prep_path + vector + "/cc3m_batches/" + str(batch) + ".pt")
                x_preds = torch.load(latent_path + model + "/cc3m_batches/" + str(batch) + ".pt")
                x_preds_t = x_preds.moveaxis(0, 1)
                
                # Pearson correlation
                pearson = PeC(xDup, x_preds_t)

                # Calculating the Average Pearson Across Samples
                top5_pearson = torch.topk(pearson, 5).values
                average_pearson += torch.mean(top5_pearson) 
                
                top5_ind = torch.topk(pearson, 5).indices
                
                
                for j, index in enumerate(top5_ind):
                    batch_max_x[5*batch + j] = x_preds_t[:,index]
                    batch_max_y[5*batch + j] = y[index]
                    
                
            xDupOut = x[i].repeat(620, 1).moveaxis(0, 1).to(device)
            batch_max_x = batch_max_x.moveaxis(0, 1).to(device)
            outPearson = outputPeC(xDupOut, batch_max_x).to("cpu")
            top5_ind_out = torch.topk(outPearson, 5).indices
            for j, index in enumerate(top5_ind_out):
                    out[i, j] = batch_max_y[index]
            
        torch.save(out, latent_path + model + "/" + vector + "_cc3m_library_preds.pt")
        print("Average Pearson Across Samples: ", (average_pearson / x.shape[0]) ) 
        return out

# Encode latent z (1x4x64x64) and condition c (1x77x1024) tensors into an image
# Strength parameter controls the weighting between the two tensors
def reconstructNImages(experiment_title, idx):
    
    # First URL: This is the original read-only NSD file path (The actual data)
    # Second URL: Local files that we are adding to the dataset and need to access as part of the data
    # Object for the NSDAccess package
    nsda = NSDAccess('/home/naxos2-raid25/kneel027/home/surly/raid4/kendrick-data/nsd', '/home/naxos2-raid25/kneel027/home/kneel027/nsd_local')
    
    # Retriving the ground truth image. 
    subj1 = nsda.stim_descriptions[nsda.stim_descriptions['subject1'] != 0]
    
<<<<<<< HEAD
    # Load in the data
    _, _, x_test, _, _, targets_c_i, test_trials = load_data(vector="c_img_0", 
=======
    # Generating predicted and target vectors
    # outputs_c, targets_c = Dc.predict(hashNum=Dc.hashNum, indices=idx)
    # outputs_c_i, targets_c_i = Dc_i.predict(model=c_img_modelId)
    # outputs_c_i = [outputs_c_i[i] for i in idx]
     _, _, x_test, _, _, targets_c_i, test_trials = load_nsd(vector="c_img_0", 
                                                             loader=False)
    _, _, _, _, _, targets_c_t, _ = load_nsd(vector="c_text_0", 
                                              loader=False)
    _, _, _, _, _, targets_z, _ = load_nsd(vector="z_img_mixer", 
                                            loader=False)
    
    test_idx = [test_trials[i] for i in idx]
    # TODO: Run the 20 test x through the autoencoder to feed into predictVector_cc3m
    x_test = x_test[test_idx]
    targets_c_i = targets_c_i[test_idx]
    targets_c_t = targets_c_t[test_idx]
    targets_z = targets_z[test_idx]
    
    
    outputs_c_i = predictVector_cc3m(model="424_model_c_img_0.pt", vector="c_img_0", x=x_test)[:,0]
    # outputs_c_t = predictVector_cc3m(model="425_model_c_text_0.pt", vector="c_text_0", x=x_test)[:,0]
    # outputs_z = predictVector_cc3m(model="426_model_z_img_mixer.pt", vector="z_img_mixer", x=x_test)[:,0]
    # outputs_c_i = torch.load("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/latent_vectors/424_model_c_img_0.pt/c_img_0_cc3m_library_preds.pt")
    # outputs_c_t = torch.load("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/latent_vectors/425_model_c_text_0.pt/c_text_0_cc3m_library_preds.pt")
    # outputs_z = torch.load("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/latent_vectors/426_model_z_img_mixer.pt/z_img_mixer_cc3m_library_preds.pt")
    
    
    strength_c = 1
    strength_z = 0
    R = Reconstructor()
    for i in range(len(idx)-1):
        test_i = test_trials[idx[i+1]]
        brain_scan = x_test[idx[i]]
        # index = int(subj1x.loc[(subj1x['subject1_rep0'] == test_i) | (subj1x['subject1_rep1'] == test_i) | (subj1x['subject1_rep2'] == test_i)].nsdId)
        rootdir = "/home/naxos2-raid25/kneel027/home/kneel027/nsd_local/cc3m/tensors/"
        # outputs_c_i[i] = torch.load(rootdir + "c_img_0/" + str(i) + ".pt")
        # outputs_c_t[i] = torch.load(rootdir + "c_text_0/" + str(i) + ".pt")
        # outputs_z[i] = torch.load(rootdir + "z_img_mixer/" + str(i) + ".pt")
        print(i)
        
        print("shape: ", outputs_c_i[i].shape)
        c_combined = format_clip(outputs_c_i[i])
        print(targets_c_i.shape, targets_c_i[0].shape)
        c_combined_target = format_clip(targets_c_i[i])
        c_0 = format_clip(outputs_c_i[i,0])
        c_1 = format_clip(outputs_c_i[i,1])
        c_2 = format_clip(outputs_c_i[i,2])
        c_3 = format_clip(outputs_c_i[i,3])
        c_4 = format_clip(outputs_c_i[i,4])
        
    
        # Make the c reconstrution images. 
        reconstructed_output_c = R.reconstruct(c=c_combined, strength=strength_c)
        reconstructed_target_c = R.reconstruct(c=c_combined_target, strength=strength_c)
        
        reconstructed_output_c_0 = R.reconstruct(c=c_0, strength=strength_c)
        reconstructed_output_c_1 = R.reconstruct(c=c_1, strength=strength_c)
        reconstructed_output_c_2 = R.reconstruct(c=c_2, strength=strength_c)
        reconstructed_output_c_3 = R.reconstruct(c=c_3, strength=strength_c)
        reconstructed_output_c_4 = R.reconstruct(c=c_4, strength=strength_c)
        
        # # Make the z reconstrution images. 
        # reconstructed_output_z = R.reconstruct(z=outputs_z[i], strength=strength_z)
        # reconstructed_target_z = R.reconstruct(z=targets_z[i], strength=strength_z)
        
        # # Make the z and c reconstrution images. 
        # z_c_reconstruction = R.reconstruct(z=outputs_z[i], c=outputs_c_i[i], strength=0.8)
        
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
        plt.imshow(reconstructed_target_c)
        plt.axis('off')
        plt.title("Target C")
        
        # Adds a subplot at the 1st position
        fig.add_subplot(rows, columns, 3)
        
        # Showing image
        plt.imshow(reconstructed_output_c)
        plt.axis('off')
        plt.title("Output 5 Cs")
        
        # Adds a subplot at the 2nd position
        fig.add_subplot(rows, columns, 4)
        
       # Showing image
        plt.imshow(reconstructed_output_c_0)
        plt.axis('off')
        plt.title("C_0")
        
        # Adds a subplot at the 1st position
        fig.add_subplot(rows, columns, 5)
        
        # Showing image
        plt.imshow(reconstructed_output_c_1)
        plt.axis('off')
        plt.title("C_1")
        
        
        # Adds a subplot at the 2nd position
        fig.add_subplot(rows, columns, 6)
        
        # Showing image
        plt.imshow(reconstructed_output_c_2)
        plt.axis('off')
        plt.title("C_2")
        
        # Adds a subplot at the 3rd position
        fig.add_subplot(rows, columns, 7)
        
        # Showing image
        plt.imshow(reconstructed_output_c_3)
        plt.axis('off')
        plt.title("C_3")
    
        # Adds a subplot at the 4th position
        fig.add_subplot(rows, columns, 8)
        
        # Showing image
        plt.imshow(reconstructed_output_c_4)
        plt.axis('off')
        plt.title("C_4")
        
        
        os.makedirs("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/reconstructions/" + experiment_title + "/", exist_ok=True)
        plt.savefig('reconstructions/' + experiment_title + '/' + str(i) + '.png', dpi=400)
    
if __name__ == "__main__":
    main()
