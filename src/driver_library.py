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
        reconstructNImages(experiment_title="cc3m top 5 comparison new split",
                       idx=[i for i in range(22)])


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
                 device="cuda:1",
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

def predictVector_cc3m(model, vector, x, device="cuda:0"):
        if(vector == "c_img_0" or vector == "c_text_0"):
            datasize = 768
        elif(vector == "z_img_mixer"):
            datasize = 16384
        prep_path = "/export/raid1/home/kneel027/nsd_local/preprocessed_data/"
        latent_path = "/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/latent_vectors/"
        # y = torch.load(prep_path + vector + "/vector_cc3m.pt").requires_grad_(False)

        # y = y.detach()
        # x_preds = x_preds.detach()
        PeC = PearsonCorrCoef(num_outputs=22735).to(device)
        outputPeC = PearsonCorrCoef(num_outputs=620).to(device)
        # loss = nn.MSELoss(reduction='none')
        out = torch.zeros((x.shape[0], 5, datasize))
        for i in tqdm(range(x.shape[0]), desc="scanning library for " + vector):
            xDup = x[i].repeat(22735, 1).moveaxis(0, 1).to(device)
            batch_max_x = torch.zeros((620, x.shape[1])).to(device)
            batch_max_y = torch.zeros((620, datasize)).to(device)
            for batch in tqdm(range(124), desc="batching sample"):
                y = torch.load(prep_path + vector + "/cc3m_batches/" + str(batch) + ".pt").to(device)
                x_preds = torch.load(latent_path + model + "/cc3m_batches/" + str(batch) + ".pt")
                x_preds_t = x_preds.moveaxis(0, 1).to(device)
                # Pearson correlation

                # TODO: Grab the pearson score 
                pearson = PeC(xDup, x_preds_t)

                # TODO: Get the top five vectors values not indices (Average them and then make a running total and add
                #  them to a counter variable and then divide by the number of samples)
                top5_ind = torch.topk(pearson, 5).indices
                for j, index in enumerate(top5_ind):
                    batch_max_x[5*batch + j] = x_preds_t[:,index]
                    batch_max_y[5*batch + j] = y[index]
            xDupOut = x[i].repeat(620, 1).moveaxis(0, 1).to(device)
            batch_max_x = batch_max_x.moveaxis(0, 1).to(device)
            outPearson = outputPeC(xDupOut, batch_max_x)
            top5_ind_out = torch.topk(outPearson, 5).indices
            for j, index in enumerate(top5_ind_out):
                    out[i, j] = batch_max_y[index] 
            print("max of pred: ", out[i].max())
        torch.save(out, latent_path + model + "/" + vector + "_cc3m_library_preds.pt")
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
        

    # Grabbing the models for a hash
    # z_modelId = Dz.hashNum + "_model_" + Dz.vector + ".pt"
    # c_img_modelId = Dc_i.hashNum + "_model_" + Dc_i.vector + ".pt"
    # c_text_modelId = Dc_t.hashNum + "_model_" + Dc_t.vector + ".pt"
    
    # Generating predicted and target vectors
    # outputs_c, targets_c = Dc.predict(hashNum=Dc.hashNum, indices=idx)
    # outputs_c_i, targets_c_i = Dc_i.predict(model=c_img_modelId)
    # outputs_c_i = [outputs_c_i[i] for i in idx]
     _, _, x_test, _, _, targets_c_i, test_trials = load_data(vector="c_img_0", 
                                                             loader=False)
    _, _, _, _, _, targets_c_t, _ = load_data(vector="c_text_0", 
                                              loader=False)
    _, _, _, _, _, targets_z, _ = load_data(vector="z_img_mixer", 
                                            loader=False)
    test_idx = test_trials[idx]
    # TODO: Run the 20 test x through the autoencoder to feed into predictVector_cc3m
    x_test = x_test[test_idx]
    targets_c_i = targets_c_i[test_idx]
    targets_c_t = targets_c_t[test_idx]
    targets_z = targets_z[test_idx]
    # use z score models here not 417
    outputs_c_i = predictVector_cc3m(model="417_model_c_img_0.pt", vector="c_img_0", x=x_test)[:,0]
    # outputs_c_t = predictVector_cc3m(model="411_model_c_text_0.pt", vector="c_text_0", x=x_test)[:,0]
    # outputs_z = predictVector_cc3m(model="412_model_z_img_mixer.pt", vector="z_img_mixer", x=x_test)[:,0]
    # outputs_c_i = torch.load("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/latent_vectors/410_model_c_img_0.pt/c_img_0_cc3m_library_preds.pt")
    # outputs_c_t = torch.load("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/latent_vectors/411_model_c_text_0.pt/c_text_0_cc3m_library_preds.pt")
    # outputs_z = torch.load("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/latent_vectors/412_model_z_img_mixer.pt/z_img_mixer_cc3m_library_preds.pt")
    strength_c = 1
    strength_z = 0
    R = Reconstructor()
    for i in range(len(idx)-1):
        test_i = test_trials[i+1]
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
    main(decode=False, encode=False)