import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3"
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
from autoencoder  import AutoEncoder
from pearson import PearsonCorrCoef, pearson_corrcoef


#   Encoders:
#      424_model_c_img_0.pt
# 
#      425_model_c_text_0.pt
#      
#      426_model_z_img_mixer.pt


def main():
    os.chdir("/export/raid1/home/kneel027/Second-Sight/")
    benchmark_library(encModel="536_model_c_img_0.pt", vector="c_img_0", device="cuda:1")
    # reconstructNImages(experiment_title="cc3m top 5 comparison new split 536 tiled", idx=[i for i in range(3)])


def predictVector_cc3m(encModel, vector, x, device="cuda:0"):
        
        if(vector == "c_img_0" or vector == "c_text_0"):
            datasize = 768
        elif(vector == "z_img_mixer"):
            datasize = 16384
        # x = x.to(device)
        prep_path = "/export/raid1/home/kneel027/nsd_local/preprocessed_data/"
        latent_path = "/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/latent_vectors/"
        
        PeC = PearsonCorrCoef(num_outputs=22735).to(device)
        outputPeC = PearsonCorrCoef(num_outputs=620).to(device)
        
        out = torch.zeros((x.shape[0], 5, datasize))
        average_pearson = 0
        
        for i in tqdm(range(x.shape[0]), desc="scanning library for " + vector):
            xDup = x[i].repeat(22735, 1).moveaxis(0, 1).to(device)
            scores = torch.zeros((2819141,))
            preds = torch.zeros((2819141,768))
            # batch_max_x = torch.zeros((620, x.shape[1]))
            # batch_max_y = torch.zeros((620, datasize))
            for batch in tqdm(range(124), desc="batching sample"):
                y = torch.load(prep_path + vector + "/cc3m_batches/" + str(batch) + ".pt")
                x_preds = torch.load(latent_path + encModel + "/cc3m_batches/" + str(batch) + ".pt")
                x_preds_t = x_preds.moveaxis(0, 1).to(device)
                preds[22735*batch:22735*batch+22735] = y.detach()
                # Pearson correlation
                scores[22735*batch:22735*batch+22735] = PeC(xDup, x_preds_t).detach()
                # Calculating the Average Pearson Across Samples
            top5_pearson = torch.topk(scores, 5)
            average_pearson += torch.mean(top5_pearson.values.detach()) 
            print(top5_pearson.indices, top5_pearson.values, scores[0:5])
                
                # for j, index in enumerate(top5_pearson.indices):
                #     batch_max_x[5*batch + j] = x_preds_t[:,index].detach()
                #     batch_max_y[5*batch + j] = y[index].detach()
                    
                
            # xDupOut = x[i].repeat(620, 1).moveaxis(0, 1).to(device)
            # batch_max_x = batch_max_x.moveaxis(0, 1).to(device)
            # outPearson = outputPeC(xDupOut, batch_max_x).to("cpu")
            # top5_ind_out = torch.topk(outPearson, 5).indices
            for j, index in enumerate(top5_pearson.indices):
                    out[i, j] = preds[index]
            
        torch.save(out, latent_path + encModel + "/" + vector + "_cc3m_library_preds.pt")
        print("Average Pearson Across Samples: ", (average_pearson / x.shape[0]) ) 
        return out


# Encode latent z (1x4x64x64) and condition c (1x77x1024) tensors into an image
# Strength parameter controls the weighting between the two tensors
def reconstructNImages(experiment_title, idx):
    
    # First URL: This is the original read-only NSD file path (The actual data)
    # Second URL: Local files that we are adding to the dataset and need to access as part of the data
    # Object for the NSDAccess package
    nsda = NSDAccess('/home/naxos2-raid25/kneel027/home/surly/raid4/kendrick-data/nsd', '/home/naxos2-raid25/kneel027/home/kneel027/nsd_local')
    os.makedirs("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/reconstructions/" + experiment_title + "/", exist_ok=True)
    # Retriving the ground truth image. 
    subj1 = nsda.stim_descriptions[nsda.stim_descriptions['subject1'] != 0]
    
    # Load in the data
    # Generating predicted and target vectors
    # outputs_c, targets_c = Dc.predict(hashNum=Dc.hashNum, indices=idx)
    # outputs_c_i, targets_c_i = Dc_i.predict(model=c_img_modelId)
    # outputs_c_i = [outputs_c_i[i] for i in idx]
    _, _, _, _, x_test, _, _, _, _, targets_c_i, test_trials = load_nsd(vector="c_img_0", loader=False, average=True)
    _, _, _, _, _, _, _, _, _, targets_c_t, _ = load_nsd(vector="c_text_0", loader=False, average=True)
    _, _, _, _, _, _, _, _, _, targets_z, _ = load_nsd(vector="z_img_mixer", loader=False, average=True)
    AE = AutoEncoder(hashNum = "540",
                 lr=0.0000001,
                 vector="c_img_0", #c_img_0, c_text_0, z_img_mixer
                 encoderHash="536",
                 log=False, 
                 batch_size=750,
                 parallel=False,
                 device="cuda"
                )
    ae_x_test = AE.predict(x_test)

    # TODO: Run the 20 test x through the autoencoder to feed into predictVector_cc3m
    x_test = ae_x_test.to("cuda:0")
    # targets_c_i = targets_c_i
    # targets_c_t = targets_c_t
    # targets_z = targets_z
    
    
    # outputs_c_t = predictVector_cc3m(model="425_model_c_text_0.pt", vector="c_text_0", x=x_test)[:,0]
    # outputs_z = predictVector_cc3m(model="426_model_z_img_mixer.pt", vector="z_img_mixer", x=x_test)[:,0]
    # outputs_c_i = torch.load("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/latent_vectors/424_model_c_img_0.pt/c_img_0_cc3m_library_preds.pt")
    # outputs_c_t = torch.load("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/latent_vectors/425_model_c_text_0.pt/c_text_0_cc3m_library_preds.pt")
    # outputs_z = torch.load("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/latent_vectors/426_model_z_img_mixer.pt/z_img_mixer_cc3m_library_preds.pt")
    
    
    strength_c = 1
    strength_z = 0
    R = Reconstructor()
    for i in idx:
        # index = int(subj1x.loc[(subj1x['subject1_rep0'] == test_i) | (subj1x['subject1_rep1'] == test_i) | (subj1x['subject1_rep2'] == test_i)].nsdId)
        # rootdir = "/home/naxos2-raid25/kneel027/home/kneel027/nsd_local/cc3m/tensors/"
        # outputs_c_i[i] = torch.load(rootdir + "c_img_0/" + str(i) + ".pt")
        # outputs_c_t[i] = torch.load(rootdir + "c_text_0/" + str(i) + ".pt")
        # outputs_z[i] = torch.load(rootdir + "z_img_mixer/" + str(i) + ".pt")
        print(i)
        outputs_c_i = predictVector_cc3m(encModel="536_model_c_img_0.pt", vector="c_img_0", x=x_test[i].reshape((1,11838)))
        outputs_c_i = outputs_c_i.reshape((5, 768))
        c_combined = format_clip(outputs_c_i)
        c_combined_target = format_clip(targets_c_i[i])
        c_0 = format_clip(outputs_c_i[0])
        c_1 = format_clip(outputs_c_i[1])
        c_2 = format_clip(outputs_c_i[2])
        c_3 = format_clip(outputs_c_i[3])
        c_4 = format_clip(outputs_c_i[4])
        
    
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
        
        # returns a numpy array 
        nsdId = test_trials[i]
        ground_truth_np_array = nsda.read_images([nsdId], show=False)
        ground_truth = Image.fromarray(ground_truth_np_array[0])
        ground_truth = ground_truth.resize((512, 512), resample=Image.Resampling.LANCZOS)
        rows = 4
        columns = 2
        images = [ground_truth, reconstructed_target_c, reconstructed_output_c, reconstructed_output_c_0, reconstructed_output_c_1, reconstructed_output_c_2, reconstructed_output_c_3, reconstructed_output_c_4]
        captions = ["Ground Truth", "Target C", "Output 5 Cs", "C_0", "C_1","C_2","C_3","C_4"]
        figure = tileImages(experiment_title, images, captions, rows, columns)
        
        figure.save('reconstructions/' + experiment_title + '/' + str(i) + '.png')
        
    
def benchmark_library(encModel, vector, device="cuda:0"):
    _, _, _, _, x_test, _, _, _, _, target, test_trials = load_nsd(vector=vector, 
                                                        loader=False, average=False)
    if(not os.path.isfile("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/latent_vectors/" + encModel + "/library_preds_nsd_test.pt")):
        out = predictVector_cc3m(encModel=encModel, vector=vector, x=x_test, device=device)[:,0]
        torch.save(out, "/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/latent_vectors/" + encModel + "/library_preds_nsd_test.pt")
        
    else:
        out = torch.load("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/latent_vectors/" + encModel + "/library_preds_nsd_test.pt")
    
    criterion = nn.MSELoss()
    
    PeC = PearsonCorrCoef(num_outputs=x_test.shape[0]).to(device)
    target = target.to(device)
    out = out.to(device)

    loss = criterion(out, target)
    out = out.moveaxis(0,1)
    target = target.moveaxis(0,1)
    pearson_loss = torch.mean(PeC(out, target))
    
    out = out.detach()
    target = target.detach()
    PeC = PearsonCorrCoef()
    r = []
    for p in range(out.shape[1]):
        
        # Correlation across voxels for a sample (Taking a column)
        r.append(PeC(out[:,p], target[:,p]))
    r = np.array(r)
    
    print("Vector Correlation: ", float(pearson_loss))
    print("Mean Pearson: ", np.mean(r))
    print("Loss: ", float(loss))
    plt.hist(r, bins=40, log=True)
    plt.savefig("/export/raid1/home/kneel027/Second-Sight/charts/" + encModel + "_pearson_histogram_library_decoder.png")

if __name__ == "__main__":
    main()
