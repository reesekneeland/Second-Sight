import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
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
import random
import copy
from tqdm import tqdm
from diffusers import StableUnCLIPImg2ImgPipeline
from library_decoder import LibraryDecoder
from pearson import PearsonCorrCoef


def main():
    # benchmark_library("c_img_uc", average=True, config=["c_img_uc"])
    # reconstructNImages(experiment_title="coco top 500 UC 738",
    #                    idx=[i for i in range(0, 20)],
    #                    mask=[],
    #                    test=False,
    #                    average=True,
    #                    config=["c_img_uc"])
    # reconstruct_test_samples("SCS UC 747 10:100:4 0.4 Exp3 AE", idx=[], test=False, average=True)
    # reconstruct_test_samples("SCS UC 747 10:100:4 0.5 Exp3 AE", idx=[], test=False, average=True)
    reconstruct_test_samples("SCS UC 10:250:5 0.6 Exp3 AE Fixed", idx=[], test=True, average=True)


            
# Encode latent z (1x4x64x64) and condition c (1x77x1024) tensors into an image
# Strength parameter controls the weighting between the two tensors
def reconstructNImages(experiment_title, idx, mask=[], test=True, average=True, config=["AlexNet"]):
    
    # First URL: This is the original read-only NSD file path (The actual data)
    # Second URL: Local files that we are adding to the dataset and need to access as part of the data
    # Object for the NSDAccess package
    nsda = NSDAccess('/export/raid1/home/surly/raid4/kendrick-data/nsd', '/export/raid1/home/kneel027/nsd_local')
    os.makedirs("reconstructions/" + experiment_title + "/", exist_ok=True)
    # Retriving the ground truth image. 
    subj1 = nsda.stim_descriptions[nsda.stim_descriptions['subject1'] != 0]
    LD = LibraryDecoder(vector="images",
                        config=config,
                        device="cuda:0")

    # Load data and targets
    if test:
        _, _, _, x, _, _, _, targets_c_i, _, trials = load_nsd(vector="c_img_uc", loader=False, average=False, nest=True)
        _, _, _, _, _, _, _, targets_c_t, _, _ = load_nsd(vector="c_text_uc", loader=False, average=False, nest=True)
    else:
        _, _, x, _, _, _, targets_c_i, _, trials, _ = load_nsd(vector="c_img_uc", loader=False, average=False, nest=True)
        _, _, _, _, _, _, targets_c_t, _, _, _ = load_nsd(vector="c_text_uc", loader=False, average=False, nest=True)
    x = x[idx]
    
    
    output_images, _ = LD.predictVector_coco(x, average=average)
    LD = LibraryDecoder(vector="c_img_uc",
                        config=config,
                        device="cuda:0")
    output_clips, _ = LD.predictVector_coco(x, average=average)
    output_clips = output_clips.reshape((len(idx), 500, 1, 1024))
    targets_c_i = targets_c_i[idx].reshape((len(idx), 1, 1024))
    R = StableUnCLIPImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16, variation="fp16")
    R = R.to("cuda")
    R.enable_xformers_memory_efficient_attention()
    for i, val in enumerate(tqdm(idx, desc="Generating reconstructions")):
        
        print(i, val)
        c_i_mean = torch.mean(output_images[i,0:5].to(torch.float32), dim=0)
        i_combined = process_image(c_i_mean.to(torch.uint8))
        i_0 = process_image(output_images[i,0])
        i_1 = process_image(output_images[i,1])
        i_2 = process_image(output_images[i,2])
        i_3 = process_image(output_images[i,3])
        i_4 = process_image(output_images[i,4])
        
        
        ci_0 = output_clips[i,0]
        ci_1 = torch.mean(output_clips[i,0:100], dim=0)
        ci_2 = torch.mean(output_clips[i,0:25], dim=0)
        ci_3 = torch.mean(output_clips[i,0:10], dim=0)
        ci_4 = output_clips[i,0]
        c_i_combined = torch.mean(output_clips[i,0:500], dim=0)
        
        # c_i_combined = torch.mean(outputs_c_i[i], dim=0)
        # ci_0 = outputs_c_i[i,0]
        # ci_1 = outputs_c_i[i,1]
        # ci_2 = outputs_c_i[i,2]
        # ci_3 = outputs_c_i[i,3]
        # ci_4 = outputs_c_i[i,4]
        
        # outputs_c_t = outputs_c_t
        # c_t_combined = torch.mean(outputs_c_t[i], dim=0)
        # ct_0 = outputs_c_t[i,0]
        # ct_1 = outputs_c_t[i,1]
        # ct_2 = outputs_c_t[i,2]
        # ct_3 = outputs_c_t[i,3]
        # ct_4 = outputs_c_t[i,4]
    
        # Make the c reconstrution images. 
        # reconstructed_output_c = R.reconstruct(c=c_combined, strength=strength_c)
        # reconstructed_target_c = R.reconstruct(c=c_combined_target, strength=strength_c)
        target_c_i = R.reconstruct(image_embeds=targets_c_i[i])
        output_ci = R.reconstruct(image_embeds=c_i_combined)
        output_ci_0 = R.reconstruct(image_embeds=ci_0)
        output_ci_1 = R.reconstruct(image_embeds=ci_1)
        output_ci_2 = R.reconstruct(image_embeds=ci_2)
        output_ci_3 = R.reconstruct(image_embeds=ci_3)
        output_ci_4 = R.reconstruct(image_embeds=ci_4)
        
        # target_c_t = R.reconstruct(c_t=targets_c_t[val])
        # output_ct = R.reconstruct(c_t=c_t_combined)
        # output_ct_0 = R.reconstruct(c_t=ct_0)
        # output_ct_1 = R.reconstruct(c_t=ct_1)
        # output_ct_2 = R.reconstruct(c_t=ct_2)
        # output_ct_3 = R.reconstruct(c_t=ct_3)
        # output_ct_4 = R.reconstruct(c_t=ct_4)
        
        # target_c_c = R.reconstruct(c_i=targets_c_i[val], c_t=targets_c_t[val])
        # output_cc = R.reconstruct(c_i=c_i_combined, c_t=c_t_combined)
        # output_cc_0 = R.reconstruct(c_i=ci_0, c_t=ct_0)
        # output_cc_1 = R.reconstruct(c_i=ci_1, c_t=ct_1)
        # output_cc_2 = R.reconstruct(c_i=ci_2, c_t=ct_2)
        # output_cc_3 = R.reconstruct(c_i=ci_3, c_t=ct_3)
        # output_cc_4 = R.reconstruct(c_i=ci_4, c_t=ct_4)
       
        
        # returns a numpy array 
        nsdId = trials[val]
        ground_truth_np_array = nsda.read_images([nsdId], show=False)
        ground_truth = Image.fromarray(ground_truth_np_array[0])
        ground_truth = ground_truth.resize((768, 768), resample=Image.Resampling.LANCZOS)
        empty = Image.new('RGB', (768, 768), color='white')
        rows = 7
        columns = 2
        images = [ground_truth, target_c_i, 
                  i_combined,   output_ci,
                  i_0,          output_ci_0,
                  i_1,          output_ci_1,
                  i_2,          output_ci_2,
                  i_3,          output_ci_3,
                  i_4,          output_ci_4,]
        captions = ["ground_truth","target_c_i", 
                  "output_i",     "top 500", 
                  "output_i_0",   "top 100",
                  "output_i_1",   "top 25",
                  "output_i_2",   "top 10", 
                  "output_i_3",   "top 5", 
                  "output_i_4",   "top 1",]
        # images = [ground_truth, target_c_c, target_c_i, target_c_t,
        #           i_combined,   output_cc,  output_ci, output_ct,
        #           i_0,          output_cc_0, output_ci_0, output_ct_0,
        #           i_1,          output_cc_1, output_ci_1, output_ct_1,
        #           i_2,          output_cc_2, output_ci_2, output_ct_2,
        #           i_3,          output_cc_3, output_ci_3, output_ct_3,
        #           i_4,          output_cc_4, output_ci_4, output_ct_4]
        # captions = ["ground_truth","target_c_c", "target_c_i", "target_c_t",
        #           "output_i", "output_cc", "output_ci", "output_ct",
        #           "output_i_0", "output_cc_0", "output_ci_0", "output_ct_0",
        #           "output_i_1", "output_cc_1", "output_ci_1", "output_ct_1",
        #           "output_i_2", "output_cc_2", "output_ci_2", "output_ct_2",
        #           "output_i_3", "output_cc_3", "output_ci_3", "output_ct_3",
        #           "output_i_4", "output_cc_4", "output_ci_4", "output_ct_4"]
        figure = tileImages(experiment_title + ": " + str(val), images, captions, rows, columns)
        
        figure.save('reconstructions/' + experiment_title + '/' + str(val) + '.png')
        
    
def benchmark_library(vector, average=True, config=["AlexNet"]):
    device = "cuda"
    LD = LibraryDecoder(vector=vector,
                        config=config,
                        device=device)
    LD.benchmark(average=average)



def reconstruct_test_samples(experiment_title, idx=[], test=False, average=True):
    if(len(idx) == 0):
        for file in os.listdir("reconstructions/" + experiment_title + "/"):
            if file.endswith(".png") and file not in ["Search Iterations.png", "Results.png"]:
                idx.append(int(file[:-4]))
        idx = sorted(idx)

    if test:
        _, _, _, x, _, _, _, targets_c_i, _, trials = load_nsd(vector="c_img_uc", loader=False, average=False, nest=True)
    else:
        _, _, x, _, _, _, targets_c_i, _, trials, _ = load_nsd(vector="c_img_uc", loader=False, average=False, nest=True)
    
    x = x[idx]
    
    device = "cuda"
    LD = LibraryDecoder(vector="images",
                        config=["AlexNet"],
                        device=device)
    output_images, _ = LD.predictVector_coco(x, average=average)
    for i, image in enumerate(output_images):
        top_choice = image[0].reshape(425, 425, 3)
        top_choice = top_choice.detach().cpu().numpy().astype(np.uint8)
        pil_image = Image.fromarray(top_choice).resize((768, 768), resample=Image.Resampling.LANCZOS)
        pil_image.save("reconstructions/" + experiment_title + "/" + str(idx[i]) + "/Library Reconstruction.png")

if __name__ == "__main__":
    main()