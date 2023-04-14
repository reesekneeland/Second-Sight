import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
import torch
import numpy as np
from PIL import Image
from nsd_access import NSDAccess
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch.nn as nn
from pycocotools.coco import COCO
from utils import *
import copy
from tqdm import tqdm
from decoder_uc import Decoder_UC
from encoder_uc import Encoder_UC
from diffusers import StableUnCLIPImg2ImgPipeline
from autoencoder  import AutoEncoder
from alexnet_encoder import AlexNetEncoder


def main():
    # train_decoder_uc() 

    # train_encoder_uc()

    # reconstructNImagesST(experiment_title="UC 747 ST", idx=[i for i in range(20)])
    

    reconstructNImages(experiment_title="UC Test Bug", idx=[i for i in range(194, 214)])


    # train_autoencoder()


def train_autoencoder():
    
    # hashNum = update_hash()
    hashNum = "742"
     
    AE = AutoEncoder(hashNum = hashNum,
                        lr=0.00001,
                        vector="c_text_uc", #c_img_0, c_text_0, z_img_mixer, alexnet_encoder_sub1
                        encoderHash="739",
                        log=True, 
                        device="cuda:0",
                        num_workers=16,
                        epochs=300
                        )
    
    AE.train()
    AE.benchmark(encodedPass=False, average=False)
    AE.benchmark(encodedPass=False, average=True)
    AE.benchmark(encodedPass=True, average=False)
    AE.benchmark(encodedPass=True, average=True)
    
def train_encoder_uc():
    
    # hashNum = update_hash()
    hashNum = "738"
    E = Encoder_UC(hashNum = hashNum,
                 lr=0.00001,
                 vector="c_img_uc", #c_img_vd, c_text_vd
                 log=False, 
                 batch_size=750,
                 device="cuda:0",
                 num_workers=16,
                 epochs=300
                )
    # E.train()
    
    
    # E.benchmark(average=False)
    # E.benchmark(average=True)
    
    # modelId = E.hashNum + "_model_" + E.vector + ".pt"
    # process_x_encoded(Encoder=E, modelId=modelId, subject=1)
    
    return hashNum


def train_decoder_uc():
    hashNum = update_hash()
    # hashNum = "746"
    D = Decoder_UC(hashNum = hashNum,
                 lr=0.000001,
                 vector="c_img_uc", #c_img_0 , c_text_0, z_img_mixer
                 log=True, 
                 batch_size=64,
                 device="cuda:0",
                 num_workers=4,
                 epochs=500
                )
    
    D.train()
    
    D.benchmark(average=False)
    D.benchmark(average=True)
    
    return hashNum


def reconstructNImages(experiment_title, idx):
    

    _, _, x_test, _, _, targets_c_i, trials = load_nsd(vector="c_img_uc", loader=False, average=True)

    Dc_i = Decoder_UC(hashNum = "747",
                 vector="c_img_uc", 
                 log=False, 
                 device="cuda",
                 )
    outputs_c_i = Dc_i.predict(x=x_test[idx]).reshape((len(idx), 1, 1024))
    del Dc_i
    
    # First URL: This is the original read-only NSD file path (The actual data)
    # Second URL: Local files that we are adding to the dataset and need to access as part of the data
    # Object for the NSDAccess package
    nsda = NSDAccess('/export/raid1/home/surly/raid4/kendrick-data/nsd', '/export/raid1/home/kneel027/nsd_local')
    os.makedirs("reconstructions/{}/".format(experiment_title), exist_ok=True)
    
    
    targets_c_i = targets_c_i[idx].reshape((len(idx), 1, 1024))
    
    R = StableUnCLIPImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16, variation="fp16")
    R = R.to("cuda")
    # R.enable_xformers_memory_efficient_attention()
    
    for i, val in enumerate(tqdm(idx, desc="Generating reconstructions")):
        
        # Make the c_img reconstrution images
        reconstructed_output_c_i = R.reconstruct(image_embeds=outputs_c_i[i], strength=1)
        reconstructed_target_c_i = R.reconstruct(image_embeds=targets_c_i[i], strength=1)
        
        # # Make the prompt guided reconstrution images
        reconstructed_output_c = R.reconstruct(image_embeds=outputs_c_i[i], prompt="photorealistic", negative_prompt="cartoon, art, saturated, text, caption", strength=1, guidance_scale=10)
        reconstructed_target_c = R.reconstruct(image_embeds=targets_c_i[i], prompt="photorealistic", negative_prompt="cartoon, art, saturated, text, caption", strength=1, guidance_scale=10)
        
        nsdId = trials[val]

        ground_truth_np_array = nsda.read_images([nsdId], show=True)
        ground_truth = Image.fromarray(ground_truth_np_array[0])
        ground_truth = ground_truth.resize((768, 768), resample=Image.Resampling.LANCZOS)
        empty = Image.new('RGB', (768, 768), color='white')
        rows = 3
        columns = 2
        images = [ground_truth, empty, reconstructed_target_c, reconstructed_output_c, reconstructed_target_c_i, reconstructed_output_c_i]
        captions = ["Ground Truth", "", "Target C_2", "Output C_2", "Target C_i", "Output C_i"]
        figure = tileImages(experiment_title + ": " + str(val), images, captions, rows, columns)
        
        figure.save('/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/reconstructions/{}/{}.png'.format(experiment_title, val))


def reconstructNImagesST(experiment_title, idx):
    
    Dc_i = Decoder_UC(hashNum = "747",
                 vector="c_img_uc", 
                 log=False, 
                 device="cuda"
                 )
    
    # First URL: This is the original read-only NSD file path (The actual data)
    # Second URL: Local files that we are adding to the dataset and need to access as part of the data
    # Object for the NSDAccess package
    nsda = NSDAccess('/export/raid1/home/surly/raid4/kendrick-data/nsd', '/export/raid1/home/kneel027/nsd_local')
    os.makedirs("reconstructions/{}/".format(experiment_title), exist_ok=True)
    # Load test data and targets
    _, _, x_test, _, _, targets_c_i, trials = load_nsd(vector="c_img_vd", loader=False, average=False, nest=True)
    outputs_c_i = Dc_i.predict(x=torch.mean(x_test, dim=1))
    
    
    R = StableUnCLIPImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16, variation="fp16")
    R = R.to("cuda")
    R.enable_xformers_memory_efficient_attention()
    for i in tqdm(idx, desc="Generating reconstructions"):

        TCc = R.reconstruct(image_embeds=targets_c_i[i], prompt="photorealistic", negative_prompt="cartoon, art, saturated, text, caption")
        TCi = R.reconstruct(image_embeds=targets_c_i[i])
        OCc = R.reconstruct(image_embeds=outputs_c_i[i], prompt="photorealistic", negative_prompt="cartoon, art, saturated, text, caption")
        OCi = R.reconstruct(image_embeds=outputs_c_i[i])
        OCcS, OCiS = [], []
        for j in range(len(x_test[i])):
            outputs_c_i_j = Dc_i.predict(x=x_test[i, j])
            OCcS.append(R.reconstruct(image_embeds=outputs_c_i_j[i], prompt="photorealistic", negative_prompt="cartoon, art, saturated, text, caption"))
            OCiS.append(R.reconstruct(image_embeds=outputs_c_i_j[i]))
            
        nsdId = trials[i]
        ground_truth_np_array = nsda.read_images([nsdId], show=True)
        ground_truth = Image.fromarray(ground_truth_np_array[0])
        ground_truth = ground_truth.resize((768, 768), resample=Image.Resampling.LANCZOS)
        empty = Image.new('RGB', (768, 768), color='white')
        rows = 3 + numTrials
        numTrials = len(OCcS)
        columns = 2
        images = [ground_truth, empty, TCc, TCi, OCc, OCi]
        captions = ["Ground Truth", "", "Target C_c", "Target C_i", "Output C_c", "Output C_i"]
        for k in range(numTrials):
            images.append(OCcS[k])
            captions.append("Output C_c Trial " + str(k))
            images.append(OCiS[k])
            captions.append("Output C_i Trial " + str(k))
        figure = tileImages(experiment_title + ": " + str(i), images, captions, rows, columns)
        
        figure.save('/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/reconstructions/{}/{}.png'.format(experiment_title, i))
if __name__ == "__main__":
    main()
