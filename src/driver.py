import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
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
from decoder import Decoder
from decoder_pca import Decoder_PCA
from decoder_uc import Decoder_UC
from encoder_uc import Encoder_UC
from encoder import Encoder
from diffusers import StableUnCLIPImg2ImgPipeline
from autoencoder  import AutoEncoder
from alexnet_encoder import AlexNetEncoder
from ss_decoder import SS_Decoder
from mask import Masker


def main():
    # train_decoder_uc() 

    # train_encoder_uc()

    # reconstructNImagesST(experiment_title="VD mixed decoders", idx=[i for i in range(21)])
    
    reconstructNImages(experiment_title="UC 747 prompt \"photorealistic\" neg \"cartoon, art, saturated, text, caption\" gc10", idx=[i for i in range(21)])

    # train_autoencoder()


def train_autoencoder():
    
    # hashNum = update_hash()
    hashNum = "742"
    
    # x, y = load_nsd(vector="c_text_vd", encoderModel="660_model_c_text_vd.pt", ae=True, loader=False, split=False)
    # PeC = PearsonCorrCoef(num_outputs=x.shape[0])
    # print(torch.mean(PeC(x.moveaxis(0,1), y.moveaxis(0,1))))
     
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

# Encode latent z (1x4x64x64) and condition c (1x77x1024) tensors into an image
# Strength parameter controls the weighting between the two tensors
def reconstructNImages(experiment_title, idx):
    
    _, _, x_param, x_test, _, _, targets_c_i, _, param_trials, test_trials = load_nsd(vector="c_img_uc", loader=False, average=True)
    _, _, _, _, _, _, targets_c_t, _, _, _ = load_nsd(vector="c_text_uc", loader=False, average=True)
    Dc_i = Decoder_UC(hashNum = "747",
                 vector="c_img_uc", 
                 log=False, 
                 device="cuda",
                 )
    outputs_c_i = Dc_i.predict(x=x_param[idx]).reshape((len(idx), 1, 1024))
    del Dc_i
    # Dc_t = Decoder_UC(hashNum = "741",
    #              vector="c_text_uc",
    #              log=False, 
    #              device="cuda",
    #              )
    # outputs_c_t = Dc_t.predict(x=x_param[idx]).reshape((len(idx), 1, 77, 1024))
    # del Dc_t
    
    # First URL: This is the original read-only NSD file path (The actual data)
    # Second URL: Local files that we are adding to the dataset and need to access as part of the data
    # Object for the NSDAccess package
    nsda = NSDAccess('/export/raid1/home/surly/raid4/kendrick-data/nsd', '/export/raid1/home/kneel027/nsd_local')
    os.makedirs("reconstructions/" + experiment_title + "/", exist_ok=True)
    
    
    targets_c_i = targets_c_i[idx].reshape((len(idx), 1, 1024))
    targets_c_t = targets_c_t[idx].reshape((len(idx), 1, 77, 1024))
    
    R = StableUnCLIPImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16, variation="fp16")
    R = R.to("cuda")
    R.enable_xformers_memory_efficient_attention()
    
    for i in tqdm(idx, desc="Generating reconstructions"):
        
        # Make the c reconstrution images. 
        # print("SHAPE: ", targets_c_i[i].device, targets_c_i[i].dtype)
        reconstructed_output_c_i = R.reconstruct(image_embeds=outputs_c_i[i], strength=1)
        reconstructed_target_c_i = R.reconstruct(image_embeds=targets_c_i[i], strength=1)
        
        # # Make the z and c reconstrution images. 
        reconstructed_output_c = R.reconstruct(image_embeds=outputs_c_i[i], prompt="photorealistic", negative_prompt="cartoon, art, saturated, text, caption", strength=1, guidance_scale=10)
        reconstructed_target_c = R.reconstruct(image_embeds=targets_c_i[i], prompt="photorealistic", negative_prompt="cartoon, art, saturated, text, caption", strength=1, guidance_scale=10)
        
        # # Make the z reconstrution images. 
        # reconstructed_output_c_t = R.reconstruct(image_embeds=outputs_c_i[i], prompt_embeds=outputs_c_t[i], strength=1, noise_level=999)
        # reconstructed_target_c_t = R.reconstruct(image_embeds=targets_c_i[i], prompt_embeds=targets_c_t[i], strength=1, noise_level=999)
        
        # # # Make the z and c reconstrution images. 
        # reconstructed_output_c = R.reconstruct(image_embeds=outputs_c_i[i], prompt_embeds=outputs_c_t[i], strength=1)
        # reconstructed_target_c = R.reconstruct(image_embeds=targets_c_i[i], prompt_embeds=targets_c_t[i], strength=1)
        
        # returns a numpy array 
        nsdId = param_trials[i]
        ground_truth_np_array = nsda.read_images([nsdId], show=True)
        ground_truth = Image.fromarray(ground_truth_np_array[0])
        ground_truth = ground_truth.resize((768, 768), resample=Image.Resampling.LANCZOS)
        empty = Image.new('RGB', (768, 768), color='white')
        rows = 3
        columns = 2
        images = [ground_truth, empty, reconstructed_target_c, reconstructed_output_c, reconstructed_target_c_i, reconstructed_output_c_i]
        captions = ["Ground Truth", "", "Target C_2", "Output C_2", "Target C_i", "Output C_i"]
        figure = tileImages(experiment_title + ": " + str(i), images, captions, rows, columns)
        
        figure.save('/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/reconstructions/' + experiment_title + '/' + str(i) + '.png')

def reconstructNImagesST(experiment_title, idx):
    
    Dc_i = Decoder_UC(hashNum = "731",
                 vector="c_img_uc", 
                 log=False, 
                 device="cuda"
                 )
    Dc_t = Decoder_UC(hashNum = "733",
                 vector="c_text_uc",
                 log=False, 
                 device="cuda"
                 )
    
    # First URL: This is the original read-only NSD file path (The actual data)
    # Second URL: Local files that we are adding to the dataset and need to access as part of the data
    # Object for the NSDAccess package
    nsda = NSDAccess('/export/raid1/home/surly/raid4/kendrick-data/nsd', '/export/raid1/home/kneel027/nsd_local')
    os.makedirs("reconstructions/" + experiment_title + "/", exist_ok=True)
    # Load test data and targets
    _, _, x_param, x_test, _, _, targets_c_i, _, param_trials, test_trials = load_nsd(vector="c_img_vd", loader=False, average=False, nest=True)
    _, _, _, _, _, _, targets_c_t, _, _, _ = load_nsd(vector="c_text_vd", loader=False, average=True)
    # _, _, _, _, _, _, targets_z, _, _, _ = load_nsd(vector="z_img_mixer", loader=False, average=True)
    print(x_param[1, 1].shape)
    # Generating predicted and target vectors
    # ae_x_test = AE.predict(x_test)
    # outputs_c_i = SS_Dc_i.predict(x=ae_x_test)
    outputs_c_i = Dc_i.predict(x=torch.mean(x_param, dim=1))
    outputs_c_t = Dc_t.predict(x=torch.mean(x_param, dim=1))
    print(outputs_c_i.shape)
    print(outputs_c_t.shape)
    
    # outputs_z = Dz.predict(x=x_param)
    strength_c = 1
    strength_z = 0
    R = StableUnCLIPImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16, variation="fp16")
    R = R.to("cuda")
    R.enable_xformers_memory_efficient_attention()
    for i in tqdm(idx, desc="Generating reconstructions"):

        TCc = R.reconstruct(c_i=targets_c_i[i], c_t=targets_c_t[i], textstrength=0.4, strength=strength_c)
        TCi = R.reconstruct(c_i=targets_c_i[i], c_t=targets_c_t[i], textstrength=0.0, strength=strength_c)
        TCt = R.reconstruct(c_i=targets_c_i[i], c_t=targets_c_t[i], textstrength=1.0, strength=strength_c)
        OCc = R.reconstruct(c_i=outputs_c_i[i], c_t=outputs_c_t[i], textstrength=0.4, strength=strength_c)
        OCi = R.reconstruct(c_i=outputs_c_i[i], c_t=outputs_c_t[i], textstrength=0.0, strength=strength_c)
        OCt = R.reconstruct(c_i=outputs_c_i[i], c_t=outputs_c_t[i], textstrength=1.0, strength=strength_c)
        OCcS, OCiS, OCtS = [], [], []
        for j in range(len(x_param[i])):
            outputs_c_i_j = Dc_i.predict(x=x_param[i, j])
            outputs_c_t_j = Dc_t.predict(x=x_param[i, j])
            OCcS.append(R.reconstruct(c_i=outputs_c_i_j, c_t=outputs_c_t_j, textstrength=0.4, strength=strength_c))
            OCiS.append(R.reconstruct(c_i=outputs_c_i_j, c_t=outputs_c_t_j, textstrength=0.0, strength=strength_c))
            OCtS.append(R.reconstruct(c_i=outputs_c_i_j, c_t=outputs_c_t_j, textstrength=1.0, strength=strength_c))
            
        
        # returns a numpy array 
        nsdId = param_trials[i]
        ground_truth_np_array = nsda.read_images([nsdId], show=True)
        ground_truth = Image.fromarray(ground_truth_np_array[0])
        ground_truth = ground_truth.resize((768, 768), resample=Image.Resampling.LANCZOS)
        empty = Image.new('RGB', (768, 768), color='white')
        rows = 6
        columns = 3
        images = [ground_truth, equalize_color(ground_truth), equalize_color(OCc), TCc, TCi, TCt, OCc, OCi, OCt]
        captions = ["Ground Truth", "Ground Truth EQ", "Output C_c EQ", "Target C_c", "Target C_i", "Target C_t", "Output C_c", "Output C_i", "Output C_t"]
        numTrials = len(OCcS)
        for k in range(numTrials):
            images.append(OCcS[k])
            captions.append("Output C_c Trial " + str(k))
            images.append(OCiS[k])
            captions.append("Output C_i Trial " + str(k))
            images.append(OCtS[k])
            captions.append("Output C_t Trial " + str(k))
        for p in range(3-numTrials):
            images.append(empty)
            captions.append("")
        figure = tileImages(experiment_title + ": " + str(i), images, captions, rows, columns)
        
        figure.save('/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/reconstructions/' + experiment_title + '/' + str(i) + '.png')
if __name__ == "__main__":
    main()
