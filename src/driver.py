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
from vdvae import VDVAE

# First URL: This is the original read-only NSD file path (The actual data)
# Second URL: Local files that we are adding to the dataset and need to access as part of the data
# Object for the NSDAccess package
nsda = NSDAccess('/export/raid1/home/surly/raid4/kendrick-data/nsd', '/export/raid1/home/kneel027/nsd_local')

def main():
    train_decoder_uc(subject=2)
    train_decoder_uc(subject=5)
    train_decoder_uc(subject=7)
    # reconstructVDVAE(experiment_title="CLIP + VDVAE 747 764 3", idx=[i for i in range(20)], subject=1, decHash="747", vdvaeHash="764") 
    # reconstructVDVAE(experiment_title="CLIP + VDVAE 747 764 Test", idx=[0,1], subject=1, decHash="747", vdvaeHash="764") 
    # encHash = train_encoder_uc(subject=1)

    # train_decoder_uc(subject=5) 
    # encHash = train_encoder_uc(subject=5)
    # train_autoencoder(subject=1, encHash="771")
    # train_autoencoder(subject=2, encHash="772")
    # train_autoencoder(subject=5, encHash="774")
    # train_autoencoder(subject=7, encHash="775")
    # reconstructNImagesST(experiment_title="UC 747 ST", idx=[i for i in range(20)])
    
    # reconstructNImages(experiment_title="UC CLIP S1", idx=[i for i in range(0, 20)], subject=1, decHash="750")
    # reconstructNImages(experiment_title="UC CLIP S2", idx=[i for i in range(0, 20)], subject=2, decHash="753")
    # reconstructNImages(experiment_title="UC CLIP S5", idx=[i for i in range(0, 20)], subject=5, decHash="758")
    # reconstructNImages(experiment_title="UC CLIP S7", idx=[i for i in range(0, 20)], subject=7, decHash="761")
    # reconstructNImages(experiment_title="UC Test Bug", idx=[i for i in range(194, 214)])
    
    # train_autoencoder()


def train_autoencoder(subject, encHash):
    
    hashNum = update_hash()
    # hashNum = "582"
     
    AE = AutoEncoder(hashNum = hashNum,
                        lr=0.00001,
                        vector="images", #c_img_0, c_text_0, z_img_mixer, alexnet_encoder_sub1
                        subject=subject,
                        encoderHash=encHash,
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
    
def train_encoder_uc(subject=1):
    
    hashNum = update_hash()
    # hashNum = "738"
    E = Encoder_UC(hashNum = hashNum,
                 lr=0.00001,
                 vector="c_img_uc", #c_img_vd, c_text_vd
                 subject=subject,
                 log=True, 
                 batch_size=750,
                 device="cuda:0",
                 num_workers=16,
                 epochs=300
                )
    # E.train()
    
    
    # E.benchmark(average=False)
    # E.benchmark(average=True)

    process_x_encoded(Encoder=E)
    
    return hashNum


def train_decoder_uc(subject=1):
    hashNum = update_hash()
    # hashNum = "765"
    D = Decoder_UC(hashNum = hashNum,
                 lr=0.00001,
                 vector="z_vdvae", #c_img_uc , c_text_uc, z_vdvae
                 subject=subject,
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


def reconstructVDVAE(experiment_title, idx, subject=1, decHash="747", vdvaeHash="764"):
    os.makedirs("reconstructions/{}/".format(experiment_title), exist_ok=True)
    
    _, _, x_test, y_train, y_val, targets_vdvae, trials = load_nsd(vector="z_vdvae", subject=subject, loader=False, average=True)
    _, _, _, _, _, targets_c_i, _ = load_nsd(vector="c_img_uc", subject=subject, loader=False, average=True)
    Dc_i = Decoder_UC(hashNum = decHash,
                 vector="c_img_uc",
                 subject=subject, 
                 log=False, 
                 device="cuda",
                 )
    outputs_c_i = Dc_i.predict(x=x_test[idx]).reshape((len(idx), 1, 1024))
    del Dc_i
    Dv = Decoder_UC(hashNum = vdvaeHash,
                 vector="z_vdvae",
                 subject=subject, 
                 log=False, 
                 device="cuda",
                 )
    outputs_vdvae = Dv.predict(x=x_test[idx])
    del Dv
    
    latent_mean = torch.load("vdvae/train_mean.pt").to("cuda")
    latent_std = torch.load("vdvae/train_std.pt").to("cuda")
    # train_z = torch.concat([y_train, y_val], dim=0).to("cuda")
    # latent_mean = torch.mean(train_z, dim=0)
    # latent_std = torch.std(train_z, dim=0)
    
    outputs_vdvae = (outputs_vdvae - torch.mean(outputs_vdvae, dim=0)) / torch.std(outputs_vdvae, dim=0)
    outputs_vdvae = (outputs_vdvae * latent_std) + latent_mean
    outputs_vdvae = outputs_vdvae.reshape((len(idx), 1, 91168))
    
    targets_vdvae = targets_vdvae.to("cuda")
    std_norm_test_latent = (targets_vdvae - torch.mean(targets_vdvae,dim=0)) / torch.std(targets_vdvae,dim=0)
    targets_vdvae = std_norm_test_latent * latent_std + latent_mean
    targets_vdvae = targets_vdvae[idx].reshape((len(idx), 1, 91168))
    
    targets_c_i = targets_c_i[idx].reshape((len(idx), 1, 1024))
    
    V = VDVAE()
    R = StableUnCLIPImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16, variation="fp16").to("cuda")
    for i, val in enumerate(tqdm(idx, desc="Generating reconstructions")):
        output_v = V.reconstruct(outputs_vdvae[i])
        target_v = V.reconstruct(targets_vdvae[i])
        
        # Make the c_img reconstrution images
        output_c = R.reconstruct(image_embeds=outputs_c_i[i], strength=1)
        target_c = R.reconstruct(image_embeds=targets_c_i[i], strength=1)
        
        # # Make the prompt guided reconstrution images
        output_cv = R.reconstruct(image=output_v, image_embeds=outputs_c_i[i], negative_prompt="text, caption", strength=0.8, guidance_scale=10)
        target_cv = R.reconstruct(image=target_v, image_embeds=targets_c_i[i], negative_prompt="text, caption", strength=0.8, guidance_scale=10)
        
        nsdId = trials[val]

        ground_truth_np_array = nsda.read_images([nsdId], show=True)
        ground_truth = Image.fromarray(ground_truth_np_array[0])
        ground_truth = ground_truth.resize((768, 768), resample=Image.Resampling.LANCZOS)
        empty = Image.new('RGB', (768, 768), color='white')
        rows = 4
        columns = 2
        images = [ground_truth, empty, target_v, output_v, target_c, output_c, target_cv, output_cv]
        captions = ["Ground Truth", "", "Ground Truth VDVAE", "Decoded VDVAE", "Ground Truth CLIP", "Decoded CLIP", "Ground Truth CLIP + VDVAE", "Decoded CLIP + VDVAE"]
        figure = tileImages(experiment_title + ": " + str(val), images, captions, rows, columns)
        
        figure.save('/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/reconstructions/{}/{}.png'.format(experiment_title, val))
    


def reconstructNImages(experiment_title, idx, subject=1, decHash="747"):
    
    _, _, x_test, _, _, targets_c_i, trials = load_nsd(vector="c_img_uc", subject=subject, loader=False, average=True)
    Dc_i = Decoder_UC(hashNum = decHash,
                 vector="c_img_uc",
                 subject=subject, 

                 log=False, 
                 device="cuda",
                 )
    outputs_c_i = Dc_i.predict(x=x_test[idx]).reshape((len(idx), 1, 1024))
    del Dc_i
    
    os.makedirs("reconstructions/{}/".format(experiment_title), exist_ok=True)
    
    
    targets_c_i = targets_c_i[idx].reshape((len(idx), 1, 1024))
    
    R = StableUnCLIPImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16, variation="fp16").to("cuda")
    
    for i, val in enumerate(tqdm(idx, desc="Generating reconstructions")):
        
        # Make the c_img reconstrution images
        reconstructed_output_c_i = R.reconstruct(image_embeds=outputs_c_i[i], strength=1)
        reconstructed_target_c_i = R.reconstruct(image_embeds=targets_c_i[i], strength=1)
        
        # # Make the prompt guided reconstrution images
        reconstructed_output_c = R.reconstruct(image_embeds=outputs_c_i[i], negative_prompt="text, caption", strength=1, guidance_scale=10)
        reconstructed_target_c = R.reconstruct(image_embeds=targets_c_i[i], negative_prompt="text, caption", strength=1, guidance_scale=10)
        
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
