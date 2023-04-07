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
    # _, _, _, _, _, _, _, _, _, _, _ = load_nsd(vector="c_img_0", loader=False, average=True)
    
    # train_decoder()
    
    # train_decoder_uc()
    
    # train_decoder_pca()

    # train_encoder_uc()

    # train_encoder()
    
    # mask_voxels()

    # load_cc3m("c_img_0", "410_model_c_img_0.pt")

    # reconstructNImagesST(experiment_title="VD mixed decoders", idx=[i for i in range(21)])
    
    reconstructNImages(experiment_title="UC test", idx=[i for i in range(7, 21)])

    # test_reconstruct()

    # train_autoencoder()

    # train_ss_decoder()

def mask_voxels():
    M = Masker(encoderHash="658",
                 vector="c_img_vd",
                 device="cuda:0")

    thresholds = list(torch.arange(0.10, 1.0, 0.01))
    for t in tqdm(thresholds, desc="thresholds"): 
        x = float(t)
        M.get_percentile_coco(x)
    M.make_histogram()
    # M.create_mask(threshold=-1)
    
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
    
    # AN = AlexNetEncoder()
    
    # modelId = AE.hashNum + "_model_" + AE.vector + ".pt"
    
    # os.makedirs("/export/raid1/home/kneel027/nsd_local/preprocessed_data/x_encoded/" + modelId, exist_ok=True)
    # # _, _, _, _, _, _, _, _, _, images = load_nsd(vector = "c_img_0", batch_size = AE.batch_size,
    # #                                              num_workers = AE.num_workers, loader = False, 
    # #                                              split = True, return_images=True)
    # images = process_data(image=True)
    # print("This is the length of PIL images: ", len(images))
    # print("This is the type: ", type(images[0]))
    # outputs = AN.predict(images)
    # print("This is the output shape: ", outputs.shape)
    # torch.save(outputs, "/export/raid1/home/kneel027/nsd_local/preprocessed_data/x_encoded/" + modelId + "/vector.pt")
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
    
    modelId = E.hashNum + "_model_" + E.vector + ".pt"
    os.makedirs("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/latent_vectors/" + modelId, exist_ok=True)
    coco_full = torch.load("/export/raid1/home/kneel027/nsd_local/preprocessed_data/" + E.vector + "/vector_73k.pt")
    coco_preds_full = torch.zeros((73000, 11838))
    for i in range(4):
        coco_preds_full[18250*i:18250*i + 18250] = E.predict(coco_full[18250*i:18250*i + 18250]).cpu()
    pruned_encodings = prune_vector(coco_preds_full)
    torch.save(pruned_encodings, "/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/latent_vectors/" + modelId + "/coco_brain_preds.pt")

    os.makedirs("/export/raid1/home/kneel027/nsd_local/preprocessed_data/x_encoded/" + modelId, exist_ok=True)
    x, y = load_nsd(vector = E.vector, loader = False, split = False)
    outputs = torch.zeros_like(x)
    print(outputs.shape)
    for i in range(2):
        outputs[13875*i:13875*i + 13875] = E.predict(y[13875*i:13875*i + 13875])
        
    torch.save(outputs, "/export/raid1/home/kneel027/nsd_local/preprocessed_data/x_encoded/" + modelId + "/vector.pt")
    
    return hashNum

def train_encoder():
    # hashNum = update_hash()
    
    hashNum = "658"
    E = Encoder(hashNum = hashNum,
                 lr=0.0001,
                 vector="c_img_vd", #c_img_vd, c_text_vd
                 log=False, 
                 batch_size=750,
                 device="cuda:0",
                 num_workers=16,
                 epochs=300
                )
    # E.train()
    modelId = E.hashNum + "_model_" + E.vector + ".pt"
    
    # E.benchmark(average=False)
    # E.benchmark(average=True)
    
    
    # coco_full = torch.load("/export/raid1/home/kneel027/nsd_local/preprocessed_data/" + E.vector + "/vector_73k.pt")#.reshape((73000, 257, 768))[:,0,:]
    # pca = pk.load(open("masks/pca_" + E.vector + "_10k.pkl",'rb'))
    # coco_full = torch.from_numpy(pca.transform(coco_full.numpy())).to(torch.float32)
    # coco_preds_full = torch.zeros((73000, 11838))
    # for i in range(4):
    #     coco_preds_full[18250*i:18250*i + 18250] = E.predict(coco_full[18250*i:18250*i + 18250]).cpu()
    # pruned_encodings = prune_vector(coco_preds_full)
    # # outputs = torch.zeros((73000,10000))
    # # for i in range(1000):
    # #     outputs[1000*i:i*1000 + 1000] = E.predict(coco_full[1000*i:i*1000 + 1000])
    # torch.save(pruned_encodings, "/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/latent_vectors/" + modelId + "/coco_brain_preds.pt")
    
    # os.makedirs("/export/raid1/home/kneel027/nsd_local/preprocessed_data/x_encoded/" + modelId, exist_ok=True)
    # _, y = load_nsd(vector = E.vector, batch_size = E.batch_size, 
    #                 num_workers = E.num_workers, loader = False, split = False, pca=True)
    # outputs = E.predict(y)#.reshape((y.shape[0], 257, 768))[:,0,:])
    # torch.save(outputs, "/export/raid1/home/kneel027/nsd_local/preprocessed_data/x_encoded/" + modelId + "/vector.pt")
    
    # E.predict_73K_coco(model=modelId)
    # E.predict_cc3m(model=modelId)
    return hashNum

def train_ss_decoder():
    
    # hashNum = update_hash()
    hashNum = "541"
    SS = SS_Decoder(hashNum = hashNum,
                    vector="c_img_0",
                    log=False, 
                    encoderHash="536",
                    lr=0.0001,
                    batch_size=750,
                    device="cuda:0",
                    num_workers=16,
                    epochs=300
                )
    
    # SS.train()
    
    # modelId = AE.hashNum + "_model_" + AE.vector + ".pt"
    # SS.benchmark()
    SS.benchmark_nsd(AEhash="577", ae=True)


def train_decoder():
    hashNum = update_hash()
    # hashNum = "714"
    D = Decoder(hashNum = hashNum,
                 lr=0.0001,
                 vector="c_img_vd", #c_img_0 , c_text_0, z_img_mixer
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

def train_decoder_pca():
    # hashNum = update_hash()
    hashNum = "714"
    D = Decoder_PCA(hashNum = hashNum,
                 lr=0.001,
                 vector="c_img_vd", #c_img_0 , c_text_0, z_img_mixer
                 log=True, 
                 batch_size=256,
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
    
    _, _, x_param, x_test, _, _, targets_c_i_param, targets_c_i, param_trials, test_trials = load_nsd(vector="c_img_uc", loader=False, average=True)
    _, _, _, _, _, _, targets_c_t, _, _, _ = load_nsd(vector="c_text_uc", loader=False, average=True)
    Dc_i = Decoder_UC(hashNum = "747",
                 vector="c_img_uc", 
                 log=False, 
                 device="cuda",
                 )
    outputs_c_i = Dc_i.predict(x=x_test[idx]).reshape((len(idx), 1, 1024))
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
    
    for i, val in enumerate(tqdm(idx, desc="Generating reconstructions")):
        
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
        nsdId = test_trials[val]
        ground_truth_np_array = nsda.read_images([nsdId], show=True)
        ground_truth = Image.fromarray(ground_truth_np_array[0])
        ground_truth = ground_truth.resize((768, 768), resample=Image.Resampling.LANCZOS)
        empty = Image.new('RGB', (768, 768), color='white')
        rows = 3
        columns = 2
        images = [ground_truth, empty, reconstructed_target_c, reconstructed_output_c, reconstructed_target_c_i, reconstructed_output_c_i]
        captions = ["Ground Truth", "", "Target C_2", "Output C_2", "Target C_i", "Output C_i"]
        figure = tileImages(experiment_title + ": " + str(val), images, captions, rows, columns)
        
        figure.save('/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/reconstructions/' + experiment_title + '/' + str(val) + '.png')

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
