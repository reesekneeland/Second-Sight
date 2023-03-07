import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
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
from encoder import Encoder
from reconstructor import Reconstructor
from autoencoder  import AutoEncoder
from alexnet_encoder import Alexnet
from ss_decoder import SS_Decoder
from mask import Masker

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
#    395_z_img_mixer2voxels.pt
#       - Model of 5496 voxels out of 11838 with a threshold of 0.08283 (Used for training the decoder)
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
# 385_model_z_img_mixer.pt (BEST Z MODEL)
#    - 5615
#    - trained on new MLP
#    - old normalization method
#    - 0.064564
#
#
#   Encoders:
#      # 417_model_c_img_0.pt
#     - old norm
#
# 419_model_c_text_0.pt
#     - old norm
#
# 420_model_z_img_mixer.pt
#     - old norm
# ---------------------------
# 424_model_c_img_0.pt
#     - Z score
#
# 425_model_c_text_0.pt
#     - Z score
#
# 426_model_z_img_mixer.pt
#     - Z score


def main():
    os.chdir("/export/raid1/home/kneel027/Second-Sight/")
    # _, _, _, _, _, _, _, _, _, _, _ = load_nsd(vector="c_img_0", loader=False, average=True)
    
    # train_decoder()

    # train_encoder()
    
    # mask_voxels()

    # load_cc3m("c_img_0", "410_model_c_img_0.pt")

    # reconstructNImages(experiment_title="Tiled MLP Params", idx=[i for i in range(21)])

    # test_reconstruct()

    train_autoencoder()

    # train_ss_decoder()

def mask_voxels():
    M = Masker(encoderHash="521",
                 vector="c_img_0",
                 device="cuda:1")
    # thresholds = list(torch.arange(0.01, 1.0, 0.015))
    # for t in tqdm(thresholds, desc="thresholds"): 
    #     print(t)
    #     x = torch.tensor(0.013)
    #     print(x)
    #     M.get_percentile_coco(x)
    # M.make_histogram()
    M.create_mask(threshold=-1)
    
def train_autoencoder():
    
    hashNum = update_hash()
    # hashNum = "546"
    
    AE = AutoEncoder(hashNum = hashNum,
                        lr=0.0000005,
                        vector="c_img_0", #c_img_0, c_text_0, z_img_mixer, alexnet_encoder_sub1
                        encoderHash="521",
                        log=True, 
                        parallel=False,
                        device="cuda:0",
                        num_workers=16,
                        epochs=300
                        )
    
    AE.train()
    AE.benchmark(encodedPass=False)
    
    # AN = Alexnet()
    
    # modelId = AE.hashNum + "_model_" + AE.vector + ".pt"
    
    # os.makedirs("/export/raid1/home/kneel027/nsd_local/preprocessed_data/x_encoded/" + modelId, exist_ok=True)
    # _, _, _, _, _, _, _, _, _, images = load_nsd(vector = "c_img_0", batch_size = AE.batch_size,
    #                                              num_workers = AE.num_workers, loader = False, 
    #                                              split = True, return_images=True)
    # print("This is the length of PIL images: ", len(images))
    # print("This is the type: ", type(images[0]))
    # outputs = AN.predict(images, [])
    # print("This is the output shape: ", outputs.shape)
    # torch.save(outputs, "/export/raid1/home/kneel027/nsd_local/preprocessed_data/x_encoded/" + modelId + "/vector.pt")


def train_encoder():
    # hashNum = update_hash()
    hashNum = "424"
    E = Encoder(hashNum = hashNum,
                 lr=0.000005,
                 vector="c_img_0", #c_img_0, c_text_0, z_img_mixer
                 log=False, 
                 batch_size=750,
                 parallel=False,
                 device="cuda:0",
                 num_workers=16,
                 epochs=300
                )
    # E.train()
    modelId = E.hashNum + "_model_" + E.vector + ".pt"
    
    # E.benchmark()
    
    os.makedirs("/export/raid1/home/kneel027/nsd_local/preprocessed_data/x_encoded/" + modelId, exist_ok=True)
    _, y = load_nsd(vector = E.vector, batch_size = E.batch_size, 
                    num_workers = E.num_workers, loader = False, split = False)
    outputs = E.predict(y)
    torch.save(outputs, "/export/raid1/home/kneel027/nsd_local/preprocessed_data/x_encoded/" + modelId + "/vector.pt")
    
    # E.predict_73K_coco(model=modelId)
    # E.predict_cc3m(model=modelId)
    return hashNum

def train_ss_decoder():
    
    # hashNum = update_hash()
    hashNum = "547"
    SS = SS_Decoder(hashNum = hashNum,
                    vector="c_img_0",
                    log=False, 
                    encoderHash="521",
                    lr=0.0001,
                    batch_size=750,
                    parallel=False,
                    device="cuda:0",
                    num_workers=16,
                    epochs=300
                )
    
    # SS.train()
    
    # modelId = AE.hashNum + "_model_" + AE.vector + ".pt"
    # SS.benchmark()
    SS.benchmark_nsd(AEhash="544", ae=True)


def train_decoder():
    # hashNum = update_hash()
    hashNum = "528"
    D = Decoder(hashNum = hashNum,
                 lr=0.000005,
                 vector="c_img_0", #c_img_0 , c_text_0, z_img_mixer
                 log=False, 
                 inpSize = 11838,
                 batch_size=750,
                 parallel=False,
                 device="cuda:0",
                 num_workers=16,
                 epochs=300
                )
    # D.train()
    modelId = D.hashNum + "_model_" + D.vector + ".pt"
    
    D.benchmark()
    
    return hashNum

def test_reconstruct():
    R = Reconstructor()
    z = torch.load("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/outputs_z_broken.pt").to("cuda")
    print(z.shape)
    R.reconstruct(z=z[0], strength=0.0)
    R.reconstruct(z=z[1], strength=0.0)
    R.reconstruct(z=z[20], strength=0.0)
    
    ground_truth_np_array = nsda.read_images([13], show=True)
    ground_truth = Image.fromarray(ground_truth_np_array[0])
    ground_truth.save("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/y_gt.png")

# Encode latent z (1x4x64x64) and condition c (1x77x1024) tensors into an image
# Strength parameter controls the weighting between the two tensors
def reconstructNImages(experiment_title, idx):
    Dz = Decoder(hashNum = "531",
                 vector="z_img_mixer", 
                 inpSize = 11838,
                 log=False, 
                 device="cuda",
                 parallel=False
                 )
    
    Dc_i = Decoder(hashNum = "528",
                 vector="c_img_0", 
                 inpSize = 11838,
                 log=False, 
                 device="cuda",
                 parallel=False
                 )
    SS_Dc_i = SS_Decoder(hashNum = "541",
                 vector="c_img_0",
                 encoderHash="536",
                 log=False, 
                 device="cuda:0"
                 )
    
    Dc_t = Decoder(hashNum = "529",
                 vector="c_text_0", 
                 inpSize = 11838,
                 log=False, 
                 device="cuda",
                 parallel=False
                 )
    AE = AutoEncoder(hashNum = "540",
                 lr=0.0000001,
                 vector="c_img_0", #c_img_0, c_text_0, z_img_mixer
                 encoderHash="536",
                 log=False, 
                 batch_size=750,
                 parallel=False,
                 device="cuda"
                )
    
    # First URL: This is the original read-only NSD file path (The actual data)
    # Second URL: Local files that we are adding to the dataset and need to access as part of the data
    # Object for the NSDAccess package
    nsda = NSDAccess('/home/naxos2-raid25/kneel027/home/surly/raid4/kendrick-data/nsd', '/home/naxos2-raid25/kneel027/home/kneel027/nsd_local')
    os.makedirs("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/reconstructions/" + experiment_title + "/", exist_ok=True)
    # Load test data and targets
    _, _, x_param, x_test, _, _, targets_c_i, _, param_trials, test_trials = load_nsd(vector="c_img_0", loader=False, average=True)
    _, _, _, _, _, _, targets_c_t, _, _, _ = load_nsd(vector="c_text_0", loader=False, average=True)
    _, _, _, _, _, _, targets_z, _, _, _ = load_nsd(vector="z_img_mixer", loader=False, average=True)
    
    # Generating predicted and target vectors
    # ae_x_test = AE.predict(x_test)
    # outputs_c_i = SS_Dc_i.predict(x=ae_x_test)
    
    outputs_c_i = Dc_i.predict(x=x_param)
    outputs_c_t = Dc_t.predict(x=x_param)
    outputs_z = Dz.predict(x=x_param)
    strength_c = 1
    strength_z = 0
    R = Reconstructor()
    for i in tqdm(idx, desc="Generating reconstructions"):
        
        c_combined = format_clip(torch.stack([outputs_c_i[i], outputs_c_t[i]]))
        c_combined_target = format_clip(torch.stack([targets_c_i[i], targets_c_t[i]]))
        # c_combined = format_clip(outputs_c_i[i])
        # c_combined_target = format_clip(targets_c_i[i])
        
        # Make the c reconstrution images. 
        reconstructed_output_c = R.reconstruct(c=c_combined, strength=strength_c)
        reconstructed_target_c = R.reconstruct(c=c_combined_target, strength=strength_c)
        
        # # Make the z reconstrution images. 
        reconstructed_output_z = R.reconstruct(z=outputs_z[i], strength=strength_z)
        reconstructed_target_z = R.reconstruct(z=targets_z[i], strength=strength_z)
        
        # # Make the z and c reconstrution images. 
        z_c_reconstruction = R.reconstruct(z=outputs_z[i], c=c_combined, strength=0.85)
        
        # returns a numpy array 
        nsdId = param_trials[i]
        ground_truth_np_array = nsda.read_images([nsdId], show=True)
        ground_truth = Image.fromarray(ground_truth_np_array[0])
        ground_truth = ground_truth.resize((512, 512), resample=Image.Resampling.LANCZOS)
        rows = 3
        columns = 2
        images = [ground_truth, z_c_reconstruction, reconstructed_target_c, reconstructed_output_c, reconstructed_target_z, reconstructed_output_z]
        captions = ["Ground Truth", "Z and C Reconstructed", "Reconstructed Target C", "Reconstructed Output C", "Reconstructed Target Z", "Reconstructed Output Z"]
        figure = tileImages(experiment_title + ": " + str(i), images, captions, 3, 2)
        
        
        figure.save('reconstructions/' + experiment_title + '/' + str(i) + '.png')
if __name__ == "__main__":
    main()
