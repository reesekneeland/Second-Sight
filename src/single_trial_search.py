import os
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
from encoder import Encoder
from alexnet_encoder import Alexnet
from reconstructor import Reconstructor


class SingleTrialSearch():
    def __init__(self, device="cuda:0"):
        self.device = device
        self.R = Reconstructor()
        self.nsda = NSDAccess('/home/naxos2-raid25/kneel027/home/surly/raid4/kendrick-data/nsd', '/home/naxos2-raid25/kneel027/home/kneel027/nsd_local')
    
    def main():
        generateSamples(experiment_title="Single Trial Search", idx=[i for i in range(20)])

    def generateSamples(experiment_title, idx):    

        os.makedirs("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/reconstructions/" + experiment_title + "/", exist_ok=True)
        # Load test data and targets
        _, _, _, _, x_test, _, _, _, _, targets_c_i, test_trials = load_nsd(vector="c_img_0", loader=False, average=False, nest=True)
        _, _, _, _, _, _, _, _, _, targets_c_t, _ = load_nsd(vector="c_text_0", loader=False, average=False, nest=True)
        _, _, _, _, _, _, _, _, _, targets_z, _ = load_nsd(vector="z_img_mixer", loader=False, average=False, nest=False)
        print(x_test.shape, targets_c_i.shape)
        # Generating predicted and target vectors
        # ae_x_test = AE.predict(x_test)
        # outputs_c_i = SS_Dc_i.predict(x=ae_x_test)
        
        # outputs_c_i = Dc_i.predict(x=x_test)
        # outputs_c_t = Dc_t.predict(x=x_test)
        # outputs_z = Dz.predict(x=x_test)
        # strength_c = 1
        # strength_z = 0
        # R = Reconstructor()
        # for i in idx:
        #     print(i)
            
        #     c_combined = format_clip(torch.stack([outputs_c_i[i], outputs_c_t[i]]))
        #     c_combined_target = format_clip(torch.stack([targets_c_i[i], targets_c_t[i]]))
        #     # c_combined = format_clip(outputs_c_i[i])
        #     # c_combined_target = format_clip(targets_c_i[i])
            
        #     # Make the c reconstrution images. 
        #     reconstructed_output_c = R.reconstruct(c=c_combined, strength=strength_c)
        #     reconstructed_target_c = R.reconstruct(c=c_combined_target, strength=strength_c)
            
        #     # # Make the z reconstrution images. 
        #     reconstructed_output_z = R.reconstruct(z=outputs_z[i], strength=strength_z)
        #     reconstructed_target_z = R.reconstruct(z=targets_z[i], strength=strength_z)
            
        #     # # Make the z and c reconstrution images. 
        #     z_c_reconstruction = R.reconstruct(z=outputs_z[i], c=c_combined, strength=0.85)
            
        #     # returns a numpy array 
        #     nsdId = test_trials[i]
        #     ground_truth_np_array = nsda.read_images([nsdId], show=True)
        #     ground_truth = Image.fromarray(ground_truth_np_array[0])
        #     ground_truth = ground_truth.resize((512, 512), resample=Image.Resampling.LANCZOS)
        #     rows = 3
        #     columns = 2
        #     images = [ground_truth, z_c_reconstruction, reconstructed_target_c, reconstructed_output_c, reconstructed_target_z, reconstructed_output_z]
        #     captions = ["Ground Truth", "Z and C Reconstructed", "Reconstructed Target C", "Reconstructed Output C", "Reconstructed Target Z", "Reconstructed Output Z"]
        #     figure = tileImages(experiment_title + ": " + str(i), images, captions, 3, 2)
            
            
        #     figure.save('reconstructions/' + experiment_title + '/' + str(i) + '.png')
if __name__ == "__main__":
    main()
