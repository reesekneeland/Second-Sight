import os, sys, shutil
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib as plt
from PIL import Image
from utils import *
from autoencoder import AutoEncoder
from gnet8_encoder import GNet8_Encoder
from reconstructor import Reconstructor
from matplotlib.lines import Line2D
import matplotlib as mpl
import math
import matplotlib.image as mpimg
import random

R = Reconstructor(which='v1.0', fp16=True, device="cuda")
GNet = GNet8_Encoder(device="cuda",subject=1)
seed_path = "/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/output/mindeye_extension_v6/subject1/"
output_path = "/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/output/snr_threshold_v1/subject1/"

for strength in tqdm([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]):
    for sample in tqdm(range(100)):
        clip = torch.load(f"{seed_path}{sample}/clip_best.pt")
        low_level = Image.open(f"{seed_path}{sample}/MindEye blurry.png")
        os.makedirs(f"{output_path}{strength}/{sample}", exist_ok=True)
        torch.save(clip, f"{output_path}{strength}/{sample}/clip.pt")
        low_level.save(f"{output_path}{strength}/{sample}/low_level.png")
        reconstructions = []
        for rep in range(3):
            reconstruction = R.reconstruct(image=low_level, c_i=clip, strength=strength)
            reconstruction.save(f"{output_path}{strength}/{sample}/{rep}.png")
            reconstructions.append(reconstruction)
        beta_primes = GNet.predict(reconstructions)
        torch.save(beta_primes, f"{output_path}{strength}/{sample}/beta_primes.pt")