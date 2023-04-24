import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
sys.path.append('src')
from vdvae import VDVAE
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from nsd_access import NSDAccess
from tqdm import tqdm
import time

prep_path = "/export/raid1/home/kneel027/nsd_local/preprocessed_data/"
latent_path = "/export/raid1/home/kneel027/Second-Sight/latent_vectors/"

nsda = NSDAccess('/home/naxos2-raid25/kneel027/home/surly/raid4/kendrick-data/nsd', '/home/naxos2-raid25/kneel027/home/kneel027/nsd_local')

# Iterate through all images and captions in nsd sampled from COCO
start = time.time()
for i in tqdm(range(0, 73000)):
    # Array of image data 1 x 425 x 425 x 3 (Stores pixel intensities)
    img_arr = torch.from_numpy(nsda.read_images([i], show=False)).reshape(541875)
    torch.save(img_arr, "/home/naxos2-raid25/kneel027/home/kneel027/nsd_local/nsddata_stimuli/tensors/images/" + str(i) + ".pt")

end = time.time()
print("elapsed time: ", end - start)
