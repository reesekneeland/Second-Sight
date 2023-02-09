import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
import torch
import numpy as np
from PIL import Image
sys.path.append('../src')
from encoder import Encoder, load_img

from PIL import Image
import random
from nsd_access import NSDAccess
from tqdm import tqdm
import time

nsda = NSDAccess('/home/naxos2-raid25/kneel027/home/surly/raid4/kendrick-data/nsd', '/home/naxos2-raid25/kneel027/home/kneel027/nsd_local')

captions = nsda.read_image_coco_info([i for i in range(73000)], info_type='captions', show_annot=False)
img_arr = nsda.read_images([0, 1, 2, 3, 4], show=False)
_, img_pil0 = load_img(img_arr[0])
_, img_pil1 = load_img(img_arr[1])
_, img_pil2 = load_img(img_arr[2])
_, img_pil3 = load_img(img_arr[3])
_, img_pil4 = load_img(img_arr[4])
imgs = [img_pil0, img_pil1, img_pil2, img_pil3, img_pil4]
E = Encoder()
# Iterate through all images and captions in nsd sampled from COCO
start = time.time()
for i in tqdm(range(0, 5000)):
    # Array of image data 1 x 425 x 425 x 3 (Stores pixel intensities)
    img_pil = random.choice(imgs)
    # image = Image.fromarray(img_arr.reshape((425, 425, 3)))
    
    #find best prompts
    # prompts = []
    # Load the 5 prompts for each image
    # for j in range(len(captions[i])):
    #     # Index into the caption list and get the corresponding 5 captions. 
    #     prompts.append(captions[i][j]['caption'])
    # c = E.encode_combined(img_pil, prompts)
    c = E.encode_image(img_pil)
    # z = E.encode_latents(init_image)
    
    # Save the c and z vectors into there corresponding files. 
    # torch.save(c, "/home/naxos2-raid25/kneel027/home/kneel027/nsd_local/nsddata_stimuli/tensors/c_combined/" + str(i) + ".pt")
    # torch.save(z, "/home/naxos2-raid25/kneel027/home/kneel027/nsd_local/nsddata_stimuli/tensors/z_img_mixer/" + str(i) + ".pt")
    # if(i<=10):
    #     z_only = E.reconstruct(z=z, strength=0.0)
        # c_only = E.reconstruct(c=c, strength=1.0)
        # z_and_c = E.reconstruct(z=z, c=c, strength=0.75)
        
        # z_only.save("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/test_imgs4/" + str(i) + "_z_only.png")
        # c_only.save("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/test_imgs4/" + str(i) + "_c_only.png")
        # z_and_c.save("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/test_imgs/" + str(i) + "_z_and_c.png")
            # image_and_z[0].save("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/imx_mixer/" + str(i) + "_z_and_c.png")

end = time.time()
print("elapsed time: ", end - start)