import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
import torch
import numpy as np
from PIL import Image
sys.path.append('../src')
from encoder import Encoder

from PIL import Image

from nsd_access import NSDAccess
from tqdm import tqdm


def load_img(im_array):

    image = Image.fromarray(im_array)
    w, h = image.size

    # print(f"loaded input image of size ({w}, {h}) from array")

    w, h = 512, 512  # resize to integer multiple of 64
    imagePil = image.resize((w, h), resample=Image.Resampling.LANCZOS)
    image = np.array(imagePil).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2. * image - 1., imagePil


nsda = NSDAccess('/home/naxos2-raid25/kneel027/home/surly/raid4/kendrick-data/nsd', '/home/naxos2-raid25/kneel027/home/kneel027/nsd_local')

captions = nsda.read_image_coco_info([i for i in range(73000)], info_type='captions', show_annot=False)
E = Encoder()
# Iterate through all images and captions in nsd sampled from COCO
for i in tqdm(range(0, 20)):
    # Array of image data 1 x 425 x 425 x 3 (Stores pixel intensities)
    img_arr = nsda.read_images([i], show=False)
    init_image, img_pil = load_img(img_arr[0])
    # image = Image.fromarray(img_arr.reshape((425, 425, 3)))
    
    #find best prompts
    prompts = []
    # Load the 5 prompts for each image
    for j in range(len(captions[i])):
        # Index into the caption list and get the corresponding 5 captions. 
        prompts.append(captions[i][j]['caption'])
    c = E.encode_combined(img_pil, prompts)
    z = E.encode_latents(init_image)
    
    # Save the c and z vectors into there corresponding files. 
    # torch.save(c, "/home/naxos2-raid25/kneel027/home/kneel027/nsd_local/nsddata_stimuli/tensors/c_combined/" + str(i) + ".pt")
    # torch.save(z, "/home/naxos2-raid25/kneel027/home/kneel027/nsd_local/nsddata_stimuli/tensors/z/" + str(i) + ".pt")
    z_only = E.reconstruct(z=z, strength=0.0)
    c_only = E.reconstruct(c=c, z=z, strength=0.95)
    z_and_c = E.reconstruct(z=z, c=c, strength=0.75)
    z_only.save("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/test_imgs4/" + str(i) + "_z_only.png")
    c_only.save("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/test_imgs4/" + str(i) + "_c_only.png")
    z_and_c.save("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/test_imgs/" + str(i) + "_z_and_c.png")
        # image_and_z[0].save("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/imx_mixer/" + str(i) + "_z_and_c.png")

