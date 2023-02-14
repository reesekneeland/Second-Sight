import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
import torch
import numpy as np
from PIL import Image
sys.path.append('../src')
from reconstructor import Reconstructor

from PIL import Image
import random
from nsd_access import NSDAccess
from tqdm import tqdm
import time

def load_img(im_path):
    
    image = Image.open(im_path).convert('RGB')
    w, h = 512, 512  # resize to integer multiple of 64
    imagePil = image.resize((w, h), resample=Image.Resampling.LANCZOS)
    image = np.array(imagePil).astype(np.float32) / 255.0
    # print(image.shape)
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(torch.float16)
    return 2. * image - 1., imagePil

rootdir = "/home/naxos2-raid25/kneel027/home/kneel027/nsd_local/cc3m/"
folder_list = []
R = Reconstructor()
start = time.time()
for it in os.scandir(rootdir):
    if it.is_dir():
        folder_list.append(it.name)
count = 0


# Encoding vectors for 2819140 images
for folder in folder_list:
    if(folder == "_tmp"):
        pass
    print(folder)
    for file in tqdm(sorted(os.scandir(rootdir + folder), key=lambda e: e.name)):
        try:
            if file.name.endswith(".jpg"):
                im_tensor, im = load_img(rootdir + folder + "/" + file.name)
                image_embed = R.get_im_c(im).reshape((768,))
                latent_vector = R.encode_latents(im_tensor).reshape((16384,))
                torch.save(image_embed, rootdir + "tensors/c_img_0/" + str(count) + ".pt")
                torch.save(latent_vector, rootdir + "tensors/z_img_mixer/" + str(count) + ".pt")
            elif file.name.endswith(".txt"):
                with open(file) as f:
                    prompt = f.readline()
                if(len(prompt)>75):
                    prompt = prompt[:75]
                text_embed = R.get_txt_c(prompt).reshape((768,))
                torch.save(text_embed, rootdir + "tensors/c_text_0/" + str(count) + ".pt")
                count +=1
            else:
                pass
        except:
            print(file.name)
end = time.time()

print(count, start-end)