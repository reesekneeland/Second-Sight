import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
from diffusers import StableDiffusionImageEncodingPipeline, StableDiffusionTextEncodingPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionImageVariationPipeline
from PIL import Image
import torch
import numpy as np
from nsd_access import NSDAccess
import open_clip
from tqdm import tqdm


nsda = NSDAccess('/home/naxos2-raid25/kneel027/home/surly/raid4/kendrick-data/nsd', '/home/naxos2-raid25/kneel027/home/kneel027/nsd_local')

# Setting up the device to use the GPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# Initialize OpenClip model
openclip_model, _, oc_preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k', device=device)


#home to encode_image() and encode()
im_model = StableDiffusionImageEncodingPipeline.from_pretrained("lambdalabs/sd-image-variations-diffusers",revision="v2.0")
im_model = im_model.to(device)
#home to encode_prompt() and encode_latents()
text_model = StableDiffusionTextEncodingPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to(device)
text_model = text_model.to(device)


# Importing the caption object with 5 captions per image
# Captions is a lists of lists, containing the 5 captions. 
captions = nsda.read_image_coco_info([i for i in range(73000)], info_type='captions', show_annot=False)

# Iterate through all images and captions in nsd sampled from COCO
for i in tqdm(range(0, 73000)):
    # Array of image data 1 x 425 x 425 x 3 (Stores pixel intensities)
    img_arr = nsda.read_images([i], show=False)
    image = Image.fromarray(img_arr.reshape((425, 425, 3)))

    w, h = 512, 512  # resize to integer multiple of 64
    image = image.resize((w, h), resample=Image.Resampling.LANCZOS)
    
    c = im_model.encode_image(image)
                
    # Use the Stable Diffusion method to move to latent space to create latent z vector
    z = text_model.encode_latents(image)
    
    # Save the c and z vectors into there corresponding files. 
    torch.save(c, "/home/naxos2-raid25/kneel027/home/kneel027/nsd_local/nsddata_stimuli/tensors/c_img/" + str(i) + ".pt")
    # torch.save(z, "/home/naxos2-raid25/kneel027/home/kneel027/nsd_local/nsddata_stimuli/tensors/z/" + str(i) + ".pt")
    # if(i<10):
    #   z_only = im_model.encode(z=z, strength=0.0)
    #   c_only_image = im_model.encode(c=c, strength=1.0)
    #   image_and_z = im_model.encode(z=z, c=c, strength=0.75)
    #   z_only.save("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/test_imgs/" + str(i) + "_z_only.png")
    #   c_only_image.save("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/test_imgs/" + str(i) + "_c_only.png")
    #   image_and_z.save("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/test_imgs/" + str(i) + "_z_and_c.png")
