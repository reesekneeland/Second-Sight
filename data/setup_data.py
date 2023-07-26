import os
import sys
import torch
import numpy as np
import torchvision.transforms as T
sys.path.append('src')
from data_utils import *
import nibabel as nib
from utils import * 
from transformers import CLIPProcessor, CLIPModel
from diffusers import StableUnCLIPImg2ImgPipeline
from vdvae import VDVAE
sys.path.append('vdvae')
from image_utils import *
from model_utils import *

# Create the charts directory
os.makedirs("data/charts", exist_ok=True)
os.makedirs("data/preprocessed_data", exist_ok=True)


##################### Vector Processing ###############################
def sample_from_hier_latents(latents):
  layers_num=len(latents)
  sample_latents = []
  for i in range(layers_num):
    sample_latents.append(torch.tensor(latents[i]).float().cuda())
  return sample_latents


# Transfor latents from flattened representation to hierarchical
def latent_transformation(latents, shapes):
  layer_dims = np.array([2**4,2**4,2**8,2**8,2**8,2**8,2**10,2**10,2**10,2**10,2**10,2**10,2**10,2**10,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**14])
  transformed_latents = []
  for i in range(31):
    t_lat = latents[:,layer_dims[:i].sum():layer_dims[:i+1].sum()]
    c,h,w=shapes[i]
    transformed_latents.append(t_lat.reshape(len(latents),c,h,w))
  return transformed_latents

H = {'image_size': 64, 'image_channels': 3,'seed': 0, 'port': 29500, 'save_dir': './saved_models/test', 'data_root': './', 'desc': 'test', 'hparam_sets': 'imagenet64', 'restore_path': 'imagenet64-iter-1600000-model.th', 'restore_ema_path': 'vdvae/model/imagenet64-iter-1600000-model-ema.th', 'restore_log_path': 'imagenet64-iter-1600000-log.jsonl', 'restore_optimizer_path': 'imagenet64-iter-1600000-opt.th', 'dataset': 'imagenet64', 'ema_rate': 0.999, 'enc_blocks': '64x11,64d2,32x20,32d2,16x9,16d2,8x8,8d2,4x7,4d4,1x5', 'dec_blocks': '1x2,4m1,4x3,8m4,8x7,16m8,16x15,32m16,32x31,64m32,64x12', 'zdim': 16, 'width': 512, 'custom_width_str': '', 'bottleneck_multiple': 0.25, 'no_bias_above': 64, 'scale_encblock': False, 'test_eval': True, 'warmup_iters': 100, 'num_mixtures': 10, 'grad_clip': 220.0, 'skip_threshold': 380.0, 'lr': 0.00015, 'lr_prior': 0.00015, 'wd': 0.01, 'wd_prior': 0.0, 'num_epochs': 10000, 'n_batch': 4, 'adam_beta1': 0.9, 'adam_beta2': 0.9, 'temperature': 1.0, 'iters_per_ckpt': 25000, 'iters_per_print': 1000, 'iters_per_save': 10000, 'iters_per_images': 10000, 'epochs_per_eval': 1, 'epochs_per_probe': None, 'epochs_per_eval_save': 1, 'num_images_visualize': 8, 'num_variables_visualize': 6, 'num_temperatures_visualize': 3, 'mpi_size': 1, 'local_rank': 0, 'rank': 0, 'logdir': './saved_models/test/log'}
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
H = dotdict(H)

H, preprocess_fn = set_up_data(H)

print('Models is Loading')
ema_vae = load_vaes(H)

# Initialize the clip models
init_clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
init_preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

R = StableUnCLIPImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16, variation="fp16"
)

R = R.to("cuda:0")
R.enable_xformers_memory_efficient_attention()

# Variable for latent vectors
latent_shapes_created = False
latent_tensor = torch.zeros((7300, 91168), dtype=torch.float32)

# Variable for tensor of images 
image_tensor = torch.zeros((7300, 541875), dtype=torch.float32)
batch  = 0 

# Variables for clip image vectos
c_i = True
c_i_tensor = torch.zeros((7300, 1024), dtype=torch.float32)


# Iterate through all images and captions in nsd sampled from COCO
for i in tqdm(range(0, 73000)):
    
    # Array of image data 1 x 425 x 425 x 3 (Stores pixel intensities)
    image_read = nsda.read_images([i], show=False)
    img_array = torch.from_numpy(image_read).reshape(541875)
    
    # Concetate the new image onto the tensor of images
    image_tensor[i - (batch * 7300)] = img_array
        
    # Process image for VDVAE
    img_pil = Image.fromarray(image_read.reshape((425, 425, 3))).convert("RGB")
    img_tensor = T.functional.resize(img_pil,(64,64))
    img_tensor = torch.tensor(np.array(img_tensor)).float()[None,:,:,:]
    
    # Create the latent VDVAE Z vector
    latents = []
    data_input, _ = preprocess_fn(img_tensor)
    with torch.no_grad():
        activations = ema_vae.encoder.forward(data_input)
        px_z, stats = ema_vae.decoder.forward(activations, get_latents=True)
        batch_latent = []
        latent_shapes = []
        for j in range(31):
            latent_shapes.append(torch.tensor(stats[j]['z'].shape[1:]))
            batch_latent.append(stats[j]['z'].cpu().numpy().reshape(len(data_input),-1))
        latents.append(np.hstack(batch_latent))
    latents = np.concatenate(latents)
    latents = torch.from_numpy(latents)
    
    # Save the c and z vectors into there corresponding files. 
    latent_shapes = torch.stack(latent_shapes)
    if(not latent_shapes_created): 
        torch.save(latent_shapes, "vdvae/vdvae_shapes.pt")
        latent_shapes_created = True
        
    # Concetate the new latent vector onto the tensor of latents
    latent_tensor[i - (batch * 7300)] = latents
    
    # Store the clip image vectors if the users wants them. 
    if(c_i):
        c_i_tensor[i - (batch * 7300)] = R.encode_image_raw(image=img_pil, device="cuda:0")
    
    # Save the tensor of images at every ten percent increment
    if(i % 7300 == 0):
        if(c_i):
            torch.save(c_i_tensor,  "data/preprocessed_data/c_i_{}.pt".format(batch))
            c_i_tensor = torch.zeros((7300, 1024), dtype=torch.float32)
        
        torch.save(image_tensor,  "data/preprocessed_data/images_{}.pt".format(batch))
        torch.save(latent_tensor, "data/preprocessed_data/z_vdvae_{}.pt".format(batch))
        image_tensor  = torch.zeros((7300, 541875), dtype=torch.float32)
        latent_tensor = torch.zeros((7300, 91168), dtype=torch.float32)
        batch += 1
        

# Concatenate the ten smaller tensors into one tensor block.       
if(c_i):
    process_raw_tensors(vector="c_i")
process_raw_tensors(vector="images")
process_raw_tensors(vector="z_vdvae")

# Store the mean and standard deviation of the vdvae vectors for normalization purposes. 
vdvae_73k = torch.load("data/preprocessed_data/z_vdvae_73k.pt")
torch.save(torch.mean(vdvae_73k, dim=0), "vdvae/train_mean.pt")
torch.save(torch.std(vdvae_73k, dim=0),  "vdvae/train_std.pt")


##################### Brain Data Processing ###############################
# Process the brain data and masks
subjects = [1, 2, 5, 7]
for subject in subjects:
    
    create_whole_region_unnormalized(subject=subject)
    create_whole_region_normalized(subject=subject)
    
    process_data(subject=subject, vector="c_i")
    process_data(subject=subject, vector="images")
    process_data(subject=subject, vector="z_vdvae")
    
    process_masks(subject=subject)
