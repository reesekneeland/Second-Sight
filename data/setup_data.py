import os
import sys
import torch
import numpy as np
import argparse
import torchvision.transforms as T
from data_utils import *
from diffusers import StableUnCLIPImg2ImgPipeline
sys.path.append('vdvae')
from image_utils import *
from model_utils import *
from torch.utils.data import DataLoader, Dataset

class batch_generator(Dataset):

        def __init__(self, data_path):
            self.data_path = data_path
            self.im = torch.load(data_path)


        def __getitem__(self,idx):
            img_array = self.im[idx]
            imgPil = Image.fromarray(img_array.reshape((425, 425, 3)).numpy().astype(np.uint8))
            img = T.functional.resize(imgPil,(64,64))
            img = torch.tensor(np.array(img)).float()
            #img = img/255
            #img = img*2 - 1
            return img, img_array

        def __len__(self):
            return  len(self.im)

def process_images(device):
    # Create the charts directory
    print("Making directories...")
    os.makedirs("data/charts", exist_ok=True)
    os.makedirs("data/preprocessed_data", exist_ok=True)


    print('Preparing VDVAE model...')
    H = {'image_size': 64, 'image_channels': 3,'seed': 0, 'port': 29500, 'save_dir': './saved_models/test', 'data_root': './', 'desc': 'test', 'hparam_sets': 'imagenet64', 'restore_path': 'imagenet64-iter-1600000-model.th', 'restore_ema_path': 'vdvae/model/imagenet64-iter-1600000-model-ema.th', 'restore_log_path': 'imagenet64-iter-1600000-log.jsonl', 'restore_optimizer_path': 'imagenet64-iter-1600000-opt.th', 'dataset': 'imagenet64', 'ema_rate': 0.999, 'enc_blocks': '64x11,64d2,32x20,32d2,16x9,16d2,8x8,8d2,4x7,4d4,1x5', 'dec_blocks': '1x2,4m1,4x3,8m4,8x7,16m8,16x15,32m16,32x31,64m32,64x12', 'zdim': 16, 'width': 512, 'custom_width_str': '', 'bottleneck_multiple': 0.25, 'no_bias_above': 64, 'scale_encblock': False, 'test_eval': True, 'warmup_iters': 100, 'num_mixtures': 10, 'grad_clip': 220.0, 'skip_threshold': 380.0, 'lr': 0.00015, 'lr_prior': 0.00015, 'wd': 0.01, 'wd_prior': 0.0, 'num_epochs': 10000, 'n_batch': 4, 'adam_beta1': 0.9, 'adam_beta2': 0.9, 'temperature': 1.0, 'iters_per_ckpt': 25000, 'iters_per_print': 1000, 'iters_per_save': 10000, 'iters_per_images': 10000, 'epochs_per_eval': 1, 'epochs_per_probe': None, 'epochs_per_eval_save': 1, 'num_images_visualize': 8, 'num_variables_visualize': 6, 'num_temperatures_visualize': 3, 'mpi_size': 1, 'local_rank': 0, 'rank': 0, 'logdir': './saved_models/test/log'}
    class dotdict(dict):
        """dot.notation access to dictionary attributes"""
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__
    H = dotdict(H)
    H, preprocess_fn = set_up_data(H, device=device)
    ema_vae = load_vaes(H, device=device)

    # Initialize the clip models
    print('Preparing CLIP model...')
    R = StableUnCLIPImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16)
    R = R.to(device)

    # Variable for latent vectors
    latent_shapes_created = False
    latent_tensor = torch.zeros((73000, 91168), dtype=torch.float32)

    # Can be reduced to fit on smaller GPUs
    minibatch = 50

    # Variables for clip image vectos
    c_tensor = torch.zeros((73000, 1024), dtype=torch.float32)

    image_read = read_images(image_index=[j for j in range(73000)], show=False)
    img_array = torch.from_numpy(image_read).reshape((73000, 541875))
    torch.save(img_array,  "data/preprocessed_data/images_73k.pt")

    # Iterate through all images in nsd sampled from COCO
    image_batcher = batch_generator(data_path = "data/preprocessed_data/images_73k.pt")
    imageloader = DataLoader(image_batcher,minibatch,shuffle=False)
    for i, (batch, batch_tensor) in tqdm(enumerate(imageloader), desc="Converting COCO images to VDVAE and CLIP vectors", total=len(imageloader)):
        # Create the latent VDVAE Z vector
        latents = []
        data_input, _ = preprocess_fn(batch)
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
        # print("Latents shape", latents.shape)
        # Save the c and z vectors into there corresponding files. 
        latent_shapes = torch.stack(latent_shapes)
        if(not latent_shapes_created): 
            torch.save(latent_shapes, "vdvae/vdvae_shapes.pt")
            latent_shapes_created = True
            
        # Concetate the new latent vector onto the tensor of latents
        latent_tensor[i*minibatch : i*minibatch + minibatch] = latents.to("cpu")
        c_tensor[i*minibatch : i*minibatch + minibatch]  = R.encode_image_raw(images=batch_tensor.reshape((minibatch, 425, 425, 3)), device=device).to("cpu")
    torch.save(c_tensor,  "data/preprocessed_data/c_73k.pt")  
    torch.save(latent_tensor, "data/preprocessed_data/z_vdvae_73k.pt")

    # Store the mean and standard deviation of the vdvae vectors for normalization purposes. 
    vdvae_73k = torch.load("data/preprocessed_data/z_vdvae_73k.pt")
    torch.save(torch.mean(vdvae_73k, dim=0), "vdvae/train_mean.pt")
    torch.save(torch.std(vdvae_73k, dim=0),  "vdvae/train_std.pt")

def process_trial_data(subjects):
    ##################### Brain Data Processing ###############################
    # Process the brain data and masks
    for subject in subjects:
        
        print("Processing brain data for subject {}".format(subject))
        create_whole_region_unnormalized(subject=subject)
        create_whole_region_normalized(subject=subject)
        
        print("Processing training data for subject {}".format(subject))
        process_data(subject=subject, vector="c")
        process_data(subject=subject, vector="images")
        process_data(subject=subject, vector="z_vdvae")
        
        print("Processing masks for subject {}".format(subject))
        process_masks(subject=subject)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-s',
                        '--subjects', 
                        help="list of subjects to run the algorithm on, if not specified, will run on all subjects",
                        type=str,
                        default="1,2,5,7")

    parser.add_argument('-d',
                        '--device', 
                        help="cuda device to run predicts on.",
                        type=str,
                        default="cuda:0")
    

    args = parser.parse_args()
    subject_list = [int(sub) for sub in args.subjects.split(",")]
    process_images(args.device)
    process_trial_data(subject_list)