import argparse, os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
import PIL
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
from pytorch_lightning import seed_everything
from imwatermark import WatermarkEncoder
from matplotlib import pyplot as plt
from nsd_access import NSDAccess

sys.path.append('../')
from stablediffusion.ldm.util import instantiate_from_config
from stablediffusion.ldm.models.diffusion.ddim import DDIMSampler

class Encoder():
    def __init__(self):
        seed_everything(42)
        self.config = OmegaConf.load("stablediffusion/configs/stable-diffusion/v2-inference.yaml")
        self.model = self.load_model_from_config(self.config, "stablediffusion/models/v2-1_512.ckpt")
        
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = self.model.to(self.device)
        self.scale = 9.0
        self.sampler = DDIMSampler(self.model)

        os.makedirs("reconstructions/", exist_ok=True)
        self.outpath = "reconstructions/"
        self.base_count = len(os.listdir(self.outpath))
        
    def load_model_from_config(self, config, ckpt, verbose=False):
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]
        model = instantiate_from_config(config.model)
        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0 and verbose:
            print("missing keys:")
            print(m)
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u)

        model.cuda()
        model.eval()
        return model


    # Encode latent z (1x4x64x64) and condition c (1x77x1024) tensors into an image
    # Strength parameter controls the weighting between the two tensors
    def reconstruct(self, z, c, strength=0.8):
        init_latent = z.reshape((1,4,64,64)).to(self.device)
        self.sampler.make_schedule(ddim_num_steps=50, ddim_eta=0.0, verbose=False)

        assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
        t_enc = int(strength * 50)
        print(f"target t_enc is {t_enc} steps")

        precision_scope = autocast
        with torch.no_grad():
            with precision_scope("cuda"):
                with self.model.ema_scope():
                    uc = None
                    if self.scale != 1.0:
                        uc = self.model.get_learned_conditioning(1 * [""])
                    c = c.reshape((1,77,1024)).to(self.device)
                    z_enc = self.sampler.stochastic_encode(init_latent, torch.tensor([t_enc] * 1).to(self.device))
                    samples = self.sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=self.scale,
                                            unconditional_conditioning=uc, )

                    x_sample = self.model.decode_first_stage(samples)
                    x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)

                    x_sample = 255. * rearrange(x_sample[0].cpu().numpy(), 'c h w -> h w c')
                    img = Image.fromarray(x_sample.astype(np.uint8))
                    img.save(os.path.join(self.outpath, f"{self.base_count:05}.png"))
                    self.base_count += 1
        return img
    
    # Encode latent z (1x4x64x64) and condition c (1x77x1024) tensors into an image
    # Strength parameter controls the weighting between the two tensors
    def reconstructNImages(self, z, output_c, target_c, strength_c,
                                 c, output_z, target_z, strength_z=0.8,
                                 reconstruction_strength=0.8):
        
        
        reconstructed_output_c = self.reconstruct(z, output_c[0], strength_c)
        reconstructed_target_c = self.reconstruct(z, target_c[0], strength_c)
        reconstructed_output_z  = self.reconstruct(output_z, c , strength_z)
        reconstructed_target_z  = self.reconstruct(target_z, c , strength_z)
        z_c_reconstruction = self.reconstruct(output_z, output_c[0], reconstruction_strength)
        
        
        # self.reconstruct()
        # self.reconstruct()
        
        #imgs = [reconstructed_c, ground_truth_c, reconstructed_c_1, ground_truth_c_1]
        
        # First URL: This is the original read-only NSD file path (The actual data)
        # Second URL: Local files that we are adding to the dataset and need to access as part of the data
        # Object for the NSDAccess package
        # nsda = NSDAccess('/home/naxos2-raid25/kneel027/home/surly/raid4/kendrick-data/nsd', '/home/naxos2-raid25/kneel027/home/kneel027/nsd_local')
        
        # # Retriving the ground truth image. 
        # subj1 = nsda.stim_descriptions[nsda.stim_descriptions['subject1'] != 0]
        # i=27750
        # index = int(subj1.loc[(subj1['subject1_rep0'] == i) | (subj1['subject1_rep1'] == i) | (subj1['subject1_rep2'] == i)].nsdId)
        # print(index)
        # img = nsda.read_images([index], show=True)
        
        # create figure
        fig = plt.figure(figsize=(10, 7))
        
        # setting values to rows and column variables
        rows = 3
        columns = 2
        
        # Adds a subplot at the 1st position
        fig.add_subplot(rows, columns, 1)
        
        # showing image
        plt.imshow(z_c_reconstruction)
        plt.axis('off')
        plt.title("Ground Truth")
        
        # Adds a subplot at the 2nd position
        fig.add_subplot(rows, columns, 2)
        
        # showing image
        plt.imshow(z_c_reconstruction)
        plt.axis('off')
        plt.title("Z and C Reconstructed")
        
        
        # Adds a subplot at the 1st position
        fig.add_subplot(rows, columns, 3)
        
        # showing image
        plt.imshow(reconstructed_output_c)
        plt.axis('off')
        plt.title("Reconstructed Output C")
        
        # Adds a subplot at the 2nd position
        fig.add_subplot(rows, columns, 4)
        
        # showing image
        plt.imshow(reconstructed_target_c)
        plt.axis('off')
        plt.title("Reconstructed Target C")
        
        # Adds a subplot at the 3rd position
        fig.add_subplot(rows, columns, 5)
        
        # showing image
        plt.imshow(reconstructed_output_z)
        plt.axis('off')
        plt.title("Reconstructed Output Z")
        
        # Adds a subplot at the 4th position
        fig.add_subplot(rows, columns, 6)
        
        # showing image
        plt.imshow(reconstructed_target_z)
        plt.axis('off')
        plt.title("Reconstructed Target Z")
        
        plt.show()
        plt.savefig('FourImageReconstruction.png')

    if __name__ == "__main__":
        z = torch.load(sys.argv[1])
        c = torch.load(sys.argv[2])
        strength = float(sys.argv[3])    
        reconstruct(z, c, strength)
