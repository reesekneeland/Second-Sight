import h5py
from PIL import Image
import scipy.io
import argparse, os
import pandas as pd
import PIL
import torch
import numpy as np
from omegaconf import OmegaConf
from tqdm import trange
from einops import rearrange
from torch import autocast
from contextlib import nullcontext
from pytorch_lightning import seed_everything
from random import randrange
import sys
sys.path.append("../utils/")
from nsd_access.nsda import NSDAccess
sys.path.append("stable-diffusion")
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
import pickle


def load_model_from_config(config, ckpt, gpu, verbose=False):
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
    model.cuda(f"cuda:{gpu}")
    model.eval()
    return model

def load_img(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert("RGB")
    elif isinstance(image, torch.Tensor):
        image = Image.fromarray(image.numpy().reshape((425, 425, 3)).astype(np.uint8))
    if not isinstance(image, PIL.Image.Image):
        raise ValueError("image must be a PIL Image, a numpy array, or a torch tensor")
    w, h = 512, 512
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.
    


class Reconstructor(object):
    def __init__(self, device='cuda:0'):
        self.device = device
        self.ddim_steps = 50
        ddim_eta = 0.0
        self.scale = 5.0
        # Load Stable Diffusion Model
        seed=randrange(10000)
        seed_everything(seed=seed)
        config = './stable-diffusion/configs/stable-diffusion/v1-inference.yaml'
        ckpt = './stable-diffusion/models/ldm/stable-diffusion-v1/sd-v1-4.ckpt'
        config = OmegaConf.load(f"{config}")
        torch.cuda.set_device(device)
        self.model = load_model_from_config(config, f"{ckpt}", self.device)
        self.model = self.model.to(self.device)
        self.sampler = DDIMSampler(self.model)
        self.sampler.make_schedule(ddim_num_steps=self.ddim_steps, ddim_eta=ddim_eta, verbose=False)
        print('Reconstructor_SD1.4 initialized')
    
    def encode_text(self, prompt):
        text_encoding = self.model.get_learned_conditioning(prompt)
        return text_encoding
    
    def reconstruct(self, 
                    image=None, 
                    c_i=None, 
                    c_t=None, 
                    n_samples=1, 
                    textstrength=1.0, 
                    strength=1.0, 
                    color_adjust=False,
                    fcs_lvl=0.5, 
                    seed=None
                    ):
        if strength == 0:
            return [image]*n_samples
        assert (c_t is not None) and (textstrength == 1.0), "Only text configs supported for SD 1.4"
        assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
        t_enc = int(strength * self.ddim_steps)
        print(f"target t_enc is {t_enc} steps")
    
        precision = 'autocast'
        precision_scope = autocast if precision == "autocast" else nullcontext
        batch_size = 1
        init_image = load_img(image).to(self.device)
        init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

        c = c_t.to(self.device)
        # Generate image from Z (image) + C (semantics)
        base_count = 0
        outputs = []
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    for n in range(n_samples):
                        uc = self.model.get_learned_conditioning(batch_size * [""])

                        # encode (scaled latent)
                        z_enc = self.sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(self.device))
                        # decode it
                        samples = self.sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=self.scale,
                                                unconditional_conditioning=uc,)

                        x_samples = self.model.decode_first_stage(samples)
                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                        for x_sample in x_samples:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        
                        outputs.append(Image.fromarray(x_sample.astype(np.uint8)))
                        base_count += 1
        if len(outputs) == 1:
            return outputs[0]
        else 
            return outputs

if __name__ == "__main__":
    
    reconstructor = Reconstructor()
    for mode in ["vision", "imagery"]:
        for subject in ['subj01','subj02','subj05','subj07']:
            captdir = f'/home/naxos2-raid25/kneel027/home/kneel027/StableDiffusionReconstruction/decoded/{subject}/captions/'
            outdir = f"/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/output/mental_imagery_paper_b3/{mode}/tagaki/subject{subject[-1]}/"
            captions = pd.read_csv(f'{captdir}/captions_brain_mi_{mode}.csv', sep='\t',header=None)
            for sample in range(18):
                c = reconstructor.encode_text(captions.iloc[sample][0])
                torch.save(c, f'{outdir}/{sample}/clip_text.pt')