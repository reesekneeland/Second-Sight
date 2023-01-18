"""make variations of input image"""

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


from stablediffusion.ldm.util import instantiate_from_config
from stablediffusion.ldm.models.diffusion.ddim import DDIMSampler

def load_model_from_config(config, ckpt, verbose=False):
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



def reconstruct(z, c, strength=0.8):
    seed_everything(42)
    config = OmegaConf.load("stablediffusion/configs/stable-diffusion/v2-inference.yaml")
    model = load_model_from_config(config, "stablediffusion/models/v2-1_512.ckpt")
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    scale = 9.0
    sampler = DDIMSampler(model)

    os.makedirs("reconstructions/", exist_ok=True)
    outpath = "reconstructions/"
    base_count = len(os.listdir(outpath))
    
    init_latent = z.reshape((1,4,64,64)).to(device)
    sampler.make_schedule(ddim_num_steps=50, ddim_eta=0.0, verbose=False)

    assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(strength * 50)
    print(f"target t_enc is {t_enc} steps")

    precision_scope = autocast
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                all_samples = list()
                for n in trange(1, desc="Sampling"):
                        uc = None
                        if scale != 1.0:
                            uc = model.get_learned_conditioning(1 * [""])
                        c = c.reshape((1,77,1024)).to(device)
                        
                        print(c.shape)
                        z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc] * 1).to(device))
                        # decode it
                        samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=scale,
                                                 unconditional_conditioning=uc, )

                        x_sample = model.decode_first_stage(samples)
                        x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)

                        x_sample = 255. * rearrange(x_sample[0].cpu().numpy(), 'c h w -> h w c')
                        img = Image.fromarray(x_sample.astype(np.uint8))
                        img.save(os.path.join(outpath, f"{base_count:05}.png"))
                        base_count += 1
    print(f"Your samples are ready and waiting for you here: \n{outpath} \nEnjoy.")
    return img

if __name__ == "__main__":
    z = torch.load(sys.argv[1])
    c = torch.load(sys.argv[2])
    strength = float(sys.argv[3])    
    reconstruct(z, c, strength)
