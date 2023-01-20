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

    if __name__ == "__main__":
        z = torch.load(sys.argv[1])
        c = torch.load(sys.argv[2])
        strength = float(sys.argv[3])    
        reconstruct(z, c, strength)
