import sys, os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
sys.path.append('versatile_diffusion')
import os.path as osp
import PIL
from PIL import Image
from pathlib import Path
import numpy as np
import numpy.random as npr
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as tvtrans
from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model
from lib.experiments.sd_default import color_adjust, auto_merge_imlist
from torch.utils.data import DataLoader, Dataset
from lib.model_zoo.ddim_vd import DDIMSampler_VD
from lib.model_zoo.vd import VD
from lib.cfg_holder import cfg_unique_holder as cfguh
from lib.cfg_helper import get_command_line_args, cfg_initiates, load_cfg_yaml

from skimage.transform import resize, downscale_local_mean

def regularize_image(x):
    BICUBIC = PIL.Image.Resampling.BICUBIC
    if isinstance(x, str):
        x = Image.open(x).resize([512, 512], resample=BICUBIC)
        x = tvtrans.ToTensor()(x)
    elif isinstance(x, PIL.Image.Image):
        x = x.resize([512, 512], resample=BICUBIC)
        x = tvtrans.ToTensor()(x)
    elif isinstance(x, np.ndarray):
        x = PIL.Image.fromarray(x).resize([512, 512], resample=BICUBIC)
        x = tvtrans.ToTensor()(x)
    elif isinstance(x, torch.Tensor):
        pass
    else:
        assert False, 'Unknown image type'

    assert (x.shape[1]==512) & (x.shape[2]==512), \
        'Wrong image size'
    return x
class Reconstructor(object):
    #Can't take device, must be configured via CUDA_VISIBLE_DEVICES
    def __init__(self, device='cuda:0'):
        cfgm_name = 'vd_noema'
        sampler = DDIMSampler_VD
        pth = 'versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth'
        cfgm = model_cfg_bank()(cfgm_name)
        self.net = get_model()(cfgm)
        sd = torch.load(pth, map_location='cpu')
        self.net.load_state_dict(sd, strict=False)   
        self.sampler = sampler(self.net)
        self.net.clip.cuda(0)
        self.net.autokl.cuda(0)
        self.net.autokl.half()
        print('Reconstructor_BD initialized')
    
    
    def encode_text(self, prompt):
        text_encoding = self.net.clip_encode_text(prompt)
        return text_encoding
    
    def encode_image(self, image):
        BICUBIC = PIL.Image.Resampling.BICUBIC
        cx = image.resize([512, 512], resample=BICUBIC)
        cx = tvtrans.ToTensor()(cx)[None].cuda(0).half()
        image_encoding = self.net.clip_encode_vision(cx, which='image')
        return image_encoding

    def reconstruct(self, 
                    image=None, 
                    c_i=None, 
                    c_t=None, 
                    n_samples=1, 
                    textstrength=0.5, 
                    strength=1.0, 
                    color_adjust=False,
                    fcs_lvl=0.5, 
                    seed=None
                    ):
        if strength == 0:
            return [image]*n_samples
        elif c_t is None:
            assert (c_i is not None)
            textstrength=0.0
            cim = c_i.unsqueeze(0).cuda(0).half()
            dummy = ''
            ctx = self.net.clip_encode_text(dummy)
            ctx = ctx.cuda(0).half()
        elif c_i is None:
            assert (c_t is not None)
            textstrength=1.0
            ctx = c_t.unsqueeze(0).cuda(0).half()
            dummy = torch.zeros((1,3,224,224)).cuda(0)
            cim = self.net.clip_encode_vision(dummy)
            cim = cim.cuda(0).half()
        else:
            assert (c_t is not None) and (c_i is not None)
            cim = c_i.unsqueeze(0).cuda(0).half()
            ctx = c_t.unsqueeze(0).cuda(0).half()
            
        ddim_steps = 50
        ddim_eta = 0
        scale = 3.5
        xtype = 'image'
        ctype = 'prompt'
        zim = regularize_image(image)
        zin = zim*2 - 1
        zin = zin.unsqueeze(0).cuda(0).half()

        init_latent = self.net.autokl_encode(zin)
        
        self.sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)
        #strength=0.75
        assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
        t_enc = int(strength * ddim_steps)
        device = 'cuda:0'
        z_enc = self.sampler.stochastic_encode(init_latent, torch.tensor([t_enc]).to(device))
        #z_enc,_ = sampler.encode(init_latent.cuda(0).half(), c.cuda(0).half(), torch.tensor([t_enc]).to(sampler.model.model.diffusion_model.device))

        dummy = ''
        utx = self.net.clip_encode_text(dummy)
        utx = utx.cuda(0).half()
        
        dummy = torch.zeros((1,3,224,224)).cuda(0)
        uim = self.net.clip_encode_vision(dummy)
        uim = uim.cuda(0).half()
        
        z_enc = z_enc.cuda(0)

        h, w = 512,512
        shape = [n_samples, 4, h//8, w//8]

        
        
        #c[:,0] = u[:,0]
        #z_enc = z_enc.cuda(0).half()
        
        self.sampler.model.model.diffusion_model.device='cuda:0'
        self.sampler.model.model.diffusion_model.half().cuda(0)
        mixing = 0.4
        
        z = self.sampler.decode_dc(
            x_latent=z_enc,
            first_conditioning=[uim, cim],
            second_conditioning=[utx, ctx],
            t_start=t_enc,
            unconditional_guidance_scale=scale,
            xtype='image', 
            first_ctype='vision',
            second_ctype='prompt',
            mixed_ratio=textstrength)
        
        z = z.cuda(0).half()
        x = self.net.autokl_decode(z)
        
        x = torch.clamp((x+1.0)/2.0, min=0.0, max=1.0)
        x = [tvtrans.ToPILImage()(xi) for xi in x]
        
        if n_samples == 1:
            x = x[0]
        return x