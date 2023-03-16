import ml_collections
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
sys.path.insert(0, os.getcwd() + '/unidiffuser/')
import torch
import random
import utils
from dpm_solver_pp import NoiseScheduleVP, DPM_Solver
from absl import logging
import einops
import libs.autoencoder
import libs.clip
from torchvision.utils import save_image, make_grid
import torchvision.transforms as standard_transforms
import numpy as np
import clip
from PIL import Image
import time
from absl import flags
from absl import app
from ml_collections import config_flags

from libs.caption_decoder import CaptionDecoder

def stable_diffusion_beta_schedule(linear_start=0.00085, linear_end=0.0120, n_timestep=1000):
        _betas = (
            torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )
        return _betas.numpy()

def unpreprocess(v):  # to B C H W and [0, 1]
    v = 0.5 * (v + 1.)
    v.clamp_(0., 1.)
    return v

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def split(x):
        C, H, W = 4, 64, 64
        z_dim = C * H * W
        z, clip_img = x.split([z_dim, 512], dim=1)
        z = einops.rearrange(z, 'B (C H W) -> B C H W', C=C, H=H, W=W)
        clip_img = einops.rearrange(clip_img, 'B (L D) -> B L D', L=1, D=512)
        return z, clip_img


def combine(z, clip_img):
    z = einops.rearrange(z, 'B C H W -> B (C H W)')
    clip_img = einops.rearrange(clip_img, 'B L D -> B (L D)')
    return torch.concat([z, clip_img], dim=-1)

def split_joint(x):
        C, H, W = 4, 64, 64
        z_dim = C * H * W
        z, clip_img, text = x.split([z_dim, 512, 77 * 64], dim=1)
        z = einops.rearrange(z, 'B (C H W) -> B C H W', C=C, H=H, W=W)
        clip_img = einops.rearrange(clip_img, 'B (L D) -> B L D', L=1, D=512)
        text = einops.rearrange(text, 'B (L D) -> B L D', L=77, D=64)
        return z, clip_img, text

def combine_joint(z, clip_img, text):
    z = einops.rearrange(z, 'B C H W -> B (C H W)')
    clip_img = einops.rearrange(clip_img, 'B L D -> B (L D)')
    text = einops.rearrange(text, 'B L D -> B (L D)')
    return torch.concat([z, clip_img, text], dim=-1)

def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)



class UD_Reconstructor():
    
    def __init__(self, device="cuda:0"):
        os.chdir("unidiffuser/")
        self.config = ml_collections.ConfigDict()

        # self.config.seed = 1234
        self.config.pred = 'noise_pred'
        self.config.z_shape = (4, 64, 64)
        self.config.clip_img_dim = 512
        self.config.clip_text_dim = 768
        self.config.text_dim = 64  # reduce dimension
        self.config.data_type = 1

        self.config.autoencoder = d(
            pretrained_path='models/autoencoder_kl.pth',
        )

        self.config.caption_decoder = d(
            pretrained_path="models/caption_decoder.pth",
            hidden_dim=self.config.get_ref('text_dim')
        )

        self.config.nnet = d(
            name='uvit_multi_post_ln_v1',
            img_size=64,
            in_chans=4,
            patch_size=2,
            embed_dim=1536,
            depth=30,
            num_heads=24,
            mlp_ratio=4,
            qkv_bias=False,
            pos_drop_rate=0.,
            drop_rate=0.,
            attn_drop_rate=0.,
            mlp_time_embed=False,
            text_dim=self.config.get_ref('text_dim'),
            num_text_tokens=77,
            clip_img_dim=self.config.get_ref('clip_img_dim'),
            use_checkpoint=True
        )

        self.config.sample = d(
            sample_steps=50,
            scale=7.,
            t2i_cfg_mode='true_uncond'
        )
        self.config.mode = 'ti2i'
        self.config.nnet_path = "models/uvit_v1.pth"
        self.config.output_path = "out"
        
        self.device = device

        self.config = ml_collections.FrozenConfigDict(self.config)

        self._betas = stable_diffusion_beta_schedule()
        self.N = len(self._betas)

        self.nnet = utils.get_nnet(**self.config.nnet)
        self.nnet.load_state_dict(torch.load(self.config.nnet_path, map_location='cpu'))
        self.nnet.to(device)
        self.nnet.eval()

        self.caption_decoder = CaptionDecoder(device=device, **self.config.caption_decoder)

        self.clip_text_model = libs.clip.FrozenCLIPEmbedder(device=device)
        self.clip_text_model.eval()
        self.clip_text_model.to(self.device)

        self.autoencoder = libs.autoencoder.get_model(**self.config.autoencoder)
        self.autoencoder.to(self.device)

        self.clip_img_model, self.clip_img_model_preprocess = clip.load("ViT-B/32", device=device, jit=False)

        self.empty_context = self.clip_text_model.encode([''])[0]

    def t2i_nnet(self, x, timesteps, text):  # text is the low dimension version of the text clip embedding
        """
        1. calculate the conditional model output
        2. calculate unconditional model output
            config.sample.t2i_cfg_mode == 'empty_token': using the original cfg with the empty string
            config.sample.t2i_cfg_mode == 'true_uncond: using the unconditional model learned by our method
        3. return linear combination of conditional output and unconditional output
        """
        z, clip_img = split(x)

        t_text = torch.zeros(timesteps.size(0), dtype=torch.int, device=self.device)

        z_out, clip_img_out, text_out = self.nnet(z, clip_img, text=text, t_img=timesteps, t_text=t_text,
                                             data_type=torch.zeros_like(t_text, device=self.device, dtype=torch.int) + self.config.data_type)
        x_out = combine(z_out, clip_img_out)

        if self.config.sample.scale == 0.:
            return x_out

        if self.config.sample.t2i_cfg_mode == 'empty_token':
            _empty_context = einops.repeat(self.empty_context, 'L D -> B L D', B=x.size(0))
            _empty_context = self.caption_decoder.encode_prefix(_empty_context)
            z_out_uncond, clip_img_out_uncond, text_out_uncond = self.nnet(z, clip_img, text=_empty_context, t_img=timesteps, t_text=t_text,
                                                                      data_type=torch.zeros_like(t_text, device=self.device, dtype=torch.int) + self.config.data_type)
            x_out_uncond = combine(z_out_uncond, clip_img_out_uncond)
        elif self.config.sample.t2i_cfg_mode == 'true_uncond':
            text_N = torch.randn_like(text)  # 3 other possible choices
            z_out_uncond, clip_img_out_uncond, text_out_uncond = self.nnet(z, clip_img, text=text_N, t_img=timesteps, t_text=torch.ones_like(timesteps) * self.N,
                                                                      data_type=torch.zeros_like(t_text, device=self.device, dtype=torch.int) + self.config.data_type)
            x_out_uncond = combine(z_out_uncond, clip_img_out_uncond)
        else:
            raise NotImplementedError

        return x_out + self.config.sample.scale * (x_out - x_out_uncond)
    
    def joint_nnet(self, x, timesteps):
        z, clip_img, text = split_joint(x)
        z_out, clip_img_out, text_out = self.nnet(z, clip_img, text=text, t_img=timesteps, t_text=timesteps,
                                             data_type=torch.zeros_like(timesteps, device=self.device, dtype=torch.int) + self.config.data_type)
        x_out = combine_joint(z_out, clip_img_out, text_out)

        if self.config.sample.scale == 0.:
            return x_out

        z_noise = torch.randn(x.size(0), *self.config.z_shape, device=self.device)
        clip_img_noise = torch.randn(x.size(0), 1, self.config.clip_img_dim, device=self.device)
        text_noise = torch.randn(x.size(0), 77, self.config.text_dim, device=self.device)

        _, _, text_out_uncond = self.nnet(z_noise, clip_img_noise, text=text, t_img=torch.ones_like(timesteps) * self.N, t_text=timesteps,
                                     data_type=torch.zeros_like(timesteps, device=self.device, dtype=torch.int) + self.config.data_type)
        z_out_uncond, clip_img_out_uncond, _ = self.nnet(z, clip_img, text=text_noise, t_img=timesteps, t_text=torch.ones_like(timesteps) * self.N,
                                                    data_type=torch.zeros_like(timesteps, device=self.device, dtype=torch.int) + self.config.data_type)

        x_out_uncond = combine_joint(z_out_uncond, clip_img_out_uncond, text_out_uncond)

        return x_out + self.config.sample.scale * (x_out - x_out_uncond)

    def i2t_nnet(self, x, timesteps, z, clip_img):
        """
        1. calculate the conditional model output
        2. calculate unconditional model output
        3. return linear combination of conditional output and unconditional output
        """
        t_img = torch.zeros(timesteps.size(0), dtype=torch.int, device=self.device)

        z_out, clip_img_out, text_out = self.nnet(z, clip_img, text=x, t_img=t_img, t_text=timesteps,
                                             data_type=torch.zeros_like(t_img, device=self.device, dtype=torch.int) + self.config.data_type)

        if self.config.sample.scale == 0.:
            return text_out

        z_N = torch.randn_like(z)  # 3 other possible choices
        clip_img_N = torch.randn_like(clip_img)
        z_out_uncond, clip_img_out_uncond, text_out_uncond = self.nnet(z_N, clip_img_N, text=x, t_img=torch.ones_like(timesteps) * self.N, t_text=timesteps,
                                                                  data_type=torch.zeros_like(timesteps, device=self.device, dtype=torch.int) + self.config.data_type)

        return text_out + self.config.sample.scale * (text_out - text_out_uncond)
    
    def ti2i_nnet(self, x, timesteps, clip_img, text):
        z, _, text = split_joint(x)
        # z = x
        # text=x
        t_img = torch.zeros(timesteps.size(0), dtype=torch.int, device=self.device)
        t_text = torch.zeros(timesteps.size(0), dtype=torch.int, device=self.device)

        z_out, clip_img_out, text_out = self.nnet(z, clip_img=clip_img, text=text, t_img=t_img, t_text=timesteps,
                                             data_type=torch.zeros_like(t_img, device=self.device, dtype=torch.int) + self.config.data_type)
        x_out = combine_joint(z_out, clip_img_out, text_out)

        if self.config.sample.scale == 0.:
            return x_out
        # if self.config.sample.scale == 0.:
            # return z_out

        z_N = torch.randn_like(z)  # 3 other possible choices
        clip_img_N = torch.randn_like(clip_img)
        text_N = torch.randn_like(text)
        z_out_uncond, clip_img_out_uncond, text_out_uncond = self.nnet(z, clip_img=clip_img_N, text=text, t_img=torch.ones_like(timesteps) * self.N, t_text=timesteps,
                                                                  data_type=torch.zeros_like(t_img, device=self.device, dtype=torch.int) + self.config.data_type)
        # x_out_uncond = combine_joint(z_out_uncond, clip_img_out_uncond, text_out_uncond)
        # return z_out + self.config.sample.scale * (z_out - z_out_uncond)
        x_out_uncond = combine_joint(z_out_uncond, clip_img_out_uncond, text_out_uncond)

        return x_out + self.config.sample.scale * (x_out - x_out_uncond)
    
    # def ti2i_nnet(self, x, timesteps, z, clip_img, text):
    #     """
    #     1. calculate the conditional model output
    #     2. calculate unconditional model output
    #     3. return linear combination of conditional output and unconditional output
    #     """
    #     # _, _, text = split_joint(x)
    #     text = x
    #     t_text = torch.zeros(timesteps.size(0), dtype=torch.int, device=self.device)
    #     t_img = torch.zeros(timesteps.size(0), dtype=torch.int, device=self.device)

    #     z_out, clip_img_out, text_out = self.nnet(z, clip_img, text=text, t_img=t_img, t_text=timesteps,
    #                                          data_type=torch.zeros_like(t_img, device=self.device, dtype=torch.int) + self.config.data_type)
    #     # x_out = combine_joint(z_out, clip_img_out, text_out)
    #     x_out = text_out
    #     if self.config.sample.scale == 0.:
    #         return x_out

    #     if self.config.sample.t2i_cfg_mode == 'empty_token':
    #         _empty_context = einops.repeat(self.empty_context, 'L D -> B L D', B=x.size(0))
    #         _empty_context = self.caption_decoder.encode_prefix(_empty_context)
    #         z_out_uncond, clip_img_out_uncond, text_out_uncond = self.nnet(z, clip_img, text=_empty_context, t_img=t_img, t_text=timesteps,
    #                                                                   data_type=torch.zeros_like(t_img, device=self.device, dtype=torch.int) + self.config.data_type)
    #         x_out_uncond = combine_joint(z_out_uncond, clip_img_out_uncond, text_out_uncond)
    #     elif self.config.sample.t2i_cfg_mode == 'true_uncond':
    #         text_N = torch.randn_like(text)  # 3 other possible choices
    #         z_out_uncond, clip_img_out_uncond, text_out_uncond = self.nnet(z, clip_img, text=text_N, t_img=timesteps, t_text=torch.ones_like(timesteps) * self.N,
    #                                                                   data_type=torch.zeros_like(t_img, device=self.device, dtype=torch.int) + self.config.data_type)
    #         # x_out_uncond = combine_joint(z_out_uncond, clip_img_out_uncond, text_out_uncond)
    #         x_out_uncond = text_out_uncond
    #     else:
    #         raise NotImplementedError

    #     return x_out + self.config.sample.scale * (x_out - x_out_uncond)

    @torch.cuda.amp.autocast()
    def encode(self, _batch):
        return self.autoencoder.encode(_batch)

    @torch.cuda.amp.autocast()
    def decode(self, _batch):
        return self.autoencoder.decode(_batch)
    
    def encode_text(self, prompt):
        contexts = self.clip_text_model.encode(prompt)
        contexts_low_dim = self.caption_decoder.encode_prefix(contexts)
        return contexts_low_dim
    
    def encode_image(self, image):
        resolution = 512
        image = np.array(image).astype(np.uint8)
        image = utils.center_crop(resolution, resolution, image)
        clip_img_feature = self.clip_img_model.encode_image(self.clip_img_model_preprocess(Image.fromarray(image)).unsqueeze(0).to(self.device))
        return clip_img_feature[None, :, :]
    
    def encode_latents(self, image):
        image = np.array(image).astype(np.uint8)
        image = utils.center_crop(512, 512, image)
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = einops.rearrange(image, 'h w c -> 1 c h w')
        image = torch.tensor(image, device=self.device)
        moments = self.autoencoder.encode_moments(image)
        z_img = self.autoencoder.sample(moments)
        return z_img
    
    # def sample_fn(self, mode, n_samples=1, z_init=None, **kwargs):
    #     # if z_init is not None:
    #     #     _z_init = z_init
    #     # else:
    #     _z_init = torch.randn(n_samples, *self.config.z_shape, device=self.device)
    #     _clip_img_init = torch.randn(n_samples, 1, self.config.clip_img_dim, device=self.device)
    #     _text_init = torch.randn(n_samples, 77, self.config.text_dim, device=self.device)
    #     if mode in ['joint', 'ti2i']:
    #         _x_init = combine_joint(_z_init, _clip_img_init, _text_init)
    #     elif mode in ['t2i', 'i']:
    #         _x_init = combine(_z_init, _clip_img_init)
    #     elif mode in ['i2t', 't']:
    #         _x_init = _text_init
    #     noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.tensor(self._betas, device=self.device).float())

    #     def model_fn(x, t_continuous):
    #         t = t_continuous * self.N
    #         if mode in ['joint', 'ti2i']:
    #             return self.joint_nnet(x, t)
    #         elif mode == 't2i':
    #             return self.t2i_nnet(x, t, **kwargs)

    #     dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)
    #     with torch.no_grad():
    #         with torch.autocast('cuda'):
    #             start_time = time.time()
    #             print(_x_init.shape)
    #             x = dpm_solver.sample(_x_init, steps=self.config.sample.sample_steps, eps=1. / self.N, T=1.)
    #             print(x.shape)
    #             end_time = time.time()
    #             print(f'\ngenerate {n_samples} samples with {self.config.sample.sample_steps} steps takes {end_time - start_time:.2f}s')

    #     os.makedirs(self.config.output_path, exist_ok=True)
    #     if mode in ['joint', 'ti2i']:
    #         _z, _clip_img, _text = split_joint(x)
    #         return _z, _clip_img, _text
    #     elif mode in ['t2i', 'i']:
    #         _z, _clip_img = split(x)
    #         return _z, _clip_img
    #     elif mode in ['i2t', 't']:
    #         return x

    def sample_fn(self, mode, z_i=None, **kwargs):
        if(z_i is None):
            _z_init = torch.randn(1, *self.config.z_shape, device=self.device)
        else:
            _z_init = z_i
        _clip_img_init = torch.randn(1, 1, self.config.clip_img_dim, device=self.device)
        _text_init = torch.randn(1, 77, self.config.text_dim, device=self.device)
        
        print("MODE: ", mode)
        if mode in ['joint']:
            # _x_init = _text_init
            _x_init = combine_joint(_z_init, _clip_img_init, _text_init)
        elif mode in ['t2i', 'i']:
            _x_init = combine(_z_init, _clip_img_init)
        elif mode in ['i2t', 't']:
            _x_init = _text_init
        elif mode ==  'ti2i':
            # _x_init = _z_init
            _x_init = combine_joint(_z_init, _clip_img_init, _text_init)
        print("_X_INIT SHAPE: ", _x_init.shape)
        noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.tensor(self._betas, device=self.device).float())

        def model_fn(x, t_continuous):
            t = t_continuous * self.N
            if mode == 'joint':
                return self.joint_nnet(x, t)
            elif mode == 't2i':
                return self.t2i_nnet(x, t, **kwargs)
            elif mode == 'i2t':
                return self.i2t_nnet(x, t, **kwargs)
            elif mode == 'ti2i':
                return self.ti2i_nnet(x, t, **kwargs)

        dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)
        with torch.no_grad():
            with torch.autocast(device_type='cuda'):
                start_time = time.time()
                x = dpm_solver.sample(_x_init, steps=50, eps=1. / self.N, T=1.)
                end_time = time.time()
                print(f'\ngeneration took {end_time - start_time:.2f}s')

        os.makedirs(self.config.output_path, exist_ok=True)
        if mode in ['joint']:
            _z, _clip_img, _text = split_joint(x)
            return _z, _clip_img, _text
            # return x
        elif mode in ['t2i', 'i']:
            _z, _clip_img = split(x)
            return _z, _clip_img
        elif mode in ['i2t', 't']:
            return x
        elif mode == 'ti2i':
            _z, _clip_img, _text = split_joint(x)
            return _z
        # elif mode == 'ti2i':
        #     _z, _clip_img = split_joint(x)
        #     return _z, _clip_img

    def reconstruct(self, z=None, c_i=None, c_t=None, n_samples=1, strength=1):
        _z  = self.sample_fn('ti2i', z_i=None, clip_img=c_i, text=c_t)
        # _z, _clip_img, _text = self.sample_fn('ti2i', z=z, clip_img=c_i, text=c_t)  # conditioned on the image and text embedding
        # _text = self.sample_fn('i2t', z=z, clip_img=c_i)
        # samples = self.caption_decoder.generate_captions(_text)
        # print(samples)
        # _z, _clip_img = self.sample_fn('t2i', text=_text)
        # _z, _clip_img, _text = self.sample_fn(self.config.mode, z_init=z, n_samples=n_samples, text=c_t, clip_img=c_i)
        # print("SHAPES: ", _z.shape, _clip_img.shape)
        samples = unpreprocess(self.decode(_z))
        os.makedirs(os.path.join(self.config.output_path, self.config.mode), exist_ok=True)
        outputs = []
        for idx, sample in enumerate(samples):
            save_path = os.path.join(self.config.output_path, self.config.mode, f'{idx}.png')
            grid = make_grid(sample)
            ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            im = Image.fromarray(ndarr)
            im.save(save_path)
            outputs.append(im)
        if len(outputs)==1:
            return outputs[0]
        else: 
            return outputs


    
def main():
    UR = UD_Reconstructor()
    im1 = Image.open("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/dog.png")
    text1 = "realistic"
    c_i = UR.encode_image(im1)
    c_t = UR.encode_text(text1)
    latents = UR.encode_latents(im1)
    print(c_i.shape, c_t.shape, latents.shape)
    output = UR.reconstruct(z=latents, 
                            c_i=c_i,
                            c_t=c_t)
    output.save("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/dog_ud.png")
    
if __name__ == "__main__":
    main()