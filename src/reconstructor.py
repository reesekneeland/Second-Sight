import os,sys
import PIL
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as tvtrans
sys.path.append(os.getcwd() + '/Versatile-Diffusion/')
from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model
n_sample_image = 1
n_sample_text = 1
cache_examples = True
from random import randint
from lib.model_zoo.ddim import DDIMSampler
      

def highlight_print(info):
    print('')
    print(''.join(['#']*(len(info)+4)))
    print('# '+info+' #')
    print(''.join(['#']*(len(info)+4)))
    print('')

def decompose(x, q=20, niter=100):
    x_mean = x.mean(-1, keepdim=True)
    x_input = x - x_mean
    u, s, v = torch.pca_lowrank(x_input, q=q, center=False, niter=niter)
    ss = torch.stack([torch.diag(si) for si in s])
    x_lowrank = torch.bmm(torch.bmm(u, ss), torch.permute(v, [0, 2, 1]))
    x_remain = x_input - x_lowrank
    return u, s, v, x_mean, x_remain

class adjust_rank(object):
    def __init__(self, max_drop_rank=[1, 5], q=20):
        self.max_semantic_drop_rank = max_drop_rank[0]
        self.max_style_drop_rank = max_drop_rank[1]
        self.q = q

        def t2y0_semf_wrapper(t0, y00, t1, y01):
            return lambda t: (np.exp((t-0.5)*2)-t0)/(t1-t0)*(y01-y00)+y00
        t0, y00 = np.exp((0  -0.5)*2), -self.max_semantic_drop_rank
        t1, y01 = np.exp((0.5-0.5)*2), 1
        self.t2y0_semf = t2y0_semf_wrapper(t0, y00, t1, y01)

        def x2y_semf_wrapper(x0, x1, y1):
            return lambda x, y0: (x-x0)/(x1-x0)*(y1-y0)+y0
        x0 = 0
        x1, y1 = self.max_semantic_drop_rank+1, 1
        self.x2y_semf = x2y_semf_wrapper(x0, x1, y1)
        
        def t2y0_styf_wrapper(t0, y00, t1, y01):
            return lambda t: (np.exp((t-0.5)*2)-t0)/(t1-t0)*(y01-y00)+y00
        t0, y00 = np.exp((1  -0.5)*2), -(q-self.max_style_drop_rank)
        t1, y01 = np.exp((0.5-0.5)*2), 1
        self.t2y0_styf = t2y0_styf_wrapper(t0, y00, t1, y01)

        def x2y_styf_wrapper(x0, x1, y1):
            return lambda x, y0: (x-x0)/(x1-x0)*(y1-y0)+y0
        x0 = q-1
        x1, y1 = self.max_style_drop_rank-1, 1
        self.x2y_styf = x2y_styf_wrapper(x0, x1, y1)

    def __call__(self, x, lvl):
        if lvl == 0.5:
            return x

        if x.dtype == torch.float16:
            fp16 = True
            x = x.float()
        else:
            fp16 = False
        std_save = x.std(axis=[-2, -1])

        u, s, v, x_mean, x_remain = decompose(x, q=self.q)

        if lvl < 0.5:
            assert lvl>=0
            for xi in range(0, self.max_semantic_drop_rank+1):
                y0 = self.t2y0_semf(lvl)
                yi = self.x2y_semf(xi, y0)
                yi = 0 if yi<0 else yi
                s[:, xi] *= yi

        elif lvl > 0.5:
            assert lvl <= 1
            for xi in range(self.max_style_drop_rank, self.q):
                y0 = self.t2y0_styf(lvl)
                yi = self.x2y_styf(xi, y0)
                yi = 0 if yi<0 else yi
                s[:, xi] *= yi
            x_remain = 0

        ss = torch.stack([torch.diag(si) for si in s])
        x_lowrank = torch.bmm(torch.bmm(u, ss), torch.permute(v, [0, 2, 1]))
        x_new = x_lowrank + x_mean + x_remain

        std_new = x_new.std(axis=[-2, -1])
        x_new = x_new / std_new * std_save

        if fp16:
            x_new = x_new.half()

        return x_new

class Reconstructor(object):
    def __init__(self, fp16=True, which='v1.0', device="cuda:0"):
        self.which = which
        os.chdir("Versatile-Diffusion/")
        if self.which == 'v1.0':
            cfgm = model_cfg_bank()('vd_four_flow_v1-0')
        else:
            assert False, 'Model type not supported'
        net = get_model()(cfgm)

        if fp16:
            if self.which == 'v1.0':
                net.ctx['text'].fp16 = True
                net.ctx['image'].fp16 = True
            net = net.half()
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        if self.which == 'v1.0':
            if fp16:
                sd = torch.load('pretrained/vd-four-flow-v1-0-fp16.pth', map_location='cpu')
            else:
                sd = torch.load('pretrained/vd-four-flow-v1-0.pth', map_location='cpu')
            # from huggingface_hub import hf_hub_download
            # if fp16:
            #     temppath = hf_hub_download('shi-labs/versatile-diffusion-model', 'pretrained_pth/vd-four-flow-v1-0-fp16.pth')
            # else:
            #     temppath = hf_hub_download('shi-labs/versatile-diffusion-model', 'pretrained_pth/vd-four-flow-v1-0.pth')
            # sd = torch.load(temppath, map_location='cpu')

        net.load_state_dict(sd, strict=False)

        self.device=device
        net.to(self.device)
        self.net = net
        self.sampler = DDIMSampler(net)

        self.output_dim = [512, 512]
        self.n_sample_image = n_sample_image
        self.n_sample_text = n_sample_text
        self.ddim_steps = 50
        self.ddim_eta = 0.0
        self.scale_textto = 7.5
        self.image_latent_dim = 4
        self.text_latent_dim = 768
        self.text_temperature = 1

        if which == 'v1.0':
            self.adjust_rank_f = adjust_rank(max_drop_rank=[1, 5], q=20)
            self.scale_imgto = 7.5
            self.disentanglement_noglobal = True
        os.chdir("../")
    def encode_text(self, prompt):
        text_encoding = self.net.ctx_encode([prompt], which='text')
        return text_encoding
    
    def encode_image(self, image):
        BICUBIC = PIL.Image.Resampling.BICUBIC
        cx = image.resize([512, 512], resample=BICUBIC)
        cx = tvtrans.ToTensor()(cx)[None].to(self.device).to(self.dtype)
        image_encoding = self.net.ctx_encode(cx, which='image')
        return image_encoding
    
    def reconstruct(self, 
                    image=None, 
                    c_i=None, 
                    c_t=None, 
                    n_samples=1, 
                    textstrength=0.45, 
                    strength=0.8, 
                    color_adjust=True,
                    fcs_lvl=0.5, 
                    seed=None
                    ):
        
        h, w = 512, 512
        BICUBIC = PIL.Image.Resampling.BICUBIC
        
        if strength == 0:
            return [image]*n_samples
        else:
            c_i = c_i.reshape((257,768)).to(dtype=torch.float16, device=self.device)
            c_t = c_t.reshape((77,768)).to(dtype=torch.float16, device=self.device)
        if(image):
            image = image.resize([w, h], resample=BICUBIC)
            image_tensor = tvtrans.ToTensor()(image)[None].to(self.device).to(self.dtype)
        else:
            color_adjust=False
            
        ut = self.net.ctx_encode([""], which='text').repeat(n_samples, 1, 1)
        ct = c_t.repeat(n_samples, 1, 1)
        scale = self.scale_imgto*(1-textstrength) + self.scale_textto*textstrength
        
        c_info_list = []
        c_info_list.append({
            'type':'text', 
            'conditioning':ct.to(torch.float16), 
            'unconditional_conditioning':ut,
            'unconditional_guidance_scale':scale,
            'ratio': textstrength, })
        ci = c_i

        if self.disentanglement_noglobal:
            ci_glb = ci[:, 0:1]
            ci_loc = ci[:, 1: ]
            ci_loc = self.adjust_rank_f(ci_loc, fcs_lvl)
            ci = torch.cat([ci_glb, ci_loc], dim=1).repeat(n_samples, 1, 1)
        else:
            ci = self.adjust_rank_f(ci, fcs_lvl).repeat(n_samples, 1, 1)

        c_info_list.append({
            'type':'image', 
            'conditioning':ci.to(torch.float16), 
            'unconditional_conditioning':torch.zeros_like(ci),
            'unconditional_guidance_scale':scale,
            'ratio': (1-textstrength), })

        shape = [n_samples, self.image_latent_dim, h//8, w//8]
        if(seed):
            np.random.seed(seed)
            torch.manual_seed(seed + 100)
        else:
            seed = randint(0,1000)
            np.random.seed(seed)
            torch.manual_seed(seed + 100)
        if strength!=1:
            x0 = self.net.vae_encode(image_tensor, which='image').repeat(n_samples, 1, 1, 1)
            step = int(self.ddim_steps * (strength))
            x, _ = self.sampler.sample_multicontext(
                steps=self.ddim_steps,
                x_info={'type':'image', 'x0':x0, 'x0_forward_timesteps':step},
                c_info_list=c_info_list,
                shape=shape,
                verbose=False,
                eta=self.ddim_eta)
        else:
            x, _ = self.sampler.sample_multicontext(
                steps=self.ddim_steps,
                x_info={'type':'image',},
                c_info_list=c_info_list,
                shape=shape,
                verbose=False,
                eta=self.ddim_eta)

        imout = self.net.vae_decode(x, which='image')
        if color_adjust:
            cx_mean = image_tensor.view(3, -1).mean(-1)[:, None, None]
            cx_std  = image_tensor.view(3, -1).std(-1)[:, None, None]
            imout_mean = [imouti.view(3, -1).mean(-1)[:, None, None] for imouti in imout]
            imout_std  = [imouti.view(3, -1).std(-1)[:, None, None] for imouti in imout]
            imout = [(ii-mi)/si*cx_std+cx_mean for ii, mi, si in zip(imout, imout_mean, imout_std)]
            imout = [torch.clamp(ii, 0, 1) for ii in imout]
        imout = [tvtrans.ToPILImage()(i) for i in imout]
        if len(imout)==1:
            return imout[0]
        else:
            return imout

def main():
    R = Reconstructor(which='v1.0', fp16=True, device="cuda:2")
    im1 = Image.open("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/dog.png")
    # text1 = "A dog running along a path with a yellow frisbee in its mouth"
    c_i = R.encode_image(im1)
    print(c_i.dtype)
    # dim_max = torch.max(c_i, dim=0)
    # print(len(dim_max.values), dim_max.values)
    # c_t = R.encode_text(text1)
    # output = R.reconstruct(image=im1, 
    #                         c_i=c_i, 
    #                         c_t=c_t, 
    #                         n_samples=1, 
    #                         strength=0.8)
    # output2 = R.reconstruct(image=im1, 
    #                         c_i=c_i, 
    #                         c_t=c_t, 
    #                         n_samples=1, 
    #                         textstrength=0.45, 
    #                         strength=1, 
    #                         color_adjust=True,
    #                         fcs_lvl=0.1)
    # output3 = R.reconstruct(image=im1, 
    #                         c_i=c_i, 
    #                         c_t=c_t, 
    #                         n_samples=1, 
    #                         textstrength=0.45, 
    #                         strength=1, 
    #                         color_adjust=True,
    #                         fcs_lvl=0.2)
    # output4 = R.reconstruct(image=im1, 
    #                         c_i=c_i, 
    #                         c_t=c_t, 
    #                         n_samples=1, 
    #                         textstrength=0.45, 
    #                         strength=1, 
    #                         color_adjust=True,
    #                         fcs_lvl=0.3)
    # output5 = R.reconstruct(image=im1, 
    #                         c_i=c_i, 
    #                         c_t=c_t, 
    #                         n_samples=1, 
    #                         textstrength=0.45, 
    #                         strength=1, 
    #                         color_adjust=True,
    #                         fcs_lvl=0.4)
    # output.save("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/dog_vd.png")
    # output2.save("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/dog_vd2.png")
    # output3.save("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/dog_vd3.png")
    # output4.save("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/dog_vd4.png")
    # output5.save("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/dog_vd5.png")
    
if __name__ == "__main__":
    main()