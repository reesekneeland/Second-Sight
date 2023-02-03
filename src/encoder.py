import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
import PIL
import torch
import numpy as np
from PIL import Image
from einops import rearrange
from torch import autocast
sys.path.append('/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/stablediffusion')
from ldm.extras import load_model_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from huggingface_hub import hf_hub_download
from transformers import CLIPProcessor, CLIPModel
import clip


class Encoder():
    def __init__(self):

        self.ckpt = hf_hub_download(repo_id="lambdalabs/image-mixer", filename="image-mixer-full.ckpt")
        self.config = hf_hub_download(repo_id="lambdalabs/image-mixer", filename="image-mixer-config.yaml")

        self.device = "cuda"
        self.model = load_model_from_config(self.config, self.ckpt, device=self.device, verbose=False)
        self.model = self.model.to(self.device).half()

        self.clip_model, self.preprocess = clip.load("ViT-L/14", device=self.device)
        self.sampler = DDIMSampler(self.model)
        self.scale = 5

        os.makedirs("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/test_samples", exist_ok=True)
        self.outpath = "/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/test_samples"
        self.base_count = len(os.listdir(self.outpath))+3

        self.init_clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.init_preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

        self.batch_size = 1
        
    @torch.no_grad()
    def get_im_c(self, im_path):
        # im = Image.open(im_path).convert("RGB")
        prompts = self.preprocess(im_path).to(self.device).unsqueeze(0)
        return self.clip_model.encode_image(prompts).float()

    @torch.no_grad()
    def get_txt_c(self, txt):
        text = clip.tokenize([txt,]).to(self.device)
        return self.clip_model.encode_text(text)

    def encode_image(self, image):
        conds_img = []
        c_img = self.get_im_c(image)
        conds_img.append(c_img)
        for j in range(0,4):
            conds_img.append((torch.zeros((1, 768), device=self.device)))
        conds_img = torch.cat(conds_img, dim=0).unsqueeze(0)
        conds_img = conds_img.tile(1, 1, 1)
        print("encode image shape: ", conds_img.shape)
        return conds_img
        
    def encode_combined(self, image, prompts):
        conds_combined = []
        c_img = self.get_im_c(image)
        conds_combined.append(c_img)
        inputs = self.init_preprocess(text=prompts, images=image, return_tensors="pt", padding=True)
        outputs = self.init_clip_model(**inputs)
        logits_per_image = outputs.logits_per_image # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1).tolist()[0] # we can take the softmax to get the label probabilities
        sorted_prompts = [x for _, x in sorted(zip(probs, prompts), reverse=True)]
        sorted_probs = sorted(probs, reverse=True)
        
        for j in range(0,4):
            high_prob = sorted_probs[0]
            if(sorted_probs[j] >= high_prob/2):
                c_text = 2*sorted_probs[j]*self.get_txt_c(sorted_prompts[j])
                conds_combined.append(c_text)
            else:
                conds_combined.append((torch.zeros((1, 768), device=self.device)))
        conds_combined = torch.cat(conds_combined, dim=0).unsqueeze(0)
        conds_combined = conds_combined.tile(1, 1, 1)
    
    def encode_latents(self, image):
        w, h = 512, 512  # resize to integer multiple of 64
        # imagePil = Image.fromarray(image)
        imagePil = image.resize((w, h), resample=Image.Resampling.LANCZOS)
        image = np.array(imagePil).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        image = 2. * image - 1.
        return self.model.get_first_stage_encoding(self.model.encode_first_stage(image.to(device=self.device, dtype=torch.float16)))
    
    def reconstruct(self, strength, z=None, c=None):
        if(strength==0.0):
            image_np = self.decode_latents(z.reshape((1,4,64,64)).to(torch.float32))
            image_np = (image_np * 255).round().astype("uint8")
            image = PIL.Image.fromarray(image_np.reshape((512, 512, 3)))
            img.save(os.path.join(self.outpath, f"{self.base_count:05}.png"))
            self.base_count += 1
            return image
        
        if(strength==1.0):
            init_latent = torch.randn((1,4,64,64), generator=None, device=self.device, dtype=torch.float32, layout=torch.strided).to(self.device)
        else:
            init_latent = z.reshape((1,4,64,64)).to(torch.float32)

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
                    # encode (scaled latent)
                    z_enc = self.sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*1).to(self.device))
                    # decode it
                    print(z_enc.shape, c.shape, t_enc, strength, uc.shape if uc is not None else None)
                    samples = self.sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=self.scale,
                                            unconditional_conditioning=uc,)

                    x_samples = self.model.decode_first_stage(samples)
                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                    x_sample = 255. * rearrange(x_samples[0].cpu().numpy(), 'c h w -> h w c')
                    img = Image.fromarray(x_sample.astype(np.uint8))
                    img.save(os.path.join(self.outpath, f"{self.base_count:05}.png"))
                    self.base_count += 1
        return img

    def main(self,):
        image = Image.open("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/surfer.png")
        # nsda = NSDAccess('/home/naxos2-raid25/kneel027/home/surly/raid4/kendrick-data/nsd', '/home/naxos2-raid25/kneel027/home/kneel027/nsd_local')
        # captions = nsda.read_image_coco_info([i for i in range(73000)], info_type='captions', show_annot=False)
        z = self.encode_latents(image)
        c = self.encode_image(image)
        # z = torch.load("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/test_embeds/z_0.pt").to(torch.float16)
        # c = torch.load("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/test_embeds/c_0.pt")
        output = self.reconstruct(strength=0.1, z=z, c=c)
        output.save("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/surfer_reconstruct.png")
        
if __name__ == "__main__":
    E = Encoder()
    E.main()
    