import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
import torch
import sys
from PIL import Image
sys.path.append('src')
from utils import *
import open_clip
from diffusers import UnCLIPScheduler, DDPMScheduler, StableUnCLIPPipeline
from diffusers.models import PriorTransformer
from transformers import CLIPTokenizer, CLIPTextModelWithProjection

prior_model_id = "kakaobrain/karlo-v1-alpha"
data_type = torch.float16
prior = PriorTransformer.from_pretrained(prior_model_id, subfolder="prior", torch_dtype=data_type)

# prior_text_model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', device="cuda", pretrained='laion2b_s32b_b79k')
# prior_tokenizer = open_clip.get_tokenizer('ViT-H-14')
prior_text_model_id = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
prior_tokenizer = CLIPTokenizer.from_pretrained(prior_text_model_id)
prior_text_model = CLIPTextModelWithProjection.from_pretrained(prior_text_model_id, torch_dtype=data_type)
prior_scheduler = UnCLIPScheduler.from_pretrained(prior_model_id, subfolder="prior_scheduler")
prior_scheduler = DDPMScheduler.from_config(prior_scheduler.config)

stable_unclip_model_id = "stabilityai/stable-diffusion-2-1-unclip-small"

pipe = StableUnCLIPPipeline.from_pretrained(
    stable_unclip_model_id,
    torch_dtype=data_type,
    variant="fp16",
    prior_tokenizer=prior_tokenizer,
    prior_text_encoder=prior_text_model,
    prior=prior,
    prior_scheduler=prior_scheduler,
).to("cuda")

wave_prompt = "dramatic wave, the Oceans roar, Strong wave spiral across the oceans as the waves unfurl into roaring crests; perfect wave form; perfect wave shape; dramatic wave shape; wave shape unbelievable; wave; wave shape spectacular"

images = pipe(prompt=wave_prompt).images
images[0].save("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/reconstructions/unCLIP test/waves_text.png")