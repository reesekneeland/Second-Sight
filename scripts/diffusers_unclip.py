import torch
import sys
from PIL import Image
sys.path.append('src')
from utils import *
from diffusers import StableUnCLIPImg2ImgPipeline, StableDiffusionImg2ImgPipeline

R = StableUnCLIPImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16, variation="fp16"
)
R = R.to("cuda:3")

# R.enable_model_cpu_offload()
R.enable_xformers_memory_efficient_attention()
# url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"

# response = requests.get(url)
init_image = Image.open("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/dog.png").convert("RGB")
init_image = init_image.resize((768, 768))

init_image2 = Image.open("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/pearl_earring.png").convert("RGB")
init_image2 = init_image2.resize((768, 768))

image_embeds = R.encode_image_raw(
        image=init_image,
        device="cuda:3")
print("TYPE: ", image_embeds.dtype)

image_embeds5 = R.encode_image_raw(
        image=init_image2,
        device="cuda:3")

image_embeds1 = slerp(image_embeds, image_embeds5, 0.2)
image_embeds2 = slerp(image_embeds, image_embeds5, 0.4)
image_embeds3 = slerp(image_embeds, image_embeds5, 0.6)
image_embeds4 = slerp(image_embeds, image_embeds5, 0.8)


# im1 = R.reconstruct(image_embeds=image_embeds, prompt=prompt, num_inference_steps=10).images
im1 = R.reconstruct(image_embeds=image_embeds, guidance_scale=10, noise_level=1)
im2 = R.reconstruct(image_embeds=image_embeds1, guidance_scale=10, noise_level=1)
im3 = R.reconstruct(image_embeds=image_embeds2, guidance_scale=10, noise_level=1)
im4 = R.reconstruct(image_embeds=image_embeds3, guidance_scale=10, noise_level=1)
im5 = R.reconstruct(image_embeds=image_embeds4, guidance_scale=10, noise_level=1)
im6 = R.reconstruct(image_embeds=image_embeds5, guidance_scale=10, noise_level=1)
# im1 = R.reconstruct(image=init_image, strength=0.7, prompt_embeds=prompt_embed, image_embeds=image_embeds, guidance_scale=10, noise_level=0)
# im2 = R.reconstruct(image=init_image, strength=0.7, prompt_embeds=prompt_embed, image_embeds=image_embeds, guidance_scale=10, noise_level=1)
# im3 = R.reconstruct(image=init_image, strength=0.7, prompt_embeds=prompt_embed, image_embeds=image_embeds, guidance_scale=10, noise_level=5)
# im4 = R.reconstruct(image=init_image, strength=0.7, prompt_embeds=prompt_embed, image_embeds=image_embeds, guidance_scale=10, noise_level=10)
# init_image.save("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/reconstructions/unCLIP test/init.png")
im1.save("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/reconstructions/subject1/unCLIP test/var_sinterp0.png")
im2.save("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/reconstructions/subject1/unCLIP test/var_sinterp1.png")
im3.save("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/reconstructions/subject1/unCLIP test/var_sinterp2.png")
im4.save("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/reconstructions/subject1/unCLIP test/var_sinterp3.png")
im5.save("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/reconstructions/subject1/unCLIP test/var_sinterp4.png")
im6.save("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/reconstructions/subject1/unCLIP test/var_sinterp5.png")