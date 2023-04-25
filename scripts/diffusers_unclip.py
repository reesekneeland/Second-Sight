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
# url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"

# response = requests.get(url)
init_image = Image.open("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/surfer2.png").convert("RGB")
init_image = init_image.resize((768, 768))


image_embeds = R.encode_image_raw(
        image=init_image,
        device="cuda:3")
print("TYPE: ", image_embeds.dtype)



# im1 = R.reconstruct(image_embeds=image_embeds, prompt=prompt, num_inference_steps=10).images
im9 = R.reconstruct(image = init_image, image_embeds=image_embeds, strength=1)
im8 = R.reconstruct(image = init_image, image_embeds=image_embeds, strength=1)
im7 = R.reconstruct(image = init_image, image_embeds=image_embeds, strength=1)
im1 = R.reconstruct(image = init_image, image_embeds=image_embeds, strength=0.75)
im2 = R.reconstruct(image = init_image, image_embeds=image_embeds, strength=0.75)
im3 = R.reconstruct(image = init_image, image_embeds=image_embeds, strength=0.75)
im4 = R.reconstruct(image = init_image, image_embeds=image_embeds, strength=0.5)
im5 = R.reconstruct(image = init_image, image_embeds=image_embeds, strength=0.5)
im6 = R.reconstruct(image = init_image, image_embeds=image_embeds, strength=0.5)
images = [im9, im1, im4, im8, im2, im5, im7, im3, im6]
captions = ["Strength: 1", "Strength: 0.75","Strength: 0.5"]
output = tileImages("", images, captions, 3, 3, useTitle=2, rowCaptions=False)
output.save("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/slide_examples/tiled.png")
# im1 = R.reconstruct(image=init_image, strength=0.7, prompt_embeds=prompt_embed, image_embeds=image_embeds, guidance_scale=10, noise_level=0)
# im2 = R.reconstruct(image=init_image, strength=0.7, prompt_embeds=prompt_embed, image_embeds=image_embeds, guidance_scale=10, noise_level=1)
# im3 = R.reconstruct(image=init_image, strength=0.7, prompt_embeds=prompt_embed, image_embeds=image_embeds, guidance_scale=10, noise_level=5)
# im4 = R.reconstruct(image=init_image, strength=0.7, prompt_embeds=prompt_embed, image_embeds=image_embeds, guidance_scale=10, noise_level=10)
# init_image.save("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/reconstructions/unCLIP test/init.png")
# im1.save("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/slide_examples/75_0.png")
# im2.save("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/slide_examples/75_1.png")
# im3.save("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/slide_examples/75_2.png")
# im4.save("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/slide_examples/50_0.png")
# im5.save("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/slide_examples/50_1.png")
# im6.save("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/slide_examples/50_2.png")