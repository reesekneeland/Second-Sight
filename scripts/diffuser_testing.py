import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
from diffusers import StableDiffusionImageEncodingPipeline, StableDiffusionTextEncodingPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionImageVariationPipeline
from PIL import Image
import torch
from torchvision import transforms
import numpy as np


device = "cuda"
#home to encode_image() and encode()
im_model = StableDiffusionImageEncodingPipeline.from_pretrained(
  "lambdalabs/sd-image-variations-diffusers",
  revision="v2.0",
  )
im_model = im_model.to(device)
#home to encode_prompt() and encode_latents()
text_model = StableDiffusionTextEncodingPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to(device)
text_model = text_model.to(device)

imvar_model = StableDiffusionImageVariationPipeline.from_pretrained("lambdalabs/sd-image-variations-diffusers", revision="v2.0").to(device)

img2img = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to(device)

im = Image.open("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/00001.png")
tform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(
                (224, 224),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=False,
                ),
            transforms.Normalize(
            [0.48145466, 0.4578275, 0.40821073],
            [0.26862954, 0.26130258, 0.27577711]),
        ])
image = tform(im).to(device).unsqueeze(0)

image_embed = im_model.encode_image(im)
print("image embedding shape", image_embed.shape)
# torch.save(image_embed, "/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/image_embed.pt")
# prompt_embed = text_model.encode_prompt("a dog catching a frisbee")
# print("prompt embedding shape", prompt_embed.shape)

# image_latent = text_model.encode_latents(im)
# print("image latent shape", image_latent.shape)

# out = im_model(inp, guidance_scale=3)
# img2 = img2img("a dog catching a frisbee in a green field", im, strength=1.0)
im2 = imvar_model(image, guidance_scale=4.5)[0][0]
# torch.save(image_embed, "/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/image_embed.pt")

# z_only = im_model.encode(z=image_latent, strength=0.0)
# # c_only_prompt = im_model.encode(c=prompt_embed, strength=1.0)
c_only_image = im_model.encode(c=image_embed, strength=1.0, guidance_scale=4.5)
# image_and_z = im_model.encode(z=image_latent, c=image_embed, strength=0.7)
# print("z_only", z_only)
# z_only = np.uint8((z_only * 255).round())
# print("z_only", z_only.shape, z_only)
# z_only = Image.fromarray(z_only.reshape((512, 512, 3)))
# z_only.save("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/z_only.jpg")
c_only_image.save("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/c_only_image.jpg")
# c_only_prompt.save("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/c_only_prompt.jpg")
# img2.save("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/img2img_only_prompt.jpg")
im2.save("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/imgvar.jpg")
# image_and_z.save("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/image_and_z.jpg")




