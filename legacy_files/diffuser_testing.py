import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
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

im = Image.open("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/laptop.png")
# w, h = 425, 425  # resize
# imagePil = im.resize((w, h), resample=Image.Resampling.LANCZOS)

# transform = transforms.Compose([
#     transforms.ToTensor()
# ])

# img_tr = transform(im)
# print(img_tr.shape)
# mean, std = img_tr.mean([1,2]), img_tr.std([1,2])
# print(mean, std)

# tform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Resize(
#         (224, 224),
#         interpolation=transforms.InterpolationMode.BICUBIC,
#         antialias=False,
#         ),
#     transforms.Normalize(
#       [0.48145466, 0.4578275, 0.40821073],
#       [0.26862954, 0.26130258, 0.27577711]),
# ])
# inp = tform(im).to(device).unsqueeze(0)

# image_embed = im_model.encode_image(im)
# text_embed = text_model.encode_prompt("a dog running on a path with a frisbee in its mouth, neutral, realistic, low contrast")
# image_embed = torch.load("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/dog_image_features.pt").to(device)
# text_embed = torch.load("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/dog_text_features.pt").to(device)
# img2img_test = img2img("a dog running on a path with a frisbee in its mouth", im, strength=1)[0][0]
# text_embed_img2img = torch.load("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/dog_text_features_img2img.pt").to(device)
# print(text_embed.shape)
# combined_embed = image_embed + text_embed
# torch.save(combined_embed, "/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/dog_combined_features.pt")
# combined_embed = torch.load("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/dog_combined_features.pt").to(device)
# print("combined embedding shape", combined_embed.shape)
# torch.save(image_embed, "/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/image_embed.pt")
# prompt_embed = text_model.encode_prompt("a dog catching a frisbee")
# print("prompt embedding shape", prompt_embed.shape)

# image_latent = text_model.encode_latents(im)
# print("image latent shape", image_latent.shape)

# z_only = im_model.encode(z=image_latent, strength=0.0)
# # c_only_prompt = im_model.encode(c=prompt_embed, strength=1.0)
# imvar_test = imvar_model(inp, guidance_scale=3)[0][0]
# noisy_latent = torch.load("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/test_samples/00004.pt")
# latent = torch.load("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/test_samples/00006.pt")
for i in range(0,20):
  c = torch.load("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/test_embeds/c_" + str(i) + ".pt")
  z = torch.load("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/test_embeds/z_" + str(i) + ".pt")
  cond_im = im_model.reconstruct(c=c, strength=1.0, guidance_scale=5)
  z_only = im_model.reconstruct(z=z, strength=0.0, guidance_scale=5)
  z_and_c = im_model.reconstruct(z=z, c=c, strength=0.75, guidance_scale=5)
# latent_im = im_model.reconstruct(z=latent, strength=0.0, guidance_scale=3)
# noisy_latent_im = im_model.reconstruct(z=noisy_latent, strength=0.0, guidance_scale=3)
#TEXT MODEL RECONSTRUCT RUNNING OUT OF MEMORY
# only_text_img2img = text_model("a dog running on a path with a frisbee in its mouth", im, image_embed, strength=1.0, guidance_scale=7.5)
# only_text_img2img = text_model.reconstruct(im, c=text_embed_img2img, strength=1.0, guidance_scale=7.5)
# only_text = im_model.reconstruct(c=text_embed, strength=1.0, guidance_scale=7.5)
# # only_text = text_model(c=text_embed, strength=1.0, guidance_scale=7.5)[0][0]
# combined = im_model.reconstruct(c=combined_embed, strength=1.0, guidance_scale=7.5)
# combined_text = text_model.reconstruct(im, c=combined_embed, strength=1.0, guidance_scale=7.5)
# img2img = img2img("a dog running on a path with a frisbee in its mouth", im, strength=1)[0][0]
# imvariation = imvar_model(im, guidance_scale=3)[0][0]
# image_and_z = im_model.encode(z=image_latent, c=image_embed, strength=0.7)
# print("z_only", z_only)
# z_only = np.uint8((z_only * 255).round())
# print("z_only", z_only.shape, z_only)
# z_only = Image.fromarray(z_only.reshape((512, 512, 3)))
# z_only.save("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/z_only.jpg")

# img2img.save("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/img2img_out.jpg")
# noisy_latent_im.save("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/test_samples/noisy_latent.png")
  cond_im.save("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/combined_reconstruct/" + str(i) + "_c_only.png")
  z_only.save("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/combined_reconstruct/" + str(i) + "_z_only.png")
  z_and_c.save("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/combined_reconstruct/" + str(i) + "_z_and_c.png")
# latent_im.save("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/latent.png")
# only_text.save("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/only_text.png")
# only_text_img2img.save("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/only_text_img2img.png")
# combined.save("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/combined.png")
# combined_text.save("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/combined_text.png")



