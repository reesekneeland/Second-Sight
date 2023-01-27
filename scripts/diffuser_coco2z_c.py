import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
from diffusers import StableDiffusionImageEncodingPipeline, StableDiffusionTextEncodingPipeline
from PIL import Image
import torch
from torchvision import transforms



device = "cuda:0"
#home to encode_image()
im_model = StableDiffusionImageEncodingPipeline.from_pretrained(
  "lambdalabs/sd-image-variations-diffusers",
  revision="v2.0",
  )
im_model = im_model.to(device)
#home to encode_promp() and encode_latents()
text_model = StableDiffusionTextEncodingPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to(device)
text_model = text_model.to(device)

im = Image.open("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/00000.png")
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
inp = tform(im).to(device).unsqueeze(0)
print(inp.shape)

image_embed = model.encode_image(inp)
torch.save(image_embed, "/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/image_embed.pt")

out = model(inp, guidance_scale=3)
out["images"][0].save("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/result2.jpg")