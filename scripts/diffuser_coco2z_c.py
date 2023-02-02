import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
from diffusers import StableDiffusionImageEncodingPipeline, StableDiffusionTextEncodingPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionImageVariationPipeline
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np
from nsd_access import NSDAccess
from tqdm import tqdm


def load_img(im_array):

    image = Image.fromarray(im_array)
    w, h = image.size

    # print(f"loaded input image of size ({w}, {h}) from array")

    w, h = 512, 512  # resize to integer multiple of 64
    imagePil = image.resize((w, h), resample=Image.Resampling.LANCZOS)
    image = np.array(imagePil).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2. * image - 1., imagePil

nsda = NSDAccess('/home/naxos2-raid25/kneel027/home/surly/raid4/kendrick-data/nsd', '/home/naxos2-raid25/kneel027/home/kneel027/nsd_local')

# Setting up the device to use the GPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# Initialize Clip model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

#home to encode_image() and encode()
im_model = StableDiffusionImageEncodingPipeline.from_pretrained("lambdalabs/sd-image-variations-diffusers", revision="v2.0")
im_model = im_model.to(device)
#home to encode_prompt() and encode_latents()
text_model = StableDiffusionTextEncodingPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to(device)
text_model = text_model.to(device)


# Importing the caption object with 5 captions per image
# Captions is a lists of lists, containing the 5 captions. 
captions = nsda.read_image_coco_info([i for i in range(73000)], info_type='captions', show_annot=False)

# Iterate through all images and captions in nsd sampled from COCO
for i in tqdm(range(0, 10)):
    # Array of image data 1 x 425 x 425 x 3 (Stores pixel intensities)
    img_arr = nsda.read_images([i], show=False)
    init_image, img_pil = load_img(img_arr[0])
    # image = Image.fromarray(img_arr.reshape((425, 425, 3)))

    # w, h = 512, 512  # resize to integer multiple of 64
    # image = image.resize((w, h), resample=Image.Resampling.LANCZOS)
    # im = Image.open("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/dog2.png")
    c_img = im_model.encode_image(img_pil)
    # c_img = torch.load("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/dog_image_features.pt").to(device)
    
    #find best prompt
    prompts = []
    # Load the 5 prompts for each image
    for j in range(len(captions[i])):
        # Index into the caption list and get the corresponding 5 captions. 
        prompts.append(captions[i][j]['caption'])
    inputs = preprocess(text=prompts, images=img_pil, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
    prompt = prompts[torch.argmax(probs)]
    
    #embed best prompt
    # text_embed = preprocess(text=[prompt], return_tensors="pt", padding=True)
    # text_features = clip_model.get_text_features(**text_embed)
    # negative_text_embeds = torch.zeros_like(text_features)
    # text_features = torch.cat([negative_text_embeds, text_features])
    # text_features = text_features[:, None, :].to(device)
    c_text = text_model.encode_prompt(prompt)
    
    c = c_img + c_text
    
                
    # Use the Stable Diffusion method to move to latent space to create latent z vector
    z = text_model.encode_latents(init_image)
    
    # Save the c and z vectors into there corresponding files. 
    torch.save(c, "/home/naxos2-raid25/kneel027/home/kneel027/nsd_local/nsddata_stimuli/tensors/c_combined/" + str(i) + ".pt")
    torch.save(c_text, "/home/naxos2-raid25/kneel027/home/kneel027/nsd_local/nsddata_stimuli/tensors/c_prompt/" + str(i) + ".pt")
    # torch.save(z, "/home/naxos2-raid25/kneel027/home/kneel027/nsd_local/nsddata_stimuli/tensors/z/" + str(i) + ".pt")
    if(i<10):
      z_only = im_model.reconstruct(z=z, strength=0.0)
      c_only_image = im_model.reconstruct(c=c_img, strength=1.0)
      c_only_text = im_model.reconstruct(c=c_text, strength=1.0)
      c_combined = im_model.reconstruct(c=c, strength=1.0)
      image_and_z = im_model.reconstruct(z=z, c=c, strength=0.75)
      z_only.save("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/test_imgs3/" + str(i) + "_z_only.png")
      c_only_image.save("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/test_imgs3/" + str(i) + "_c_only_image.png")
      c_only_text.save("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/test_imgs3/" + str(i) + "_c_only_text.png")
      c_combined.save("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/test_imgs3/" + str(i) + "_c_combined.png")
      image_and_z.save("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/test_imgs3/" + str(i) + "_z_and_c.png")
