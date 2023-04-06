import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
import torch
import numpy as np
from PIL import Image
sys.path.append('src')
from reconstructor import Reconstructor
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import random
from nsd_access import NSDAccess
from tqdm import tqdm
import time

nsda = NSDAccess('/home/naxos2-raid25/kneel027/home/surly/raid4/kendrick-data/nsd', '/home/naxos2-raid25/kneel027/home/kneel027/nsd_local')

captions = nsda.read_image_coco_info([i for i in range(73000)], info_type='captions', show_annot=False)

init_clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
init_preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

R = Reconstructor(device="cuda:1")
# Iterate through all images and captions in nsd sampled from COCO
start = time.time()
for i in tqdm(range(8100, 73000)):
    # Array of image data 1 x 425 x 425 x 3 (Stores pixel intensities)
    img_arr = nsda.read_images([i], show=False)
    # img_pil = Image.fromarray(img_arr)
    img_pil = Image.fromarray(img_arr.reshape((425, 425, 3)))
    
    #find best prompts
    prompts = []
    # Load the 5 prompts for each image
    for j in range(len(captions[i])):
        # Index into the caption list and get the corresponding 5 captions. 
        prompts.append(captions[i][j]['caption'])

    inputs = init_preprocess(text=prompts, images=img_pil, return_tensors="pt", padding=True)
    outputs = init_clip_model(**inputs)
    logits_per_image = outputs.logits_per_image # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1).tolist()[0] # we can take the softmax to get the label probabilities
    best_prompt = prompts[np.argmax(probs)]
    # print(prompts)
    # print(probs)
    # print(best_prompt)
    
    c_i = R.encode_image(img_pil)
    c_t = R.encode_text(best_prompt)
    # print(c_i.shape, c_t.shape)
    
    # Save the c and z vectors into there corresponding files. 
    torch.save(c_i, "/home/naxos2-raid25/kneel027/home/kneel027/nsd_local/nsddata_stimuli/tensors/c_img_vd/" + str(i) + ".pt")
    torch.save(c_t, "/home/naxos2-raid25/kneel027/home/kneel027/nsd_local/nsddata_stimuli/tensors/c_text_vd/" + str(i) + ".pt")
    # if(i<=10):
    #     c_combined = R.reconstruct(c_i=c_i, c_t=c_t)
    #     c_combined.save("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/test_imgs5/" + str(i) + "_c_combined.png")
    #     img_pil.save("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/test_imgs5/" + str(i) + "_gt.png")

end = time.time()
print("elapsed time: ", end - start)