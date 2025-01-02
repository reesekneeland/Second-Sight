import os
import sys
sys.path.append('versatile_diffusion')
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from torchmetrics import PearsonCorrCoef
from torchvision import transforms
import torch
import clip
import numpy as np
from PIL import Image
import shutil
from utils import *
from tqdm import tqdm
import os
import PIL
from PIL import Image
import numpy as np
import time
from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model
from lib.experiments.sd_default import color_adjust, auto_merge_imlist
from lib.model_zoo.vd import VD
from lib.cfg_holder import cfg_unique_holder as cfguh
from lib.cfg_helper import get_command_line_args, cfg_initiates, load_cfg_yaml

cfgm_name = 'vd_noema'

pth = 'versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth'
cfgm = model_cfg_bank()(cfgm_name)
net = get_model()(cfgm)
sd = torch.load(pth, map_location='cpu')
net.load_state_dict(sd, strict=False)    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device)
net.clip = net.clip.to(device)

    
subject_betas = []
for s, subject in enumerate([1,2,5,7]):
    _, _, x_test, _, _, _, _ = load_nsd(vector="images", subject=subject, loader=False, average=True, nest=False, include_heldout=True)
    subject_betas.append(x_test.to("cpu"))
root = "output/boi_paper"
cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

model = "braindiffuser"
folder = "brain-optimized-inference-v2.3"
for n in range(38, 64):
    PeC = PearsonCorrCoef(num_outputs=n)
    for mode in ["brain_corr", "clip"]:
        new_model_name = f"{model}_top{n}_{mode}"
        # using the rep dimension for subjects
        all_recons = torch.zeros((1000, 4, 3, 512, 512))
        for s, subject in enumerate([1,2,5,7]):
            print(f"Model: {model}, Folder: {folder}, N: {n}, Mode: {mode}, Subject: {subject}")
            x_test = subject_betas[s]
            for sample in tqdm(range(1000)):
                start = time.time()
                source_dir = f"{root}/{folder}/subject{subject}/{sample}/"
                dest_dir = f"{root}/{new_model_name}/subject{subject}/{sample}/"
                os.makedirs(dest_dir, exist_ok=True)
                
                start2 = time.time()
                shutil.copy(f"{source_dir}/basemodel_recon.png", f"{dest_dir}/basemodel_recon.png")
                start2 = time.time()
                shutil.copy(f"{source_dir}/low_level.png", f"{dest_dir}/low_level.png")
                start2 = time.time()
                if mode == "clip":
                    target_clip = torch.load(f"{root}/{folder}/subject{subject}/{sample}/clip_vision.pt", map_location="cpu", weights_only=False).to(torch.float32).reshape((1,197376))
                    torch.save(target_clip, f"{dest_dir}/clip.pt")
                    iter0_images = torch.zeros((64, 3, 512, 512))
                    for i in range(64):
                        image_path = f"{source_dir}/iter_0/images/{i}.png"
                        image = Image.open(image_path).convert("RGB")
                        iter0_images[i] = transforms.ToTensor()(image)
                    start2 = time.time()
                    iter0_clip = net.clip_encode_vision(iter0_images.to(device)).reshape((64, target_clip.shape[-2] * target_clip.shape[-1])).to("cpu")
                    start2 = time.time()
                    torch.save(iter0_clip, f"{source_dir}/iter_0/iter0_clip.pt")
                    start2 = time.time()
                    # Calculate cosine similarity
                    similarities = cos(target_clip, iter0_clip)
                    start2 = time.time()
                else:
                    beta = x_test[sample]
                    beta_prime_path = f"{source_dir}/iter_0/beta_primes"
                    beta_primes = torch.zeros((n, beta.shape[-1]))
                    start2 = time.time()
                    for i in range(n):
                        beta_primes[i] = torch.load(f"{beta_prime_path}/{i}.pt", map_location="cpu")
                    start2 = time.time()
                    similarities = PeC(beta.repeat(n, 1).moveaxis(0, 1), beta_primes.moveaxis(0, 1))
                    start2 = time.time()
                    
                # Copy the best image
                best_image_idx = torch.argmax(similarities).item()
                start2 = time.time()
                best_image_path = f"{source_dir}/iter_0/images/{best_image_idx}.png"
                shutil.copy(best_image_path, f"{dest_dir}/0.png")
                start2 = time.time()
                all_recons[sample, s] = transforms.ToTensor()(Image.open(best_image_path).resize((512,512)))
                
        torch.save(all_recons, f"{root}/{new_model_name}_all_recons.pt")