import os
import sys
sys.path.append('versatile_diffusion')
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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

    
root = "output/boi_paper"
cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

model = "braindiffuser"
for mode in ["vision", "imagery"]:
    folder = f"{mode}/boi-v2.3"
    subject_betas = []
    for s, subject in enumerate([1,2,5,7]):
        x_test, _ = load_nsd_mental_imagery(vector = "images", subject=subject, mode=mode, stimtype="all", average=True, nest=False)
        subject_betas.append(x_test.to("cpu"))
    for n in range(1, 64):
        PeC = PearsonCorrCoef(num_outputs=n)
        for metric in ["clip", "brain_corr"]:
            new_model_name = f"{model}_top{n}_{metric}"
            # using the rep dimension for subjects
            
            for s, subject in enumerate([1,2,5,7]):
                all_recons = torch.zeros((18, 1, 3, 224, 224))
                print(f"Model: {model}, Folder: {folder}, N: {n}, Metric: {metric}, Mode: {mode}, Subject: {subject}")
                x_test = subject_betas[s]
                for sample in tqdm(range(18)):
                    start = time.time()
                    source_dir = f"{root}/raw/{folder}/subject{subject}/{sample}/"
                    dest_dir = f"{root}/raw/{new_model_name}/subject{subject}/{sample}/"
                    os.makedirs(dest_dir, exist_ok=True)
                    
                    # shutil.copy(f"{source_dir}/basemodel_recon.png", f"{dest_dir}/basemodel_recon.png")
                    # shutil.copy(f"{source_dir}/low_level.png", f"{dest_dir}/low_level.png")
                    if metric == "clip":
                        target_clip = torch.load(f"{dest_dir}/clip.pt", map_location="cpu", weights_only=False).to(torch.float32).reshape((1,197376))
                        # torch.save(target_clip, f"{dest_dir}/clip.pt")
                        iter0_clip_path = f"{source_dir}/iter_0/iter0_clip.pt"
                        if not os.path.exists(iter0_clip_path):
                            iter0_images = torch.zeros((64, 3, 224, 224))
                            for i in range(64):
                                image_path = f"{source_dir}/iter_0/images/{i}.png"
                                image = Image.open(image_path).convert("RGB").resize((224,224))
                                iter0_images[i] = transforms.ToTensor()(image)
                            start2 = time.time()
                            iter0_clip = net.clip_encode_vision(iter0_images.to(device)).reshape((64, target_clip.shape[-2] * target_clip.shape[-1])).to("cpu")
                            torch.save(iter0_clip, iter0_clip_path)
                        else:
                            iter0_clip = torch.load(iter0_clip_path, map_location="cpu")
                        # Calculate cosine similarity
                        similarities = cos(target_clip, iter0_clip[:n])
                        # print("similarity length", len(similarities), target_clip.shape, iter0_clip.shape)
                    else:
                        beta = x_test[sample]
                        beta_prime_path = f"{source_dir}/iter_0/beta_primes"
                        beta_primes = torch.zeros((n, beta.shape[-1]))
                        for i in range(n):
                            beta_primes[i] = torch.load(f"{beta_prime_path}/{i}.pt", map_location="cpu")
                        similarities = PeC(beta.repeat(n, 1).moveaxis(0, 1), beta_primes.moveaxis(0, 1))
                    # Copy the best image
                    best_image_idx = torch.argmax(similarities).item()
                    best_image_path = f"{source_dir}/iter_0/images/{best_image_idx}.png"
                    shutil.copy(best_image_path, f"{dest_dir}/0.png")
                    all_recons[sample] = transforms.ToTensor()(Image.open(best_image_path).resize((224,224)))
                    
                torch.save(all_recons, f"{root}/evals/{new_model_name}_subj0{subject}_all_recons_{mode}.pt")