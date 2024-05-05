import os
import torch
import torch.nn as nn
from torchmetrics import PearsonCorrCoef
import numpy as np
from PIL import Image
from utils import *
import math
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
import yaml
from clipper import Clipper
from gnet8_encoder import GNet8_Encoder
from torchvision.transforms.functional import pil_to_tensor
from torchvision import transforms


class StochasticSearch():
    def __init__(self, 
                modelParams=["gnet"],
                device="cuda:0",
                subject=1,
                log=False, # flag to save all the intermediate images, batch_clips, clip vectors, and strength values. Used for ablation studies.
                n_iter=10,
                n_samples=100,
                n_branches=4,
                disable_SD=False,
                basemodel="mindeye",
                hidden=True):
        with torch.no_grad():
            self.subject = subject
            subject_sizes = [0, 15724, 14278, 0, 0, 13039, 0, 12682]
            self.x_size = subject_sizes[self.subject]
            self.modelParams = modelParams
            self.log = log
            self.device = device
            self.n_iter = n_iter
            self.n_samples = n_samples
            self.n_branches = n_branches
            self.vector = "images"
            if not disable_SD:
                if basemodel=="mindeye":
                    from reconstructor import Reconstructor
                elif basemodel=="braindiffuser":
                    from reconstructor_bd import Reconstructor
                elif basemodel=="tagaki":
                    from reconstructor_sd import Reconstructor
                self.R = Reconstructor(device=self.device)     
            
                
            self.clip_extractor = Clipper("ViT-L/14", hidden_state=hidden, norm_embs=True, device=self.device)

    def generateNSamples(self, image, c_i, c_t, n, strength=1.0):
        images = []
        for i in range(n):
            im = self.R.reconstruct(image=image,
                                    c_i=c_i, 
                                    c_t=c_t,
                                    strength=strength)
            images.append(im)
        return images
    
    def batchwise_cosine_similarity(self, Z,B):
        B = B.T
        Z_norm = torch.linalg.norm(Z, dim=1, keepdim=True)  # Size (n, 1).
        B_norm = torch.linalg.norm(B, dim=0, keepdim=True)  # Size (1, b).
        cosine_similarity = ((Z @ B) / (Z_norm @ B_norm)).T
        return cosine_similarity


    def score_samples(self, clip_target, images, save_path):
        image_tensor = torch.zeros((len(images), 3, 512, 512))
        for i in range(len(images)):
            image_tensor[i] = transforms.ToTensor()(images[i])
        candidate_embeddings = self.clip_extractor.embed_image(image_tensor.to(self.device)).float() # CLIP-Image
        candidate_embeddings = candidate_embeddings.reshape(len(candidate_embeddings),-1)
        candidate_embeddings = nn.functional.normalize(candidate_embeddings,dim=-1)
        
        if(self.log):
            os.makedirs(save_path + "batch_clips/", exist_ok=True)
            os.makedirs(save_path + "images/", exist_ok=True)
            for i, im in enumerate(images):
                im.save(save_path + "images/{}.png".format(i))
                torch.save(candidate_embeddings[i], "{}/batch_clips/{}.pt".format(save_path, i))
        scores = self.batchwise_cosine_similarity(clip_target, candidate_embeddings).view(-1)
        # print("SCORES", scores.shape, scores)
        return scores, candidate_embeddings
    
    

    # Main search method
    # clip_target is decoded disjoint clip vector optimized for retrieval to use as guidance targets
    # init_img is a pil image to serve as a low level strcutral guesscur_iter
    def search(self, sample_path, clip_target, c_i, c_t=None, init_img=None, basemodel_recon=None):
        with torch.no_grad():
            basemodel_recon_score, basemodel_recon_batch_clips = self.score_samples(clip_target, [basemodel_recon], save_path=f"{sample_path}basemodel_recon/")
            print(f"Basemodel recon_score: {basemodel_recon_score}")
            # Initialize search variables
            best_image, best_distribution_score = init_img, -1
            iter_images, iter_scores, var_scores = [], [], []
            pbar = tqdm(total=self.n_iter, desc="Search iterations")
            
            # Generate iteration 0
            iteration_samples = self.generateNSamples(image=init_img, 
                                                    c_i=c_i,
                                                    c_t=c_t,  
                                                    n=self.n_samples,  
                                                    strength=0.92)
            
            iter_path = "{}iter_0/".format(sample_path)
            best_batch_path = "{}best_batch".format(iter_path)
            if(self.log):
                os.makedirs(iter_path, exist_ok=True)
                torch.save(torch.tensor(0.92), iter_path+"iter_strength.pt")
                if os.path.islink(os.path.abspath(best_batch_path)):
                    remove_symlink(os.path.abspath(best_batch_path))
                os.symlink(os.path.abspath(iter_path), os.path.abspath(best_batch_path), target_is_directory=True)
            # Score iteration 0
            iteration_scores, iteration_batch_clips = self.score_samples(clip_target, iteration_samples, save_path=iter_path)
            
            #Update best image and iteration images from iteration 0
            if float(torch.mean(iteration_scores)) > best_distribution_score:
                best_distribution_score = float(torch.mean(iteration_scores))
                best_image = iteration_samples[int(torch.argmax(iteration_scores))]
            iter_scores.append(float(best_distribution_score))
            iter_images.append(best_image)
            bp_var = bootstrap_variance(iteration_batch_clips)
            var_scores.append(bp_var)
            tqdm.write("SEARCH VARIANCE BP: {:.10f}, RETRIEVAL SIM: {:.10f}".format(bp_var, float(torch.mean(iteration_scores))))
            best_distribution_params = {
                                        "images":iteration_samples,
                                        "batch_clips": iteration_batch_clips,
                                        "z_img": init_img,
                                        "strength": 0.92}
            pbar.update(1)
            # Iteration >0 loop
            for i in range(1, self.n_iter):
                # Initalize parameters for iteration
                strength = 0.92-0.40*(math.pow((i+1)/self.n_iter, 2))
                n_i = max(10, int((self.n_samples/self.n_branches)*strength))
                # Save
                iter_path = "{}iter_{}/".format(sample_path, i)
                best_batch_path = "{}best_batch".format(iter_path)
                if(self.log):
                    os.makedirs(iter_path, exist_ok=True)
                    torch.save(torch.tensor(strength), iter_path+"iter_strength.pt")
                
                # Update seeds from previous iteration
                seed_indices = torch.topk(iteration_scores, self.n_branches).indices
                z_seeds = [iteration_samples[int(seed)] for seed in seed_indices]
                if not best_image not in z_seeds:
                    z_seeds.append(best_image)
                
                # Make image batches
                tqdm.write("Strength: {}, N: {}".format(strength, n_i))
                iteration_samples, iteration_scores, iteration_vars = [], [], []
                best_batch_score = -1
                for b in range(len(z_seeds)):
                    batch_path = "{}/batch_{}/".format(iter_path, b)
                    if(self.log):
                        os.makedirs(batch_path, exist_ok=True)
                        z_seeds[b].save(batch_path+"z_img.png")
                        
                    #Generate batch samples
                    batch_samples = self.generateNSamples(image=z_seeds[b], 
                                                            c_i=c_i, 
                                                            c_t=c_t,  
                                                            n=n_i,  
                                                            strength=strength)
                    #Score batch samples
                    batch_scores, batch_batch_clips = self.score_samples(clip_target, batch_samples, batch_path)
                    
                    # Keep track of best batch for logging
                    
                    if torch.mean(batch_scores) > best_batch_score:
                        best_batch_score = torch.mean(batch_scores)
                        # Create symlink from current batch folder to "best_batch" folder for easy traversing later
                        if(self.log):
                            if os.path.islink(os.path.abspath(best_batch_path)):
                                remove_symlink(os.path.abspath(best_batch_path))
                            os.symlink(os.path.abspath(batch_path), os.path.abspath(best_batch_path), target_is_directory=True)
                            
                    # Keep track of all batches in iteration for updating seeds
                    iteration_samples += batch_samples
                    iteration_scores.append(batch_scores)
                    bp_var = bootstrap_variance(batch_batch_clips)
                    iteration_vars.append(bp_var)
                    #Update best image and iteration images
                    if torch.mean(batch_scores) > best_distribution_score:
                        best_distribution_score = torch.mean(batch_scores)
                        best_image = batch_samples[int(torch.argmax(batch_scores))]
                        best_distribution_params = {
                                        "images":batch_samples,
                                        "batch_clips": batch_batch_clips,
                                        "z_img": z_seeds[b],
                                        "strength": strength}
                
                # Concatenate scores from each batch to be sorted for seeding the next iteration
                iteration_scores = torch.concat(iteration_scores, dim=0)
                iteration_var = np.mean(np.array(iteration_vars))
                iter_scores.append(float(best_distribution_score))
                iter_images.append(best_image)
                
                
                var_scores.append(iteration_var)
                tqdm.write("SEARCH VARIANCE BP: {:.10f}, RETRIEVAL SIM: {:.10f}".format(bp_var, float(torch.mean(iteration_scores))))
                pbar.update(1)
            pbar.close()
        return best_image, best_distribution_params, iter_images, iter_scores, var_scores, basemodel_recon_score


    
