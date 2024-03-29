import os
import torch
from torchmetrics import PearsonCorrCoef
import numpy as np
from PIL import Image
from utils import *
import math
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
import yaml
from clip_encoder import CLIPEncoder
from gnet8_encoder import GNet8_Encoder
from autoencoder import AutoEncoder
from diffusers import StableUnCLIPImg2ImgPipeline
from torchvision.transforms.functional import pil_to_tensor


class StochasticSearch():
    def __init__(self, 
                modelParams=["gnet"],
                device="cuda:0",
                subject=1,
                log=False, # flag to save all the intermediate images, beta_primes, clip vectors, and strength values. Used for ablation studies.
                ae = True,
                n_iter=10,
                n_samples=100,
                n_branches=4,
                disable_SD=False):
        with torch.no_grad():
            self.subject = subject
            subject_sizes = [0, 15724, 14278, 0, 0, 13039, 0, 12682]
            self.x_size = subject_sizes[self.subject]
            self.modelParams = modelParams
            self.EncModels = []
            self.log = log
            self.ae = ae
            self.device = device
            self.n_iter = n_iter
            self.n_samples = n_samples
            self.n_branches = n_branches
            self.vector = "images"
            if not disable_SD:
                self.R = StableUnCLIPImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16).to(self.device)
            
            
            # Configure AutoEncoders
            if(self.ae):
                if len(modelParams)>1:
                    # Hybrid encoder
                    self.AEModel = AutoEncoder(config="hybrid",
                                                        inference=True,
                                                        subject=self.subject,
                                                        device=self.device)
                else:
                    self.AEModel = AutoEncoder(config=modelParams[0],
                                                        inference=True,
                                                        subject=self.subject,
                                                        device=self.device)
            for param in modelParams:
                if param == "gnet":
                    self.EncModels.append(GNet8_Encoder(device=self.device,
                                                    subject=self.subject))
                elif param == "clip":
                    self.EncModels.append(CLIPEncoder(inference=True,
                                                    subject=self.subject,
                                                    device=self.device))

    # Hybrid encoder implementation, predict beta primes using the ensemble of encoding models in the SCS config
    def predict(self, x, mask=None, return_clips=False):
        
        if(isinstance(x, list)):
            combined_preds = torch.zeros((len(self.modelParams), len(x), self.x_size)).cpu()
            img_tensor = torch.zeros((len(x), 425, 425, 3))
            for i, sample in enumerate(x):
                image = sample.resize((425, 425))
                img_tensor[i] = torch.from_numpy(np.array(image)).reshape((425, 425, 3))
            x = img_tensor
        elif(isinstance(x, torch.Tensor)):
            assert 425 in x.shape or 541875 in x.shape,"Tensor of wrong size"
            combined_preds = torch.zeros((len(self.modelParams), x.shape[0], self.x_size)).cpu()
            x = x.reshape((x.shape[0], 425, 425, 3))
        else:
            raise TypeError
        
        sample_clips = self.R.encode_image_raw(x, device=self.device)
        
        for c, mType in enumerate(self.modelParams):
            if mType == "gnet":
                combined_preds[c] = self.EncModels[c].predict(x, mask).cpu()
            elif mType == "clip":
                combined_preds[c] = self.EncModels[c].predict(sample_clips, mask).cpu()
        if return_clips:
            return torch.mean(combined_preds, dim=0), sample_clips.cpu()
        else:
            return torch.mean(combined_preds, dim=0)

    def benchmark_config(self, average=True):
        with torch.no_grad():
            # y_test = Brain data
            # x_test = clip data
            _, _, y_test, _, _, x_test, _ = load_nsd(vector="images", 
                                                    loader=False,
                                                    average=average,
                                                    subject=self.subject)

            criterion = torch.nn.MSELoss()
            PeC = PearsonCorrCoef(num_outputs=y_test.shape[0]).to(self.device)
            
            x_test = x_test.to(self.device)
            y_test = y_test.to(self.device)
            
            pred_y = self.predict(x_test)
            
            loss = criterion(pred_y, y_test)
                
            pearson = torch.mean(PeC(pred_y.moveaxis(0,1), y_test.moveaxis(0,1)))
            
            pred_y = pred_y.cpu().detach()
            y_test = y_test.cpu().detach()
            PeC = PearsonCorrCoef()
            r = []
            for voxel in range(pred_y.shape[1]):
                # Correlation across voxels for a sample (Taking a column)
                r.append(PeC(pred_y[:,voxel], y_test[:,voxel]))
            r = np.array(r)
            
            print("Model: Hybrid Encoder, Subject: {}, Averaged: {}".format(self.subject, average))
            print("Vector Correlation: ", float(pearson))
            print("Mean Pearson: ", np.mean(r))
            print("Loss: ", float(loss))
            plt.hist(r, bins=50, log=True)
            plt.savefig("data/charts/subject{}_hybrid_encoder_pearson_correlation.png".format(self.subject))

    def generateNSamples(self, image, c_i, n, strength=1.0):
        images = []
        for i in range(n):
            im = self.R.reconstruct(image=image,
                                             image_embeds=c_i, 
                                             strength=strength,
                                             negative_prompt="text, caption")
            images.append(im)
        return images


    def score_samples(self, beta, images, save_path):
        beta_primes, sample_clips = self.predict(images, return_clips=True)
        if(self.log):
            os.makedirs(save_path + "beta_primes/", exist_ok=True)
            os.makedirs(save_path + "images/", exist_ok=True)
            for i, im in enumerate(images):
                im.save(save_path + "images/{}.png".format(i))
            for i in range(beta_primes.shape[0]):
                torch.save(beta_primes[i], "{}/beta_primes/{}.pt".format(save_path, i))
        scores = []
        PeC = PearsonCorrCoef(num_outputs=beta_primes.shape[0]).to(self.device) 
        for i in range(beta.shape[0]):
            xDup = beta[i].repeat(beta_primes.shape[0], 1).moveaxis(0, 1).to(self.device)
            score = PeC(xDup, beta_primes.moveaxis(0, 1).to(self.device))
            scores.append(score)
        scores = torch.mean(torch.stack(scores), dim=0)
        return scores, sample_clips, beta_primes
    
    # Preprocess beta, remove empty trials, and autoencode if necessary
    def prepare_betas(self, beta):
        beta_list = []
        for i in range(beta.shape[0]):
            if(torch.count_nonzero(beta[i]) > 0):
                if(self.ae):
                    beta_list.append(self.AEModel.predict(beta[i]))
                else:
                    beta_list.append(beta[i])
        return torch.stack(beta_list)

    # Main search method
    # c_i is a 1024 clip vector
    # beta is a 3*x_size tensor of brain data to use as guidance targets
    # init_img is a pil image to serve as a low level strcutral guesscur_iter
    def search(self, sample_path, beta, c_i, init_img=None):
        with torch.no_grad():
            # Initialize search variables
            best_image, best_clip, best_distribution_score = init_img, c_i, -1
            iter_images, iter_scores = [], []
            pbar = tqdm(total=self.n_iter, desc="Search iterations")
            # Prepare beta as search target
            beta = self.prepare_betas(beta)
            
            # Generate iteration 0
            iteration_samples = self.generateNSamples(image=init_img, 
                                                    c_i=c_i,  
                                                    n=self.n_samples,  
                                                    strength=0.92-0.30*math.pow(1/self.n_iter, 3))
            
            iter_path = "{}iter_0/".format(sample_path)
            best_batch_path = "{}best_batch".format(iter_path)
            if(self.log):
                os.makedirs(iter_path, exist_ok=True)
                if os.path.islink(os.path.abspath(best_batch_path)):
                    remove_symlink(os.path.abspath(best_batch_path))
                os.symlink(os.path.abspath(iter_path), os.path.abspath(best_batch_path), target_is_directory=True)
            # Score iteration 0
            iteration_scores, iteration_clips, iteration_beta_primes = self.score_samples(beta, iteration_samples, save_path=iter_path)
            
            #Update best image and iteration images from iteration 0
            if float(torch.mean(iteration_scores)) > best_distribution_score:
                best_distribution_score = float(torch.mean(iteration_scores))
                best_image = iteration_samples[int(torch.argmax(iteration_scores))]
                best_clip = iteration_clips[int(torch.argmax(iteration_scores))]
                c_i = slerp(c_i, best_clip, 0.2*(math.pow(1/self.n_iter, 2)))
            iter_scores.append(best_distribution_score)
            iter_images.append(best_image)
            
            best_distribution_params = {
                                        "images":iteration_samples,
                                        "beta_primes": iteration_beta_primes,
                                        "clip":c_i,
                                        "z_img": init_img,
                                        "strength": 0.92-0.30*math.pow(1/self.n_iter, 3)}
            pbar.update(1)
            # Iteration >0 loop
            for i in range(1, self.n_iter):
                # Initalize parameters for iteration
                strength = 0.92-0.30*(math.pow((i+1)/self.n_iter, 3))
                momentum = 0.2*(math.pow((i+1)/self.n_iter, 2))
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
                clip_seeds = [slerp(c_i, iteration_clips[int(seed)], momentum) for seed in seed_indices]
                if not best_image not in z_seeds:
                    z_seeds.append(best_image)
                    clip_seeds.append(c_i)
                
                # Make image batches
                tqdm.write("Strength: {}, Momentum: {}, N: {}".format(strength, momentum, n_i))
                iteration_samples, iteration_clips, iteration_scores, = [], [], []
                best_batch_score = -1
                for b in range(len(z_seeds)):
                    #Generate and save batch clip
                    branch_clip = slerp(c_i, clip_seeds[b], momentum)
                    batch_path = "{}/batch_{}/".format(iter_path, b)
                    if(self.log):
                        os.makedirs(batch_path, exist_ok=True)
                        torch.save(branch_clip, batch_path+"batch_clip.pt")
                        z_seeds[b].save(batch_path+"z_img.png")
                        
                    #Generate batch samples
                    batch_samples = self.generateNSamples(image=z_seeds[b], 
                                                            c_i=branch_clip,  
                                                            n=n_i,  
                                                            strength=strength)
                    #Score batch samples
                    batch_scores, batch_clips, batch_beta_primes = self.score_samples(beta, batch_samples, batch_path)
                    
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
                    iteration_clips.append(batch_clips)
                    
                    #Update best image and iteration images
                    if torch.mean(batch_scores) > best_distribution_score:
                        best_distribution_score = torch.mean(batch_scores)
                        best_image = batch_samples[int(torch.argmax(batch_scores))]
                        best_clip = clip_seeds[b]
                        best_distribution_params = {
                                        "images":batch_samples,
                                        "beta_primes": batch_beta_primes,
                                        "clip":branch_clip,
                                        "z_img": z_seeds[b],
                                        "strength": strength}
                
                # Concatenate scores and clips from each batch to be sorted for seeding the next iteration
                iteration_scores = torch.concat(iteration_scores, dim=0)
                iteration_clips = torch.concat(iteration_clips, dim=0)
                c_i = slerp(c_i, best_clip, momentum)
                iter_scores.append(best_distribution_score)
                iter_images.append(best_image)
                pbar.update(1)
                tqdm.write("BC: {}".format(best_distribution_score))
            pbar.close()
        return best_image, best_distribution_params, iter_images, iter_scores


    
