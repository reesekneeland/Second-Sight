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
            self.device = device
            self.n_iter = n_iter
            self.n_samples = n_samples
            self.n_branches = n_branches
            self.vector = "images"
            if not disable_SD:
                self.R = StableUnCLIPImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16, variation="fp16").to(self.device)
            
            
            # Hybrid encoder
            if len(modelParams)>1:
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
        combined_preds = torch.zeros((len(self.modelParams), len(x), self.x_size)).cpu()
        if(isinstance(x, list)):
            img_tensor = torch.zeros((len(x), 425, 425, 3))
            for i, sample in enumerate(x):
                image = sample.resize((425, 425))
                img_tensor[i] = pil_to_tensor(image)
            x = img_tensor
        x = x.reshape((x.shape[0], 425, 425, 3))

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

    def generateNSamples(self, image, c_i, n, strength=1):
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
            for im in images:
                im.save(save_path + "images/{}.png".format(i))
            for i in range(beta_primes.shape[0]):
                torch.save(beta_primes[i], "{}/beta_primes/{}.pt".format(save_path, i))
        beta_primes = beta_primes.moveaxis(0, 1).to(self.device)
        scores = []
        PeC = PearsonCorrCoef(num_outputs=beta_primes.shape[1]).to(self.device) 
        for i in range(beta.shape[0]):
            xDup = beta[i].repeat(beta_primes.shape[1], 1).moveaxis(0, 1).to(self.device)
            score = PeC(xDup, beta_primes)
            scores.append(score)
        scores = torch.mean(torch.stack(scores), dim=0)
        return scores, sample_clips
    
    # Preprocess beta, remove empty trials, and autoencode if necessary
    def denoise(self, beta):
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
            best_image, best_clip, cur_clip = init_img, c_i, c_i
            init_clip = [c_i] * self.n_branches
            iter_images, iter_scores = [], []
            best_search_score, best_distribution_score = -1, -1
            
            # Prepare beta as search target
            beta = self.denoise(beta)
            
            # Iteration 0
            iteration_samples = self.generateNSamples(image=init_img, 
                                                    c_i=c_i,  
                                                    n=self.n_samples,  
                                                    strength=0.92-0.30*(1/self.n_iter, 3))
            # Update best image
            iteration_scores, iteration_clips = self.score_samples(beta, iteration_samples, save_path=sample_path + "/iter_0/")
            
            # Iteration loop
            for i in tqdm(range(1, self.n_iter), desc="search iterations"):
                # Initalize parameters for iteration
                strength = 0.92-0.30*(math.pow((i+1)/self.n_iter, 3))
                momentum = 0.2*(math.pow((i+1)/self.n_iter, 2))
                n_i = max(10, int((self.n_samples/self.n_branches)*strength))
                
                if(self.log):
                    iter_path = "{}iter_{}/".format(sample_path, i)
                    os.makedirs(iter_path, exist_ok=True)
                    torch.save(torch.tensor(strength), iter_path+"iter_strength.pt")
                
                #Update best image and iteration images
                if float(torch.max(iteration_scores)) > best_search_score:
                    best_search_score = float(torch.max(iteration_scores))
                    best_image = iteration_samples[int(torch.argmax(iteration_scores))]
                    c_i = slerp(c_i, iteration_clips[int(torch.argmax(iteration_scores))], momentum)
                iter_scores.append(best_search_score)
                iter_images.append(best_image)
                
                # Update seeds from previous iteration
                z_seeds = iteration_samples[torch.topk(iteration_scores, self.n_branches).indices]
                clip_seeds = [slerp(c_i, c_prev, momentum) for c_prev in iteration_clips[torch.topk(iteration_scores, self.n_branches).indices]]
                if not best_image not in z_seeds:
                    z_seeds.append(best_image)
                    clip_seeds.append(best_clip)
                
                # Make image batches
                tqdm.write("Strength: {}, Momentum: {}, N: {}".format(strength, momentum, n_i))
                iteration_samples, iteration_clips, iteration_scores, best_iteration_score = [], [], 0
                for b in range(len(z_seeds)):
                    #Generate and save batch clip
                    branch_clip = slerp(cur_clip, clip_seeds[b], momentum)
                    if(self.log):
                        batch_path = "{}/batch_{}/".format(iter_path, b)
                        os.makedirs(batch_path, exist_ok=True)
                        torch.save(branch_clip, batch_path+"batch_clip.pt")
                        
                    #Generate batch samples
                    batch_samples = self.generateNSamples(image=z_seeds[b], 
                                                            c_i=branch_clip,  
                                                            n=n_i,  
                                                            strength=strength)
                    #Score batch samples
                    batch_scores, batch_clips = self.score_samples(beta, batch_samples, batch_path)
                    
                    if torch.mean(batch_scores) > best_iteration_score:
                        iteration_scores = batch_scores
                        iteration_samples = batch_samples
                    
                    best_batch_images.append(batch_samples[int(torch.argmax(batch_scores))])
                    average_batch_scores.append(float(torch.mean(batch_scores)))
                    best_batch_scores.append(float(torch.max(batch_scores)))
                    #Append scores, samples, and clips to iteration lists
                    scores.append(batch_scores)
                    samples += batch_samples
                    sample_clips += batch_clips
                if i > 0:
                    if not best_image not in z_seeds:
                        #Create batch directory
                        batch_count +=1
                        batch_path = "{}/batch_{}/".format(iter_path, self.n_branches+1)
                        os.makedirs(batch_path, exist_ok=True)
                        tqdm.write("Adding best image and clip to branches!")
                        #Save branch clip
                        if(self.log):
                            torch.save(cur_clip, batch_path+"batch_clip.pt")
                        branch_clips.append(cur_clip)
                        samples += self.generateNSamples(save_path=batch_path,
                                                        image=best_image, 
                                                        c_i=cur_clip,
                                                        n=n_i,  
                                                        strength=strength,
                                                        noise_level=noise)
                        #Score batch samples
                        batch_scores, batch_clips = self.score_samples(beta, batch_samples, batch_path)
                        best_batch_images.append(batch_samples[int(torch.argmax(batch_scores))])
                        best_batch_scores.append(float(torch.max(batch_scores)))
                        average_batch_scores.append(float(torch.mean(batch_scores)))
                        #Append scores, samples, and clips to iteration lists
                        scores.append(batch_scores)
                        samples += batch_samples
                        sample_clips += batch_clips
                        
                scores = torch.concat(scores, dim=0)
                #Set branches for next iteration based on top scores from this iteration, ignoring batches
                topn_pearson = torch.topk(scores, self.n_branches)
                for i in range(self.n_branches):
                    init_images[i] = samples[int(topn_pearson.indices[i])]
                    init_clip[i] = sample_clips[int(topn_pearson.indices[i])]
                best_batch_index = int(np.argmax(average_batch_scores))
                best_batch_corrrelation = float(best_batch_scores[best_batch_index])
                if(self.log):
                    torch.save(torch.tensor(best_batch_index), "{}/best_batch_index.pt".format(iter_path))
                tqdm.write("BC: {}".format(best_batch_corrrelation))
                
                iter_images.append(best_batch_images[int(best_batch_index)])
                iter_scores.append(best_batch_corrrelation)
                var_scores.append(float(torch.var(scores)))
                
                if best_batch_corrrelation > best_search_score or best_search_score == -1:
                    best_search_score = best_batch_corrrelation
                    best_image = best_batch_images[best_batch_index]
                    best_clip = branch_clips[best_batch_index]
                    cur_clip = slerp(cur_clip, best_clip, momentum)
            
        return best_image, iter_images, iter_scores, var_scores


    
