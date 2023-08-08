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


class StochasticSearch():
    def __init__(self, 
                modelParams=["gnetEncoder"],
                device="cuda:0",
                subject=1,
                log=False,
                n_iter=10,
                n_samples=100,
                n_branches=4,
                disable_SD=False):
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
                self.R = StableUnCLIPImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16, variation="fp16").to(self.device)
            self.EncModels = []
            self.EncType = []
            
            
            # Hybrid encoder
            if len(modelParams)>1:
                self.AEModel = AutoEncoder(config="hybrid",
                                                    inference=True,
                                                    subject=self.subject,
                                                    device=self.device)
                #self.encoderWeights = torch.load("masks/subject{}/{}_encoder_prediction_weights.pt".format(self.subject, "_".join(self.modelParams))).to(self.device)
            else:
                self.AEModel = AutoEncoder(config="gnet",
                                                    inference=True,
                                                    subject=self.subject,
                                                    device=self.device)
            for param in modelParams:
                if param == "gnetEncoder":
                    self.EncModels.append(GNet8_Encoder(device=self.device,
                                                    subject=self.subject))
                    self.EncType.append("images")
                elif param == "clipEncoder":
                    self.EncModels.append(CLIPEncoder(inference=True,
                                                    subject=self.subject,
                                                    device=self.device))
                    self.EncType.append("c_i")

    # Predict using the ensemble of encoding models in the SCS config
    def predict(self, x, mask=None, return_clips=False):
        combined_preds = torch.zeros((len(self.EncType), len(x), self.x_size))
        if(isinstance(x, torch.Tensor)):
            img_list = []
            for i, sample in enumerate(x):
                imagePil = process_image(x[i], 425, 425)
                img_list.append(imagePil)
            x = img_list

        if "clipEncoder" in self.modelParams:
            sample_clips = []
            for sample in x:
                sample_clips.append(self.R.encode_image_raw(sample, device=self.device))
        
        for c, mType in enumerate(self.EncType):
            if mType == "images":
                combined_preds[c] = self.EncModels[c].predict(x, mask).to(self.device)
            elif mType == "c_i":
                combined_preds[c] = self.EncModels[c].predict(torch.stack(sample_clips)[:,0,:], mask).to(self.device)
        if return_clips:
            return torch.mean(combined_preds, dim=0).cpu(), sample_clips
        else:
            return torch.mean(combined_preds, dim=0).cpu()

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

    def generateNSamples(self, save_path, image, c_i, n, strength=1, noise_level=1):
        images = []
        for i in range(n):
            im = self.R.reconstruct(image=image,
                                             image_embeds=c_i, 
                                             strength=strength,
                                             noise_level=noise_level,
                                             negative_prompt="text, caption")
            images.append(im)
            if(self.log):
                im.save(save_path + "{}.png".format(i))
            
        return images


    def score_samples(self, beta, images, save_path):
        beta_primes, sample_clips = self.predict(images, return_clips=True)
        if(self.log):
            for i in range(beta_primes.shape[0]):
                torch.save(beta_primes[i], "{}/{}_beta_prime.pt".format(save_path, i))
        beta_primes = beta_primes.moveaxis(0, 1).to(self.device)
        scores = []
        PeC = PearsonCorrCoef(num_outputs=beta_primes.shape[1]).to(self.device) 
        for i in range(beta.shape[0]):
            xDup = beta[i].repeat(beta_primes.shape[1], 1).moveaxis(0, 1).to(self.device)
            score = PeC(xDup, beta_primes)
            scores.append(score)
        scores = torch.mean(torch.stack(scores), dim=0)
        return scores, sample_clips

    # Main search method
    # c_i is a 1024 clip vector
    # beta is a 3*x_size tensor of brain data to use as guidance targets
    # init_img is a pil image to serve as a low level strcutral guess
    def search(self, sample_path, beta, c_i, init_img=None):
        with torch.no_grad():
            #initialize
            best_image, best_clip, cur_clip = init_img, c_i, c_i
            init_clip = [c_i] * self.n_branches
            init_images = [init_img] * self.n_branches
            iter_images, iter_scores, var_scores = [], [], []
            best_search_corrrelation = -1
            # Preprocess beta, remove empty trials, and autoencode if necessary
            beta_list = []
            for i in range(beta.shape[0]):
                if(torch.count_nonzero(beta[i]) > 0):
                    if(self.ae):
                        beta_list.append(self.AEModel.predict(beta[i]))
                    else:
                        beta_list.append(beta[i])
                else:
                    tqdm.write("SKIPPING EMPTY BETA")
            beta = torch.stack(beta_list)
            for cur_iter in tqdm(range(self.n_iter), desc="search iterations"):
                iter_path = "{}/iter_{}/".format(sample_path, cur_iter)
                os.makedirs(iter_path, exist_ok=True)
                batch_count = self.n_branches
                if init_img is None:
                    strength = 1-0.5*(math.pow(cur_iter/self.n_iter, 3))
                else:
                    strength = 0.92-0.30*(math.pow((cur_iter+1)/self.n_iter, 3))
                
                momentum = 0.2*(math.pow((cur_iter+1)/self.n_iter, 2))
                noise = 0
                if(self.log):
                    torch.save(torch.tensor(strength), iter_path+"iter_strength.pt")
                n_i = max(10, int((self.n_iter/self.n_branches)*strength))
                tqdm.write("Strength: {}, Momentum: {}, Noise: {}, N: {}".format(strength, momentum, noise, n_i))
                
                samples = []
                sample_clips = []
                scores = []

                best_batch_scores = []
                best_batch_images = []
                average_batch_scores = []
                branch_clips = []
                for i in range(self.n_branches):
                    #Create batch directory
                    batch_path = "{}/batch_{}/".format(iter_path, i)
                    os.makedirs(batch_path, exist_ok=True)
                    #Generate and save batch clip
                    branch_clip = slerp(cur_clip, init_clip[i], momentum)
                    branch_clips.append(branch_clip)
                    if(self.log):
                        torch.save(branch_clip, batch_path+"batch_clip.pt")
                    #Generate batch samples
                    batch_samples = self.generateNSamples(save_path=batch_path,
                                                    image=init_images[i], 
                                                    c_i=branch_clip,  
                                                    n=n_i,  
                                                    strength=strength,
                                                    noise_level=noise)
                    #Score batch samples
                    batch_scores, batch_clips = self.score_samples(beta, batch_samples, batch_path)
                    best_batch_images.append(batch_samples[int(torch.argmax(batch_scores))])
                    average_batch_scores.append(float(torch.mean(batch_scores)))
                    best_batch_scores.append(float(torch.max(batch_scores)))
                    #Append scores, samples, and clips to iteration lists
                    scores.append(batch_scores)
                    samples += batch_samples
                    sample_clips += batch_clips
                if cur_iter > 0:
                    if not best_image not in init_images:
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
                
                if best_batch_corrrelation > best_search_corrrelation or best_search_corrrelation == -1:
                    best_search_corrrelation = best_batch_corrrelation
                    best_image = best_batch_images[best_batch_index]
                    best_clip = branch_clips[best_batch_index]
                    cur_clip = slerp(cur_clip, best_clip, momentum)
            
        return best_image, iter_images, iter_scores, var_scores


    
