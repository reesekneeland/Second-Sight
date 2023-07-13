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
from clip_encoder import CLIP_Encoder
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
                ae=True,
                disable_SD=False):
        with torch.no_grad():
            self.subject = subject
            with open("config.yml", "r") as yamlfile:
                self.config = yaml.load(yamlfile, Loader=yaml.FullLoader)[self.subject]
            self.x_size = self.config[modelParams[0]]["x_size"]
            self.modelParams = modelParams
            self.log = log
            self.device = device
            self.n_iter = n_iter
            self.n_samples = n_samples
            self.n_branches = n_branches
            self.ae = ae
            self.hashNum = "SCS_" + "_".join(self.modelParams)
            self.vector = "images"
            if not disable_SD:
                self.R = StableUnCLIPImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16, variation="fp16").to("cuda")
            self.EncModels = []
            self.EncType = []
            
            
            
            if len(modelParams)>1:
                self.AEModel = AutoEncoder(config="dualAutoEncoder",
                                                    inference=True,
                                                    subject=self.subject,
                                                    device=self.device)
                #self.encoderWeights = torch.load("masks/subject{}/{}_encoder_prediction_weights.pt".format(self.subject, "_".join(self.modelParams))).to(self.device)
            else:
                self.AEModel = AutoEncoder(config="gnetAutoEncoder",
                                                    inference=True,
                                                    subject=self.subject,
                                                    device=self.device)
            for param in modelParams:
                self.EncType.append(self.config[param]["vector"])
                if param == "gnetEncoder":
                    self.EncModels.append(GNet8_Encoder(device=self.device,
                                                    subject=self.subject))
                elif param == "clipEncoder":
                    self.EncModels.append(CLIP_Encoder(config="clipEncoder",
                                                    inference=True,
                                                    subject=self.subject,
                                                    device=self.device))


    # Predict using the ensemble of encoding models in the SCS config
    def predict(self, x, mask=None, return_clips=False):
        combined_preds = torch.zeros((len(self.EncType), len(x), self.config[self.modelParams[0]]["x_size"]))
        if(isinstance(x, torch.Tensor)):
            img_list = []
            for i, sample in enumerate(x):
                imagePil = process_image(x[i], 425, 425)
                img_list.append(imagePil)
            x = img_list

        if "clipEncoder" in self.modelParams:
            sample_clips = []
            for sample in x:
                sample_clips.append(self.R.encode_image_raw(sample))
        
        for c, mType in enumerate(self.EncType):
            if mType == "images" or mType == "alexnet_encoder_sub1":
                combined_preds[c] = self.EncModels[c].predict(x, mask).to(self.device)
            elif mType == "c_img_uc":
                combined_preds[c] = self.EncModels[c].predict(torch.stack(sample_clips)[:,0,:], mask).to(self.device)
        if return_clips:
            return torch.mean(combined_preds, dim=0).cpu(), sample_clips
        else:
            return torch.mean(combined_preds, dim=0).cpu()

    # NEEDS TO UPDATED WITH PREDICT() METHOD REFACTOR
    def benchmark_config(self):
        
        with torch.no_grad():
            
            _, _, x_test, _, _, y_test_im, _ = load_nsd(vector="images", 
                                                    loader=False,
                                                    average=False,
                                                    nest=True,
                                                    subject=self.subject)
            _, _, x_test_avg, _, _, y_test_c, _ = load_nsd(vector="c_img_uc", 
                                                    loader=False,
                                                    average=True,
                                                    subject=self.subject)
            # Prepare data
            images = []
            y_test_c = y_test_c.to(self.device)
            for im in y_test_im:
                images.append(process_image(im))
                
            data = {"images": images, "c_img_uc": y_test_c}
            
            #Generate combined predictions
            x_test = x_test.to(self.device)
            x_test_avg = x_test_avg.to(self.device)

            PeC = PearsonCorrCoef().to(self.device)
            PeCFull = PearsonCorrCoef(num_outputs=x_test.shape[0]).to(self.device)
            combined_preds_unweighted = torch.zeros((len(self.modelParams), x_test_avg.shape[0], x_test_avg.shape[1])).to(self.device)
            combined_preds = torch.zeros((len(self.modelParams), x_test_avg.shape[0], x_test_avg.shape[1])).to(self.device)
            for m, model in enumerate(self.EncModels):
                sample = data[self.EncType[m]]
                
                combined_preds_unweighted[m] = model.predict(sample).to(self.device)
                combined_preds[m] = (self.encoderWeights[m] * combined_preds_unweighted[m]).to(self.device)
            combined_preds = torch.sum(combined_preds, dim=0).to(self.device)
            absolute_preds = torch.where(self.encoderWeights[0]>0.5, combined_preds_unweighted[0], combined_preds_unweighted[1])
            print("COMBINED PREDS: {}".format(combined_preds.shape))

            pearson = PeCFull(combined_preds.moveaxis(0,1), x_test_avg.moveaxis(0,1))
            pearson_absolute = PeCFull(absolute_preds.moveaxis(0,1), x_test_avg.moveaxis(0,1))
            
            r = []
            for voxel in range(x_test.shape[1]):
                # Correlation across voxels for a sample (Taking a column)
                r.append(PeC(combined_preds[:,voxel], x_test_avg[:,voxel]).cpu().detach())
            r = np.array(r)
            
            print("Models: {}, Subject: {}".format(", ".join(self.modelParams), self.subject))
            print("Average Vector Correlation: ", float(torch.mean(pearson)))
            print("Absolute Average Vector Correlation: ", float(torch.mean(pearson_absolute)))
            print("Mean Voxel Pearson Avg: ", np.mean(r))
            plt.hist(r, bins=50, log=True)
            plt.savefig("charts/subject{}/{}_combined_encoders_voxel_PeC.png".format(self.subject, "_".join(self.modelParams)))

    def generateNSamples(self, save_path, image, c_i, n, strength=1, noise_level=1):
        images = []
        for i in range(n):
            im = self.R.reconstruct(image=image,
                                             image_embeds=c_i, 
                                             strength=strength,
                                             noise_level=noise_level,
                                             negative_prompt="text, caption")
            images.append(im)
            im.save(save_path + "{}.png".format(i))
            
        return images


    def score_samples(self, beta, images, save_path):
        beta_primes, sample_clips = self.predict(images, return_clips=True)
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
    # n is the number of samples to generate at each iteration
    # max_iter caps the number of iterations it will perform
    def search(self, sample_path, beta, c_i, init_img=None, refine_z=True, refine_clip=True, n=10, max_iter=10, n_branches=4):
        with torch.no_grad():
            best_image, best_clip, cur_clip = init_img, c_i, c_i
            init_clip = [c_i] * n_branches
            init_images = [init_img] * n_branches
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
            for cur_iter in tqdm(range(max_iter), desc="search iterations"):
                iter_path = "{}/iter_{}/".format(sample_path, cur_iter)
                os.makedirs(iter_path, exist_ok=True)
                batch_count = n_branches
                if refine_z:
                    if init_img is None:
                        strength = 1-0.5*(math.pow(cur_iter/max_iter, 3))
                    else:
                        strength = 0.92-0.30*(math.pow((cur_iter+1)/max_iter, 3))
                else:
                    strength = 1
                if refine_clip:
                    momentum = 0.2*(math.pow((cur_iter+1)/max_iter, 2))
                    # noise = int(50-50*(cur_iter/max_iter))
                    # noise = 25
                    noise = 0
                else:
                    momentum = 0
                    noise = 0
                torch.save(torch.tensor(strength), iter_path+"iter_strength.pt")
                n_i = max(10, int((n/n_branches)*strength))
                tqdm.write("Strength: {}, Momentum: {}, Noise: {}, N: {}".format(strength, momentum, noise, n_i))
                
                samples = []
                sample_clips = []
                scores = []

                best_batch_scores = []
                best_batch_images = []
                average_batch_scores = []
                branch_clips = []
                for i in range(n_branches):
                    #Create batch directory
                    batch_path = "{}/batch_{}/".format(iter_path, i)
                    os.makedirs(batch_path, exist_ok=True)
                    #Generate and save batch clip
                    branch_clip = slerp(cur_clip, init_clip[i], momentum)
                    branch_clips.append(branch_clip)
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
                        batch_path = "{}/batch_{}/".format(iter_path, n_branches+1)
                        os.makedirs(batch_path, exist_ok=True)
                        tqdm.write("Adding best image and clip to branches!")
                        #Save branch clip
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
                # print("SCORES SHAPE: ", scores.shape)
                #Set branches for next iteration based on top scores from this iteration, ignoring batches
                topn_pearson = torch.topk(scores, n_branches)
                for i in range(n_branches):
                    if refine_z:
                        init_images[i] = samples[int(topn_pearson.indices[i])]
                    if refine_clip:
                        init_clip[i] = sample_clips[int(topn_pearson.indices[i])]
                best_batch_index = int(np.argmax(average_batch_scores))
                best_batch_corrrelation = float(best_batch_scores[best_batch_index])
                torch.save(torch.tensor(best_batch_index), "{}/best_batch_index.pt".format(iter_path))
                tqdm.write("BC: {}".format(best_batch_corrrelation))
                
                iter_images.append(best_batch_images[int(best_batch_index)])
                iter_scores.append(best_batch_corrrelation)
                var_scores.append(float(torch.var(scores)))
                
                if best_batch_corrrelation > best_search_corrrelation or best_search_corrrelation == -1:
                    best_search_corrrelation = best_batch_corrrelation
                    best_image = best_batch_images[best_batch_index]
                    if refine_clip:
                        best_clip = branch_clips[best_batch_index]
                        cur_clip = slerp(cur_clip, best_clip, momentum)
            
        return best_image, iter_images, iter_scores, var_scores


    
