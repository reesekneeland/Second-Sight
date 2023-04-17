import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
import torch
from torchmetrics import PearsonCorrCoef
import numpy as np
from PIL import Image
from nsd_access import NSDAccess
from utils import *
import math
import wandb
from tqdm import tqdm
from decoder_uc import Decoder_UC
from encoder_uc import Encoder_UC
from alexnet_encoder import AlexNetEncoder
from autoencoder import AutoEncoder
from diffusers import StableUnCLIPImg2ImgPipeline
from library_decoder import LibraryDecoder
from vdvae import VDVAE


def main():
    # os.chdir("/export/raid1/home/kneel027/Second-Sight/")
    # for i in range(10):
    #     print(int(250-250*(1/(1+math.exp(-((i/10)/0.1 - 5))))))
    # for i in range(10):
    #     print(0.3+0.4*(math.pow(i/10, 2)))
    # for i in range(10):
    #     print(int(500-500*(1/(1+math.exp(-((i/10)/0.1 - 5))))))
    # for i in range(10):
    #     print(0.1+0.8*(math.pow(i/10, 2)))
    # S0 = StochasticSearch(config=["AlexNet"],
    #                       device="cuda:0",
    #                       log=False,
    #                       n_iter=10,
    #                       n_samples=100,
    #                       n_branches=4,
    #                       ae=True)
    # S1 = StochasticSearch(config=["AlexNet"],
    #                       device="cuda:0",
    #                       log=False,
    #                       n_iter=10,
    #                       n_samples=250,
    #                       n_branches=5,
                        #   ae=True)
    # S2 = StochasticSearch(config=["AlexNet"],
    #                       device="cuda:0",
    #                       log=False,
    #                       n_iter=2,
    #                       n_samples=10,
    #                       n_branches=2,
                        #   ae=True)

    # S4 = StochasticSearch(config=["c_img_uc"],
    #                       device="cuda:0",
    #                       log=False,
    #                       n_iter=10,
    #                       n_samples=100,
    #                       n_branches=4,
    #                       ae=True)
    # S5 = StochasticSearch(config=["AlexNet", "c_img_uc"],
    #                       device="cuda:0",
    #                       log=False,
    #                       n_iter=6,
    #                       n_samples=100,
    #                       n_branches=4,
    #                       ae=True)
    S6 = StochasticSearch(config=["AlexNet"],
                          device="cuda:0",
                          log=False,
                          n_iter=6,
                          n_samples=100,
                          n_branches=4,
                          ae=True)
    # S0.generateTestSamples(experiment_title="SCS UC 747 10:100:4 0.4 Exp3 AE", idx=[i for i in range(0, 20)], average=True)
    # S0.generateTestSamples(experiment_title="SCS UC 747 10:100:4 0.5 Exp3 AE", idx=[i for i in range(0, 20)], average=True)
    # S0.generateTestSamples(experiment_title="SCS UC 747 10:100:4 0.6 Exp3 AE photorealistic, negative prompts for text, caption", idx=[i for i in range(0, 20)], average=True)
    # S1.generateTestSamples(experiment_title="SCS UC 10:250:5 0.6 Exp3 AE Fixed", idx=[i for i in range(194, 209)], average=True)
    # S1.generateTestSamples(experiment_title="SCS UC 10:250:5 0.6 Exp3 AE Fixed", idx=[i for i in range(209, 224)], average=True)
    # S1.generateTestSamples(experiment_title="SCS UC 10:250:5 0.6 Exp3 AE Fixed", idx=[i for i in range(224, 239)], average=True)
    # S1.generateTestSamples(experiment_title="SCS UC 10:250:5 0.6 Exp3 AE Fixed", idx=[i for i in range(254, 269)], average=True)
    # S1.generateTestSamples(experiment_title="SCS UC 10:250:5 0.6 Exp3 AE Fixed", idx=[i for i in range(239, 254)], average=True)
    # S1.generateTestSamples(experiment_title="SCS UC 10:250:5 0.6 Exp3 AE Fixed", idx=[i for i in range(269, 284)], average=True)
    # S1.generateTestSamples(experiment_title="SCS UC 10:250:5 0.6 Exp3 AE", idx=[i for i in range(25, 50)], average=True)
    # S1.generateTestSamples(experiment_title="SCS UC 10:250:5 0.6 Exp3 AE", idx=[i for i in range(75, 100)], average=True)
    # S4.generateTestSamples(experiment_title="SCS UC 747 10:100:4 CLIP Guided 22", idx=[i for i in range(0, 20)],  average=True)
    # S4.generateTestSamples(experiment_title="SCS UC 747 10:100:4 CLIP Guided 23", idx=[i for i in range(0, 20)], average=True)
    # S4.generateTestSamples(experiment_title="SCS UC 747 10:100:4 CLIP Guided 24", idx=[i for i in range(0, 20)], average=True)
    # S4.generateTestSamples(experiment_title="SCS UC 747 10:100:4 CLIP Guided 27", idx=[i for i in range(0, 20)], average=True)
    # S5.generateTestSamples(experiment_title="SCS UC 747 10:100:4 Dual Guided 3", idx=[i for i in range(0, 20)], average=True)
    # S5.generateTestSamples(experiment_title="SCS UC 747 6:100:4 Dual Guided Z only 5", idx=[i for i in range(0, 20)], average=True, refine_clip=False)
    # S5.generateTestSamples(experiment_title="SCS UC 747 6:100:4 Dual Guided clip_iter 2", idx=[i for i in range(0, 20)], average=True, refine_clip=True, dual_guided=True)
    S6.generateTestSamples(experiment_title="SCS UC 747 VDVAE Init 6:100:4 4", idx=[i for i in range(0, 20)], average=True)
class StochasticSearch():
    def __init__(self, 
                config=["AlexNet"],
                device="cuda:0",
                subject=1,
                log=True,
                n_iter=10,
                n_samples=10,
                n_branches=1,
                ae=True):
        self.config = config
        self.log = log
        self.device = device
        self.subject = subject
        self.n_iter = n_iter
        self.n_samples = n_samples
        self.n_branches = n_branches
        self.ae = ae
        self.R = StableUnCLIPImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16, variation="fp16")
        self.R = self.R.to("cuda")
        self.R.enable_xformers_memory_efficient_attention()
        self.EncModels = []
        self.EncType = []
        self.AEModels = []
        self.nsda = NSDAccess('/export/raid1/home/surly/raid4/kendrick-data/nsd', '/export/raid1/home/kneel027/nsd_local')
        mask_path = "/export/raid1/home/kneel027/Second-Sight/masks/"
        
        for param in config:
            if param == "AlexNet":
                self.AEModels.append(AutoEncoder(hashNum = "582",
                                                vector="alexnet_encoder_sub1",
                                                subject=self.subject,
                                                encoderHash="579",
                                                log=False, 
                                                device=self.device))
                self.EncModels.append(AlexNetEncoder(device=self.device))
                self.EncType.append("images")
            elif param == "c_img_uc":
                self.AEModels.append(AutoEncoder(hashNum = "743",
                                        vector="c_img_uc",
                                        subject=self.subject,
                                        encoderHash="738",
                                        log=False, 
                                        batch_size=750,
                                        device=self.device))
                self.EncModels.append(Encoder_UC(hashNum="738",
                                                 subject=self.subject,
                                                 vector="c_img_uc",
                                                 device=self.device))
                self.EncType.append("c_img_uc")

    def generateNSamples(self, image, c_i, n, strength=1, noise_level=1):
        images = []
        for i in range(n):
            images.append(self.R.reconstruct(image=image,
                                             image_embeds=c_i, 
                                             strength=strength,
                                             noise_level=noise_level,
                                             negative_prompt="text, caption"))
        return images

    #clip is a 1024 clip vector
    #beta is a 3xdim tensor of brain data to reconstruct
    #n is the number of samples to generate at each iteration
    #max_iter caps the number of iterations it will perform
    def zSearch(self, beta, c_i, init_img=None, n=10, max_iter=10, n_branches=1, mask=None, average=True):
        best_image = None
        iter_images = [init_img] * n_branches
        images, iter_scores, var_scores = [], [], []
        best_vector_corrrelation = -1
        if(self.ae):
            beta = self.AEModels[0].predict(beta)
        #Conglomerate masks
        if(mask):
            beta = beta[:, mask]
        for cur_iter in tqdm(range(max_iter), desc="search iterations"):
            strength = 0.9-0.3*(math.pow(cur_iter/max_iter, 3))
            # strength = 0.95-0.35*(math.pow((cur_iter/max_iter)+0.5, 3))
            n_i = max(10, int((n/n_branches)*strength))
            tqdm.write("Strength: {}, N: {}".format(strength, n_i))
            
            samples = []
            for i in range(n_branches):
                samples += self.generateNSamples(image=iter_images[i], 
                                                c_i=c_i,  
                                                n=n_i,  
                                                strength=strength)
            if not(best_image in iter_images):
                samples += self.generateNSamples(image=best_image, 
                                                c_i=c_i, 
                                                n=n_i,  
                                                strength=strength)
        
            beta_primes = self.EncModels[0].predict(samples, mask)
            beta_primes = beta_primes.moveaxis(0, 1).to(self.device)
            scores = []
            PeC = PearsonCorrCoef(num_outputs=beta_primes.shape[1]).to(self.device) 
            if(average):
                for i in range(beta.shape[0]):
                    xDup = beta[i].repeat(beta_primes.shape[1], 1).moveaxis(0, 1).to(self.device)
                    if(torch.count_nonzero(xDup) > 0):
                        scores.append(PeC(xDup, beta_primes))
                    else:
                        tqdm.write("SKIPPING EMPTY BETA")
                scores = torch.stack(scores)
                scores = torch.mean(scores, dim=0)
            else:
                xDup = beta[0].repeat(beta_primes.shape[1], 1).moveaxis(0, 1).to(self.device)
                scores = PeC(xDup, beta_primes)
            cur_var = float(torch.var(scores))
            topn_pearson = torch.topk(scores, n_branches)
            cur_vector_corrrelation = float(torch.max(scores))
            if(self.log):
                wandb.log({'Alexnet Brain encoding pearson correlation': cur_vector_corrrelation, 'score variance': cur_var})
            tqdm.write("VC: {}, Var: {}".format(cur_vector_corrrelation, cur_var))
            images.append(samples[int(torch.argmax(scores))])
            iter_scores.append(cur_vector_corrrelation)
            var_scores.append(cur_var)
            for i in range(n_branches):
                iter_images[i] = samples[int(topn_pearson.indices[i])]
            if cur_vector_corrrelation > best_vector_corrrelation or best_vector_corrrelation == -1:
                best_vector_corrrelation = cur_vector_corrrelation
                best_image = samples[int(torch.argmax(scores))]
        return best_image, images, iter_scores, var_scores

    def zSearch_clip(self, beta, c_i, n=10, max_iter=10, n_branches=1, mask=None, average=True):
        best_image, best_clip = None, None
        iter_clips = [None] * n_branches
        images, iter_scores, var_scores = [], [], []
        best_vector_corrrelation = -1

        if(self.ae):
            beta = self.AEModels[0].predict(beta)
        if(mask):
            beta = beta[:, mask]
        for cur_iter in tqdm(range(max_iter), desc="search iterations"):
            # momentum = 0.1+0.4*(math.pow(cur_iter/max_iter, 2))
            momentum = 0.05*(math.pow(cur_iter/max_iter, 2))
            # momentum = 0.01
            # noise = int(500-500*(1/(1+math.exp(-((cur_iter/max_iter)/0.1 - 5)))))
            # noise = int(250-250*(math.pow(cur_iter/max_iter, 2)))
            noise = 50

            n_i = max(10, int(n/n_branches))
            tqdm.write("Noise: {}, Momentum: {}, N: {}".format(noise, momentum, n_i))
            samples = []
            sample_clips = []
            for i in range(n_branches):
                if cur_iter > 0:
                    cur_c_i = slerp(cur_c_i, iter_clips[i], momentum)
                else:
                    cur_c_i = c_i
                samples += self.generateNSamples(image=None, 
                                                c_i=cur_c_i,  
                                                n=n_i,  
                                                strength=1,
                                                noise_level=noise)
            
            if cur_iter > 0:
                if not any([torch.equal(best_clip,clip) for clip in iter_clips]):
                    tqdm.write("Adding best image to branches!")
                    cur_c_i = slerp(cur_c_i, best_clip, momentum)
                    samples += self.generateNSamples(image=None, 
                                                    c_i=cur_c_i, 
                                                    n=n_i,  
                                                    strength=1,
                                                    noise_level=noise)

            for image in samples:
                sample_clips.append(self.R.encode_image_raw(image))

            beta_primes = self.EncModels[0].predict(torch.stack(sample_clips)[:,0,:], mask)
            beta_primes = beta_primes.moveaxis(0, 1).to(self.device)
            scores = []
            PeC = PearsonCorrCoef(num_outputs=beta_primes.shape[1]).to(self.device) 
            if(average):
                for i in range(beta.shape[0]):
                    xDup = beta[i].repeat(beta_primes.shape[1], 1).moveaxis(0, 1).to(self.device)
                    if(torch.count_nonzero(xDup) > 0):
                        scores.append(PeC(xDup, beta_primes))
                scores = torch.stack(scores)
                scores = torch.mean(scores, dim=0)
            else:
                xDup = beta[0].repeat(beta_primes.shape[1], 1).moveaxis(0, 1).to(self.device)
                scores = PeC(xDup, beta_primes)
            cur_var = float(torch.var(scores))
            topn_pearson = torch.topk(scores, n_branches)
            cur_vector_corrrelation = float(torch.max(scores))
            if(self.log):
                wandb.log({'Alexnet Brain encoding pearson correlation': cur_vector_corrrelation, 'score variance': cur_var})
            tqdm.write("Brain Correlation: {}, Var: {}".format(cur_vector_corrrelation, cur_var))
            images.append(samples[int(torch.argmax(scores))])
            iter_scores.append(cur_vector_corrrelation)
            var_scores.append(cur_var)
            for i in range(n_branches):
                iter_clips[i] = sample_clips[int(topn_pearson.indices[i])]
            if cur_vector_corrrelation > best_vector_corrrelation or best_vector_corrrelation == -1:
                best_vector_corrrelation = cur_vector_corrrelation
                best_image = samples[int(torch.argmax(scores))]
                best_clip = sample_clips[int(torch.argmax(scores))]
            best_image.save("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/reconstructions/UC Refactor Test5/best_im.png")
        return best_image, images, iter_scores, var_scores
    
    def zSearch_clip_iter(self, beta, c_i, n=10, max_iter=10, n_branches=1, mask=None, average=True):
        best_output, best_image, best_clip = None, None, None
        iter_clips = [None] * n_branches
        iter_images = [None] * n_branches
        images, iter_scores, iter_clip_scores, iter_image_scores, var_scores = [], [], [], [], []
        best_vector_corrrelation, best_image_score, best_clip_score = -1, -1, -1
        
        #Conglomerate masks
        if(mask):
            beta = beta[:, mask]
        for cur_iter in tqdm(range(max_iter), desc="search iterations"):
            strength = 1.0-0.4*(math.pow(cur_iter/max_iter, 3))
            momentum = 0.05*(math.pow(cur_iter/max_iter, 2))
            noise = 50
            n_i = max(10, int((n/n_branches)*strength))
            tqdm.write("Strength: {}, Momentum: {}, Noise: {}, N: {}".format(strength, momentum, noise, n_i))
            
            samples = []
            sample_clips = []
            if cur_iter > 0:
                if not any([torch.equal(best_clip,clip) for clip in iter_clips] or best_image not in iter_images):
                    tqdm.write("Adding best image and clip to branches!")
                    cur_c_i = slerp(cur_c_i, best_clip, momentum)
                    samples += self.generateNSamples(image=best_image, 
                                                    c_i=cur_c_i, 
                                                    n=n_i,  
                                                    strength=strength,
                                                    noise_level=noise)
                cur_c_i = slerp(cur_c_i, iter_clips[i], momentum)
            else:
                cur_c_i = c_i
            for i in range(n_branches):
                samples += self.generateNSamples(image=iter_images[i], 
                                                c_i=cur_c_i,  
                                                n=n_i,  
                                                strength=strength,
                                                noise_level=noise)
            for image in samples:
                sample_clips.append(self.R.encode_image_raw(image))
            combined_scores_list = []
            for c, mType in enumerate(self.EncType):
                if mType == "images":
                    beta_primes = self.EncModels[c].predict(samples, mask)
                elif mType == "c_img_uc":
                    beta_primes = self.EncModels[c].predict(torch.stack(sample_clips)[:,0,:], mask)
                if(self.ae):
                    cur_beta = self.AEModels[c].predict(beta)
                else:
                    cur_beta = beta
                beta_primes = beta_primes.moveaxis(0, 1).to(self.device)
                scores = []
                PeC = PearsonCorrCoef(num_outputs=beta_primes.shape[1]).to(self.device) 
                if(average):
                    for i in range(beta.shape[0]):
                        xDup = cur_beta[i].repeat(beta_primes.shape[1], 1).moveaxis(0, 1).to(self.device)
                        if(torch.count_nonzero(beta[i]) > 0):
                            scores.append(PeC(xDup, beta_primes))
                        else:
                            tqdm.write("SKIPPING EMPTY BETA")
                    scores = torch.stack(scores)
                    scores = torch.mean(scores, dim=0)
                else:
                    xDup = cur_beta[0].repeat(beta_primes.shape[1], 1).moveaxis(0, 1).to(self.device)
                    scores = PeC(xDup, beta_primes)
                combined_scores_list.append(scores)
                cur_var = float(torch.var(scores))
                topn_pearson = torch.topk(scores, n_branches)
                cur_vector_corrrelation = float(torch.max(scores))
                if(self.log):
                    wandb.log({'Alexnet Brain encoding pearson correlation_{}'.format(mType): cur_vector_corrrelation, 'score variance_{}'.format(mType): cur_var})
                tqdm.write("Type: {}, VC: {}, Var: {}".format(mType, cur_vector_corrrelation, cur_var))
                if mType == "images":
                    for i in range(n_branches):
                        iter_images[i] = samples[int(topn_pearson.indices[i])]
                    iter_image_scores.append(float(torch.max(scores)))
                    
                    if (float(torch.max(scores)) > best_image_score or best_image_score == -1):
                        best_image = samples[int(torch.argmax(scores))]
                        best_image_score = int(torch.argmax(scores))
                elif mType == "c_img_uc":
                    for i in range(n_branches):
                        iter_clips[i] = sample_clips[int(topn_pearson.indices[i])]
                    iter_clip_scores.append(float(torch.max(scores)))
                    if (float(torch.max(scores)) > best_clip_score or best_clip_score == -1):
                        best_clip = sample_clips[int(torch.argmax(scores))]
                        best_clip_score = int(torch.argmax(scores))
            combined_scores = torch.mean(torch.stack(combined_scores_list), dim=0)
            combined_best_score = float(torch.max(combined_scores))
            print("COMBINED SCORES SHAPE: {}".format(combined_scores.shape))
            images.append(samples[int(torch.argmax(combined_scores))])
            iter_image_scores.append(float(combined_scores_list[0][int(torch.argmax(combined_scores))]))
            iter_clip_scores.append(float(combined_scores_list[1][int(torch.argmax(combined_scores))]))
            iter_scores.append(combined_best_score)
            var_scores.append(float(torch.var(combined_scores)))
            
            if combined_best_score > best_vector_corrrelation or best_vector_corrrelation == -1:
                best_vector_corrrelation = combined_best_score
                best_output = samples[int(torch.argmax(combined_scores))]
        return best_output, images, iter_scores, iter_clip_scores, iter_image_scores, var_scores

    def zSearch_dual_guidance_z_only(self, beta, c_i, init_img=None, n=10, max_iter=10, n_branches=1, mask=None, average=True):
        best_output = None
        iter_images = [init_img] * n_branches
        images, iter_scores, var_scores = [], [], []
        best_vector_corrrelation = -1
        
        #Conglomerate masks
        if(mask):
            beta = beta[:, mask]
        for cur_iter in tqdm(range(max_iter), desc="search iterations"):
            # strength = 1.0-0.4*(math.pow(cur_iter/max_iter, 3))
            strength = 0.9-0.3*(math.pow(cur_iter/max_iter, 3))
            noise = 25
            n_i = max(10, int((n/n_branches)*strength))
            tqdm.write("Strength: {}, Noise: {}, N: {}".format(strength, noise, n_i))
            
            samples = []
            sample_clips = []
            if cur_iter > 0:
                if best_output not in iter_images:
                    tqdm.write("Adding best image and clip to branches!")
                    samples += self.generateNSamples(image=best_output, 
                                                    c_i=c_i, 
                                                    n=n_i,  
                                                    strength=strength,
                                                    noise_level=noise)
            for i in range(n_branches):
                samples += self.generateNSamples(image=iter_images[i], 
                                                c_i=c_i,  
                                                n=n_i,  
                                                strength=strength,
                                                noise_level=noise)
            for image in samples:
                sample_clips.append(self.R.encode_image_raw(image))
            combined_scores = []
            for c, mType in enumerate(self.EncType):
                if mType == "images":
                    beta_primes = self.EncModels[c].predict(samples, mask)
                elif mType == "c_img_uc":
                    beta_primes = self.EncModels[c].predict(torch.stack(sample_clips)[:,0,:], mask)
                if(self.ae):
                    cur_beta = self.AEModels[c].predict(beta)
                else:
                    cur_beta = beta
                beta_primes = beta_primes.moveaxis(0, 1).to(self.device)
                scores = []
                PeC = PearsonCorrCoef(num_outputs=beta_primes.shape[1]).to(self.device) 
                if(average):
                    for i in range(beta.shape[0]):
                        xDup = cur_beta[i].repeat(beta_primes.shape[1], 1).moveaxis(0, 1).to(self.device)
                        if(torch.count_nonzero(beta[i]) > 0):
                            scores.append(PeC(xDup, beta_primes))
                        else:
                            tqdm.write("SKIPPING EMPTY BETA")
                    scores = torch.stack(scores)
                    scores = torch.mean(scores, dim=0)
                else:
                    xDup = cur_beta[0].repeat(beta_primes.shape[1], 1).moveaxis(0, 1).to(self.device)
                    scores = PeC(xDup, beta_primes)
                combined_scores.append(scores)
                cur_var = float(torch.var(scores))
                topn_pearson = torch.topk(scores, n_branches)
                cur_vector_corrrelation = float(torch.max(scores))
                if(self.log):
                    wandb.log({'Alexnet Brain encoding pearson correlation_{}'.format(mType): cur_vector_corrrelation, 'score variance_{}'.format(mType): cur_var})
                tqdm.write("Type: {}, VC: {}, Var: {}".format(mType, cur_vector_corrrelation, cur_var))
            combined_scores = torch.mean(torch.stack(combined_scores), dim=0)
            cur_vector_corrrelation = float(torch.max(combined_scores))
            cur_var = float(torch.var(combined_scores))
            tqdm.write("Type: Combined, VC: {}, Var: {}".format(cur_vector_corrrelation, cur_var))
            images.append(samples[int(torch.argmax(combined_scores))])
            iter_scores.append(cur_vector_corrrelation)
            var_scores.append(float(torch.var(combined_scores)))
            topn_pearson = torch.topk(combined_scores, n_branches)
            for i in range(n_branches):
                iter_images[i] = samples[int(topn_pearson.indices[i])]
            if cur_vector_corrrelation > best_vector_corrrelation or best_vector_corrrelation == -1:
                best_vector_corrrelation = cur_vector_corrrelation
                best_output = samples[int(torch.argmax(combined_scores))]
        return best_output, images, iter_scores, var_scores
    
    def zSearch_dual_guidance_clip_iter(self, beta, c_i, init_img=None, n=10, max_iter=10, n_branches=1, mask=None, average=True):
        best_output, best_clip = None, None
        iter_clips = [None] * n_branches
        iter_images = [init_img] * n_branches
        images, iter_scores, iter_clip_scores, iter_image_scores, var_scores = [], [], [], [], []
        best_vector_corrrelation, best_clip_score = -1, -1
        
        #Apply masks
        if(mask):
            beta = beta[:, mask]
        for cur_iter in tqdm(range(max_iter), desc="search iterations"):
            # strength = 1.0-0.4*(math.pow(cur_iter/max_iter, 3))
            strength = 0.9-0.3*(math.pow(cur_iter/max_iter, 3))
            momentum = 0.05*(math.pow(cur_iter/max_iter, 2))
            noise = 25
            n_i = max(10, int((n/n_branches)*strength))
            tqdm.write("Strength: {}, Momentum: {}, Noise: {}, N: {}".format(strength, momentum, noise, n_i))
            
            samples = []
            sample_clips = []
            if cur_iter > 0:
                if not any([torch.equal(best_clip,clip) for clip in iter_clips] or best_output not in iter_images):
                    tqdm.write("Adding best image and clip to branches!")
                    cur_c_i = slerp(cur_c_i, best_clip, momentum)
                    samples += self.generateNSamples(image=best_output, 
                                                    c_i=cur_c_i, 
                                                    n=n_i,  
                                                    strength=strength,
                                                    noise_level=noise)
                cur_c_i = slerp(cur_c_i, iter_clips[i], momentum)
            else:
                cur_c_i = c_i
            for i in range(n_branches):
                samples += self.generateNSamples(image=iter_images[i], 
                                                c_i=cur_c_i,  
                                                n=n_i,  
                                                strength=strength,
                                                noise_level=noise)
            for image in samples:
                sample_clips.append(self.R.encode_image_raw(image))
            combined_scores_list = []
            for c, mType in enumerate(self.EncType):
                if mType == "images":
                    beta_primes = self.EncModels[c].predict(samples, mask)
                elif mType == "c_img_uc":
                    beta_primes = self.EncModels[c].predict(torch.stack(sample_clips)[:,0,:], mask)
                if(self.ae):
                    cur_beta = self.AEModels[c].predict(beta)
                else:
                    cur_beta = beta
                beta_primes = beta_primes.moveaxis(0, 1).to(self.device)
                scores = []
                PeC = PearsonCorrCoef(num_outputs=beta_primes.shape[1]).to(self.device) 
                if(average):
                    for i in range(beta.shape[0]):
                        xDup = cur_beta[i].repeat(beta_primes.shape[1], 1).moveaxis(0, 1).to(self.device)
                        if(torch.count_nonzero(beta[i]) > 0):
                            scores.append(PeC(xDup, beta_primes))
                        else:
                            tqdm.write("SKIPPING EMPTY BETA")
                    scores = torch.stack(scores)
                    scores = torch.mean(scores, dim=0)
                else:
                    xDup = cur_beta[0].repeat(beta_primes.shape[1], 1).moveaxis(0, 1).to(self.device)
                    scores = PeC(xDup, beta_primes)
                combined_scores_list.append(scores)
                cur_var = float(torch.var(scores))
                topn_pearson = torch.topk(scores, n_branches)
                cur_vector_corrrelation = float(torch.max(scores))
                if(self.log):
                    wandb.log({'Alexnet Brain encoding pearson correlation_{}'.format(mType): cur_vector_corrrelation, 'score variance_{}'.format(mType): cur_var})
                tqdm.write("Type: {}, VC: {}, Var: {}".format(mType, cur_vector_corrrelation, cur_var))
                if mType == "c_img_uc":
                    for i in range(n_branches):
                        iter_clips[i] = sample_clips[int(topn_pearson.indices[i])]
                    iter_clip_scores.append(float(torch.max(scores)))
                    if (float(torch.max(scores)) > best_clip_score or best_clip_score == -1):
                        best_clip = sample_clips[int(torch.argmax(scores))]
                        best_clip_score = int(torch.argmax(scores))
            combined_scores = torch.mean(torch.stack(combined_scores_list), dim=0)
            topn_pearson = torch.topk(combined_scores, n_branches)
            combined_best_score = float(torch.max(combined_scores))
            images.append(samples[int(torch.argmax(combined_scores))])
            iter_image_scores.append(float(combined_scores_list[0][int(torch.argmax(combined_scores))]))
            iter_clip_scores.append(float(combined_scores_list[1][int(torch.argmax(combined_scores))]))
            iter_scores.append(combined_best_score)
            var_scores.append(float(torch.var(combined_scores)))
            for i in range(n_branches):
                iter_images[i] = samples[int(topn_pearson.indices[i])]
            if combined_best_score > best_vector_corrrelation or best_vector_corrrelation == -1:
                best_vector_corrrelation = combined_best_score
                best_output = samples[int(torch.argmax(combined_scores))]
        return best_output, images, iter_scores, iter_clip_scores, iter_image_scores, var_scores


    def generateTestSamples(self, experiment_title, idx, mask=[], ae=False, average=True, library=False, average_clips=False, refine_clip=True, dual_guided=False):    

        os.makedirs("reconstructions/{}/".format(experiment_title), exist_ok=True)
        os.makedirs("logs/{}/".format(experiment_title), exist_ok=True)

         # Load data and targets
        _, _, x_test, _, _, targets_c_i, trials = load_nsd(vector="c_img_uc", subject=self.subject, loader=False, average=False, nest=True)
        _, _, x_test_averaged, _, _, _, _ = load_nsd(vector="c_img_uc", subject=self.subject, loader=False, average=True, nest=False)
        _, _, _, _, _, targets_vdvae, _ = load_nsd(vector="z_vdvae", subject=self.subject, loader=False, average=True, nest=False)
            
        targets_c_i = targets_c_i[idx]
        x_test = x_test[idx]
        outputs_c_i = [None] * len(idx)
        
        Dv = Decoder_UC(hashNum = "764",
                        vector="z_vdvae",
                        subject=self.subject, 
                        log=False, 
                        device="cuda",
                        )
        outputs_vdvae = Dv.predict(x=x_test_averaged[idx])
        del Dv
        
        latent_mean = torch.load("vdvae/train_mean.pt").to("cuda")
        latent_std = torch.load("vdvae/train_std.pt").to("cuda")
        outputs_vdvae = (outputs_vdvae - torch.mean(outputs_vdvae, dim=0)) / torch.std(outputs_vdvae, dim=0)
        outputs_vdvae = outputs_vdvae * latent_std + latent_mean
        outputs_vdvae = outputs_vdvae.reshape((len(idx), 1, 91168))
        
        targets_vdvae = targets_vdvae[idx].to("cuda")
        targets_vdvae = (targets_vdvae - torch.mean(targets_vdvae,dim=0)) / torch.std(targets_vdvae,dim=0)
        targets_vdvae = targets_vdvae * latent_std + latent_mean
        targets_vdvae = targets_vdvae.reshape((len(idx), 1, 91168))
        V = VDVAE()
        
        if library:
            LD = LibraryDecoder(vector="c_img_uc",
                                config=["AlexNet"],
                                device="cuda:0")
            outputs_c_i, _ = LD.predictVector_coco(x_test, average=average)
            outputs_c_i = torch.mean(outputs_c_i, dim=1)
            print(outputs_c_i.shape)
        else:
            Dc_i = Decoder_UC(hashNum = "747",
                              subject=self.subject,
                              vector="c_img_uc", 
                              log=False, 
                              device="cuda")
            outputs_c_i = Dc_i.predict(x=x_test_averaged[idx])
            print(outputs_c_i.shape)
            del Dc_i
                
        PeC = PearsonCorrCoef(num_outputs=len(idx)).to("cpu")
        PeC1 = PearsonCorrCoef(num_outputs=1).to("cpu")
        #Log the CLIP scores
        clip_scores = np.array(PeC(outputs_c_i.moveaxis(0,1).to("cpu"), targets_c_i.moveaxis(0,1).to("cpu")).detach())
        np.save("logs/{}/decoded_c_img_PeC.npy".format(experiment_title), clip_scores)
        scs_c_i = []
        
        for i, val in enumerate(tqdm(idx, desc="Reconstructing samples")):
            os.makedirs("reconstructions/{}/{}/".format(experiment_title, val), exist_ok=True)
            
            if(self.log):
                wandb.init(
                    # set the wandb project where this run will be logged
                    project="StochasticSearch",
                    # track hyperparameters and run metadata
                    config={
                    "experiment": experiment_title,
                    "sample": val-194,
                    "masks": mask,
                    "n_iter": self.n_iter,
                    "n_samples": self.n_samples
                    }
                )
            output_v = V.reconstruct(outputs_vdvae[i])
            target_v = V.reconstruct(targets_vdvae[i])
            output_c = self.R.reconstruct(image_embeds=outputs_c_i[i], negative_prompt="text, caption", strength=1.0)
            target_c = self.R.reconstruct(image_embeds=targets_c_i[i], negative_prompt="text, caption", strength=1.0)
            output_cv = self.R.reconstruct(image=output_v, image_embeds=outputs_c_i[i], negative_prompt="text, caption", strength=0.9)
            target_cv = self.R.reconstruct(image=target_v, image_embeds=targets_c_i[i], negative_prompt="text, caption", strength=0.9)
            
            if len(self.config ) > 1:
                if refine_clip and dual_guided:
                    scs_reconstruction, image_list, score_list, clip_score_list, image_score_list, var_list = self.zSearch_dual_guidance_clip_iter(beta=x_test[i], c_i=outputs_c_i[i], init_img=output_v, n=self.n_samples, max_iter=self.n_iter, n_branches=self.n_branches, mask=mask, average=average)
                    np.save("logs/{}/{}_clip_score_list.npy".format(experiment_title, val), np.array(clip_score_list))
                    np.save("logs/{}/{}_image_score_list.npy".format(experiment_title, val), np.array(image_score_list))
                elif refine_clip:
                    scs_reconstruction, image_list, score_list, clip_score_list, image_score_list, var_list = self.zSearch_clip_iter(beta=x_test[i], c_i=outputs_c_i[i], n=self.n_samples, max_iter=self.n_iter, n_branches=self.n_branches, mask=mask, average=average)
                    np.save("logs/{}/{}_clip_score_list.npy".format(experiment_title, val), np.array(clip_score_list))
                    np.save("logs/{}/{}_image_score_list.npy".format(experiment_title, val), np.array(image_score_list))
                else:
                    scs_reconstruction, image_list, score_list, var_list = self.zSearch_dual_guidance_z_only(beta=x_test[i], c_i=outputs_c_i[i], init_img=output_v, n=self.n_samples, max_iter=self.n_iter, n_branches=self.n_branches, mask=mask, average=average)
                    clip_score_list = score_list
                    image_score_list = score_list
            elif self.config[0] == "AlexNet":
                scs_reconstruction, image_list, score_list, var_list = self.zSearch(beta=x_test[i], c_i=outputs_c_i[i], init_img=output_v, n=self.n_samples, max_iter=self.n_iter, n_branches=self.n_branches, mask=mask, average=average)
                clip_score_list = score_list
                image_score_list = score_list
            elif self.config[0] == "c_img_uc":
                scs_reconstruction, image_list, score_list, var_list = self.zSearch_clip(beta=x_test[i], c_i=outputs_c_i[i], n=self.n_samples, max_iter=self.n_iter, n_branches=self.n_branches, mask=mask, average=average)
                clip_score_list = score_list
                image_score_list = score_list
            #log the data to a file
            np.save("logs/{}/{}_score_list.npy".format(experiment_title, val), np.array(score_list))
            np.save("logs/{}/{}_var_list.npy".format(experiment_title, val), np.array(var_list))
            scs_c_i.append(self.R.encode_image_raw(scs_reconstruction).reshape((1024,)))
            new_clip_score = float(PeC1(scs_c_i[i].to("cpu"), targets_c_i[i].to("cpu").detach()))
            tqdm.write("CLIP IMPROVEMENT: {} -> {}".format(clip_scores[i], new_clip_score))
            nsdId = trials[val]
            ground_truth_np_array = self.nsda.read_images([nsdId], show=True)
            ground_truth = Image.fromarray(ground_truth_np_array[0])
            ground_truth = ground_truth.resize((768, 768), resample=Image.Resampling.LANCZOS)
            rows = int(math.ceil(len(image_list)/2 + 2))
            columns = 2
            images = [ground_truth, scs_reconstruction, target_v, output_v, target_c, output_c, target_cv, output_cv]
            captions = ["Ground Truth", "Search Reconstruction", "Ground Truth VDVAE", "Decoded VDVAE", "Ground Truth CLIP", "Decoded CLIP Only", "Ground Truth CLIP+VDVAE", "Decoded CLIP+VDVAE"]
            for j in range(len(image_list)):
                images.append(image_list[j])
                captions.append("Comb: {} CLIP: {} AN: {}".format(round(score_list[j], 3), round(clip_score_list[j], 3), round(image_score_list[j], 3)))
            figure = tileImages("{}:{}".format(experiment_title, val), images, captions, rows, columns)
            if(self.log):
                wandb.finish()
            
            figure.save('reconstructions/{}/{}.png'.format(experiment_title, val))
            try:
                count = 0
                for j in range(len(images)):
                    if("BC" in captions[j] or "Comb" in captions[j]):
                        images[j].save("reconstructions/{}/{}/iter_{}.png".format(experiment_title, val, count))
                        count +=1
                    else:
                        images[j].save("reconstructions/{}/{}/{}.png".format(experiment_title, val, captions[j]))
            except:
                pass
        scs_c_i = torch.stack(scs_c_i)
        scs_clip_scores = np.array(PeC(scs_c_i.moveaxis(0,1).to("cpu"), targets_c_i.moveaxis(0,1).to("cpu")).detach())
        np.save("logs/{}/scs_c_img_PeC.npy".format(experiment_title), scs_clip_scores)
if __name__ == "__main__":
    main()
