import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
import torch
import numpy as np
from PIL import Image
from nsd_access import NSDAccess
from pycocotools.coco import COCO
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
from random import randrange



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
    #                       n_branches=4)
    S1 = StochasticSearch(config=["AlexNet"],
                          device="cuda:0",
                          log=False,
                          n_iter=10,
                          n_samples=250,
                          n_branches=5)
    # S2 = StochasticSearch(config=["AlexNet"],
    #                       device="cuda:0",
    #                       log=False,
    #                       n_iter=2,
    #                       n_samples=10,
    #                       n_branches=2)
    # S4 = StochasticSearch(config=["c_img_uc"],
    #                       device="cuda:0",
    #                       log=False,
    #                       n_iter=10,
    #                       n_samples=100,
    #                       n_branches=4)
    # S0.generateTestSamples(experiment_title="SCS UC 747 10:100:4 0.4 Exp3 AE", idx=[i for i in range(0, 20)], mask=[], ae=True, test=False, average=True)
    # S0.generateTestSamples(experiment_title="SCS UC 747 10:100:4 0.5 Exp3 AE", idx=[i for i in range(0, 20)], mask=[], ae=True, test=False, average=True)
    # S0.generateTestSamples(experiment_title="SCS UC 747 10:100:4 0.6 Exp3 AE", idx=[i for i in range(0, 20)], mask=[], ae=True, test=False, average=True)
    S1.generateTestSamples(experiment_title="SCS UC 10:250:5 0.6 Exp3 AE", idx=[i for i in range(7, 25)], mask=[], ae=True, test=True, average=True)
    S1.generateTestSamples(experiment_title="SCS UC 10:250:5 0.6 Exp3 AE", idx=[i for i in range(50, 75)], mask=[], ae=True, test=True, average=True)
    # S1.generateTestSamples(experiment_title="SCS UC 10:250:5 0.6 Exp3 AE", idx=[i for i in range(25, 50)], mask=[], ae=True, test=True, average=True)
    # S1.generateTestSamples(experiment_title="SCS UC 10:250:5 0.6 Exp3 AE", idx=[i for i in range(75, 100)], mask=[], ae=True, test=True, average=True)
    # S4.generateTestSamples(experiment_title="SCS UC 747 10:100:4 CLIP Guided 6", idx=[i for i in range(0, 20)], mask=[], ae=True, test=False, average=True)
    # S4.generateTestSamples(experiment_title="SCS UC 747 10:100:4 CLIP Guided 7", idx=[i for i in range(0, 20)], mask=[], ae=True, test=False, average=True)

class StochasticSearch():
    def __init__(self, 
                config=["AlexNet"],
                device="cuda:0",
                log=True,
                n_iter=10,
                n_samples=10,
                n_branches=1):
        self.config = config
        self.log = log
        self.device = device
        self.n_iter = n_iter
        self.n_samples = n_samples
        self.n_branches = n_branches
        self.R = StableUnCLIPImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16, variation="fp16")
        self.R = self.R.to("cuda")
        self.R.enable_xformers_memory_efficient_attention()
        self.EncModels = []
        self.EncType = []
        self.AEModels = []
        self.nsda = NSDAccess('/export/raid1/home/surly/raid4/kendrick-data/nsd', '/export/raid1/home/kneel027/nsd_local')
        mask_path = "/export/raid1/home/kneel027/Second-Sight/masks/"
        self.masks = {0:torch.full((11838,), False),
                      1:torch.load(mask_path + "V1.pt"),
                      2:torch.load(mask_path + "V2.pt"),
                      3:torch.load(mask_path + "V3.pt"),
                      4:torch.load(mask_path + "V4.pt"),
                      5:torch.load(mask_path + "V5.pt"),
                      6:torch.load(mask_path + "V6.pt"),
                      7:torch.load(mask_path + "V7.pt")}
        for param in config:
            if param == "AlexNet":
                self.AEModels.append(AutoEncoder(hashNum = "582",
                                                vector="alexnet_encoder_sub1", 
                                                encoderHash="579",
                                                log=False, 
                                                device=self.device))
                self.EncModels.append(AlexNetEncoder(device=self.device))
                self.EncType.append("images")
            elif param == "c_img_uc":
                self.AEModels.append(AutoEncoder(hashNum = "743",
                                        vector="c_img_uc", #c_img_0, c_text_0, z_img_mixer
                                        encoderHash="738",
                                        log=False, 
                                        batch_size=750,
                                        device=self.device))
                self.EncModels.append(Encoder_UC(hashNum="738",
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
                                             prompt="photorealistic", 
                                             negative_prompt="cartoon, art, saturated, text, caption"))
        return images

    #clip is a 5x768 clip vector
    #beta is a 3x11838 tensor of brain data to reconstruct
    #cross validate says whether to cross validate between scans
    #n is the number of samples to generate at each iteration
    #max_iter caps the number of iterations it will perform
    def zSearch(self, beta, c_i, n=10, max_iter=10, n_branches=1, mask=[], average=True):
        best_image = None
        iter_images = [None] * n_branches
        images, iter_scores, var_scores = [], [], []
        best_vector_corrrelation, best_var = -1, -1
        loss_counter = 0
        
        #Conglomerate masks
        if(len(mask)>0):
            beta_mask = self.masks[0]
            for i in mask:
                beta_mask = torch.logical_or(beta_mask, self.masks[i])
                
            beta = beta[:, beta_mask]
        for cur_iter in tqdm(range(max_iter), desc="search iterations"):
            # if(loss_counter > 3):
            #     break
            # strength = 1.0-0.5*(cur_iter/max_iter)
            strength = 1.0-0.4*(math.pow(cur_iter/max_iter, 3))
            n_i = max(10, int((n/n_branches)*strength))
            tqdm.write("Strength: " + str(strength) + ", N: " + str(n_i))
            
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
            tqdm.write("VC: " + str(cur_vector_corrrelation) + ", Var: " + str(cur_var))
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
        if(mask):
            beta = beta[:, mask]
        for cur_iter in tqdm(range(max_iter), desc="search iterations"):
            # momentum = 0.1+0.4*(math.pow(cur_iter/max_iter, 2))
            # momentum = 0.1+0.8*(math.pow(cur_iter/max_iter, 2))
            momentum = 0.2
            # noise = int(500-500*(1/(1+math.exp(-((cur_iter/max_iter)/0.1 - 5)))))
            # noise = int(200-200*(1/(1+math.exp(-((cur_iter/max_iter)/0.1 - 5)))))
            noise = 100
            n_i = max(10, int(n/n_branches))
            tqdm.write("Noise: " + str(noise) + ", Momentum: " + str(momentum) +", N: " + str(n_i))
            samples = []
            sample_clips = []
            for i in range(n_branches):
                if cur_iter > 0:
                    c_i = slerp(c_i, iter_clips[i], momentum)
                samples += self.generateNSamples(image=None, 
                                                c_i=c_i,  
                                                n=n_i,  
                                                strength=1,
                                                noise_level=noise)
            
            if cur_iter > 0:
                if not any([torch.equal(best_clip,clip) for clip in iter_clips]):
                    tqdm.write("Adding best image to branches!")
                    c_i = slerp(c_i, best_clip, momentum)
                    samples += self.generateNSamples(image=None, 
                                                    c_i=c_i, 
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
            tqdm.write("Brain Correlation: " + str(cur_vector_corrrelation) + ", Var: " + str(cur_var))
            images.append(samples[int(torch.argmax(scores))])
            iter_scores.append(cur_vector_corrrelation)
            var_scores.append(cur_var)
            for i in range(n_branches):
                iter_clips[i] = sample_clips[int(topn_pearson.indices[i])]
            if cur_vector_corrrelation > best_vector_corrrelation or best_vector_corrrelation == -1:
                best_vector_corrrelation = cur_vector_corrrelation
                best_image = samples[int(torch.argmax(scores))]
                best_clip = sample_clips[int(torch.argmax(scores))]
        return best_image, images, iter_scores, var_scores


    def generateTestSamples(self, experiment_title, idx, mask=[], ae=False, test=True, average=True, library=False):    

        os.makedirs("reconstructions/" + experiment_title + "/", exist_ok=True)
        os.makedirs("logs/" + experiment_title + "/", exist_ok=True)

         # Load data and targets
        if test:
            _, _, _, x, _, _, _, targets_c_i, _, trials = load_nsd(vector="c_img_uc", loader=False, average=False, nest=True)
        else:
            _, _, x, _, _, _, targets_c_i, _, trials, _ = load_nsd(vector="c_img_uc", loader=False, average=False, nest=True)
            
        targets_c_i = targets_c_i[idx]

        if(ae):
            x_pruned_ae = torch.zeros((len(idx), 3, 11838))
        x_pruned = torch.zeros((len(idx), 3, 11838))
        for i, index in enumerate(tqdm(idx, desc="Pruning and autoencoding samples")):
            x_pruned[i] = x[index]
            if(ae):
                x_pruned_ae[i] = self.AEModels[0].predict(x_pruned[i])
        x = x_pruned
        outputs_c_i = [None] * len(idx)
        
        if library:
            LD = LibraryDecoder(vector="c_img_uc",
                        config=["AlexNet"],
                        device="cuda:0")
            outputs_c_i, _ = LD.predictVector_coco(x, average=average)
            outputs_c_i = torch.mean(outputs_c_i, dim=1)
            print(outputs_c_i.shape)
        else:
            Dc_i = Decoder_UC(hashNum = "747",
                    vector="c_img_uc", 
                    log=False, 
                    device="cuda",
                    )
            outputs_c_i = Dc_i.predict(x=torch.mean(x, dim=1))
            del Dc_i
        if(ae):
            x = x_pruned_ae
                
        PeC = PearsonCorrCoef(num_outputs=len(idx)).to("cpu")
        PeC1 = PearsonCorrCoef(num_outputs=1).to("cpu")
        #Log the CLIP scores
        clip_scores = np.array(PeC(outputs_c_i.moveaxis(0,1).to("cpu"), targets_c_i.moveaxis(0,1).to("cpu")).detach())
        np.save("logs/" + experiment_title + "/" + "decoded_c_img_PeC.npy", clip_scores)
        # np.save("logs/" + experiment_title + "/" + "c_text_PeC.npy", np.array(PeC(outputs_c_t.moveaxis(0,1).to("cpu"), targets_c_t[idx].moveaxis(0,1).to("cpu")).detach()))
        scs_c_i = []
        for i, val in enumerate(tqdm(idx, desc="Reconstructing samples")):
            os.makedirs("reconstructions/" + experiment_title + "/" + str(val) + "/", exist_ok=True)
            
            if(self.log):
                wandb.init(
                    # set the wandb project where this run will be logged
                    project="StochasticSearch",
                    # track hyperparameters and run metadata
                    config={
                    "experiment": experiment_title,
                    "sample": val,
                    "masks": mask,
                    "n_iter": self.n_iter,
                    "n_samples": self.n_samples
                    }
                )
            reconstructed_output_c = self.R.reconstruct(image_embeds=outputs_c_i[i], prompt="photorealistic", negative_prompt="cartoon, art, saturated, text, caption", strength=1.0)
            reconstructed_target_c = self.R.reconstruct(image_embeds=targets_c_i[i], prompt="photorealistic", negative_prompt="cartoon, art, saturated, text, caption", strength=1.0)
            if self.config[0] == "AlexNet":
                scs_reconstruction, image_list, score_list, var_list = self.zSearch(beta=x[i], c_i=outputs_c_i[i], n=self.n_samples, max_iter=self.n_iter, n_branches=self.n_branches, mask=mask, average=average)
            elif self.config[0] == "c_img_uc":
                scs_reconstruction, image_list, score_list, var_list = self.zSearch_clip(beta=x[i], c_i=outputs_c_i[i], n=self.n_samples, max_iter=self.n_iter, n_branches=self.n_branches, mask=mask, average=average)
            #log the data to a file
            np.save("logs/" + experiment_title + "/" + str(val) + "_score_list.npy", np.array(score_list))
            np.save("logs/" + experiment_title + "/" + str(val) + "_var_list.npy", np.array(var_list))
            scs_c_i.append(self.R.encode_image_raw(scs_reconstruction).reshape((1024,)))
            tqdm.write("CLIP IMPROVEMENT: " + str(clip_scores[i]) + " -> " + str(float(PeC1(scs_c_i[i].to("cpu"), targets_c_i[i].to("cpu").detach()))))
        #     # returns a numpy array 
            nsdId = trials[val]
            ground_truth_np_array = self.nsda.read_images([nsdId], show=True)
            ground_truth = Image.fromarray(ground_truth_np_array[0])
            ground_truth = ground_truth.resize((768, 768), resample=Image.Resampling.LANCZOS)
            rows = int(math.ceil(len(image_list)/2 + 2))
            columns = 2
            images = [ground_truth, scs_reconstruction, reconstructed_target_c, reconstructed_output_c]
            captions = ["Ground Truth", "Search Reconstruction", "Ground Truth CLIP", "Decoded CLIP Only"]
            for j in range(len(image_list)):
                images.append(image_list[j])
                captions.append("BC: " + str(round(score_list[j], 3)) + " VAR: " + str(round(var_list[j], 3)))
            figure = tileImages(experiment_title + ": " + str(val), images, captions, rows, columns)
            if(self.log):
                wandb.finish()
            
            figure.save('reconstructions/' + experiment_title + '/' + str(val) + '.png')
            try:
                count = 0
                for j in range(len(images)):
                    if("BC" in captions[j]):
                        images[j].save('reconstructions/' + experiment_title + "/" + str(val) + "/iter_" + str(count) + '.png')
                        count +=1
                    else:
                        images[j].save('reconstructions/' + experiment_title + "/" + str(val) + "/" + captions[j] + '.png')
            except:
                pass
        scs_c_i = torch.stack(scs_c_i)
        scs_clip_scores = np.array(PeC(scs_c_i.moveaxis(0,1).to("cpu"), targets_c_i.moveaxis(0,1).to("cpu")).detach())
        np.save("logs/" + experiment_title + "/" + "scs_c_img_PeC.npy", scs_clip_scores)
if __name__ == "__main__":
    main()
