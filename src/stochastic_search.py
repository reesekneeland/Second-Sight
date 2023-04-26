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
from encoder_uc import Encoder_UC
from alexnet_encoder import AlexNetEncoder
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
                ae=True):
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
            self.R = StableUnCLIPImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16, variation="fp16").to("cuda")
            self.EncModels = []
            self.EncType = []
            
            
            
            if len(modelParams)>1:
                self.AEModel = AutoEncoder(config="gnetAutoEncoder",
                                                    inference=True,
                                                    subject=self.subject,
                                                    device=self.device)
                self.encoderWeights = torch.load("masks/subject{}/{}_encoder_prediction_weights.pt".format(self.subject, "_".join(self.modelParams))).to(self.device)
            else:
                self.AEModel = AutoEncoder(config="gnetAutoEncoder",
                                                    inference=True,
                                                    subject=self.subject,
                                                    device=self.device)
            for param in modelParams:
                self.EncType.append(self.config[param]["vector"])
                #AlexNet only works for subject1
                if param == "alexnetEncoder":
                    self.EncModels.append(AlexNetEncoder(device=self.device))
                elif param == "gnetEncoder":
                    self.EncModels.append(GNet8_Encoder(device=self.device,
                                                    subject=self.subject))
                elif param == "clipEncoder":
                    self.EncModels.append(Encoder_UC(config="clipEncoder",
                                                    inference=True,
                                                    subject=self.subject,
                                                    device=self.device))
            self.masks = {0:torch.full((self.x_size,), False),
                        1:torch.load("masks/subject{}/V1_big.pt".format(self.subject)),
                        2:torch.load("masks/subject{}/V2_big.pt".format(self.subject)),
                        3:torch.load("masks/subject{}/V3_big.pt".format(self.subject)),
                        4:torch.load("masks/subject{}/V4_big.pt".format(self.subject)),
                        5:torch.load("masks/subject{}/early_vis_big.pt".format(self.subject)),
                        6:torch.load("masks/subject{}/higher_vis_big.pt".format(self.subject))}  

    def generate_accuracy_weights(self):
        _, _, x_test_avg, _, _, y_test_c, _ = load_nsd(vector="c_img_uc", 
                                                loader=False,
                                                average=True,
                                                subject=self.subject)
        # Load and compute the prediction weights
        SM = torch.nn.Softmax(dim=0)
        prediction_accuracies = torch.zeros((len(self.modelParams), x_test_avg.shape[1]))
        for m, model in enumerate(self.modelParams):
            prediction_accuracies[m] = torch.load("masks/subject{}/{}_{}_encoder_voxel_PeC.pt".format(self.subject, self.config[model]["hashNum"], self.EncType[m]))
        total_accuracies = torch.sum(prediction_accuracies, dim=0)
        accuracy_ratios = torch.zeros((len(self.modelParams), x_test_avg.shape[1]))
        for i in range(prediction_accuracies.shape[0]):
            accuracy_ratios[i] = (prediction_accuracies[i] / total_accuracies)
        accuracy_probabilities = (SM(accuracy_ratios)).to(self.device)
        torch.save(accuracy_probabilities, "masks/subject{}/{}_encoder_prediction_weights.pt".format(self.subject, "_".join(self.modelParams)))

    # Predict using the ensemble of encoding models in the SCS config
    def predict(self, x, mask=None):
        combined_preds = torch.zeros((len(self.EncType), len(x), self.config[self.modelParams[0]]["x_size"]))
        print("COMBINED PREDS SHAPE: {}".format(combined_preds.shape))
        if "clipEncoder" in self.modelParams:
            sample_clips = []
            for sample in tqdm(x):
                sample_clips.append(self.R.encode_image_raw(Image.fromarray(sample.numpy().reshape((425, 425, 3)).astype(np.uint8)).convert("RGB")))
        
        for c, mType in enumerate(self.EncType):
            if mType == "images" or mType == "alexnet_encoder_sub1":
                combined_preds[c] = self.encoderWeights[c] * self.EncModels[c].predict(x, mask).to(self.device)
            elif mType == "c_img_uc":
                combined_preds[c] = self.encoderWeights[c] * self.EncModels[c].predict(torch.stack(sample_clips)[:,0,:], mask).to(self.device)
        return torch.sum(combined_preds, dim=0).cpu()


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

    def generateNSamples(self, image, c_i, n, strength=1, noise_level=1):
        images = []
        for i in range(n):
            images.append(self.R.reconstruct(image=image,
                                             image_embeds=c_i, 
                                             strength=strength,
                                             noise_level=noise_level,
                                             negative_prompt="text, caption"))
        return images


    # Method to generate image distributions from a given spot in a search (when it crosses threshold)
    # experiment_title: title of experiment to use
    # sample: sample number to use, individual value of idx list generated in experiment
    # iteration: iteration number to generate a distribution at: 
    #   - provide the iteration number for the iteration BEFORE the threshold is crossed
    #   - provide -1 to generate distribution before search is initiated (decoded clip + VDVAE)
    #   - provide last iteration number (5 for searches of 6 iterations) to generate distribution from final state
    # n: number of images to generate in distribution
    # vdvae_override: if True, will generate distribution not using VDVAE for iteration -1, clip only
    def generate_image_distribution(self, experiment_title, sample, iteration, n, vdvae_override=False):
        exp_path = "reconstructions/subject{}/{}/{}/".format(self.subject, experiment_title, sample)
        dist_path = exp_path + "distribution_{}/".format(iteration)
        os.makedirs(dist_path, exist_ok=True)
        contents = os.listdir(dist_path)
        images = []
        if(len(contents)> 0):
            for file in contents:
                images.append(Image.open(os.path.join(dist_path,file)))
            return images
        if iteration == -1:
            if vdvae_override:
                image = None
                strength=1
            else:
                image = Image.open(exp_path+"Decoded VDVAE.png")
                strength=0.9
            c_i = torch.load(exp_path+"decoded_clip.pt")
            
        else:
            image = Image.open(exp_path+"iter_{}.png".format(iteration))
            c_i = torch.load(exp_path+"iter_clip_{}.pt".format(iteration))
            strength = 0.9-0.4*(math.pow(iteration/10, 3))
        images = []
        for i in tqdm(range(n), desc="generating distribution around iteration {}".format(iteration)):
            im = self.R.reconstruct(image=image,
                                    image_embeds=c_i, 
                                    strength=strength,
                                    noise_level=25,
                                    negative_prompt="text, caption")
            images.append(im)
            im.save(dist_path+"/{}.png".format(i))
        return images

    # Main search method
    # c_i is a 1024 clip vector
    # beta is a 3*x_size tensor of brain data to use as guidance targets
    # n is the number of samples to generate at each iteration
    # max_iter caps the number of iterations it will perform
    def search(self, beta, c_i, init_img=None, refine_z=True, refine_clip=True, n=10, max_iter=10, n_branches=1, custom_weighting=False):
        with torch.no_grad():
            best_image, best_clip, cur_clip = init_img, c_i, c_i
            iter_clips = [c_i] * n_branches
            iter_images = [init_img] * n_branches
            images, clips, iter_scores, var_scores = [], [], [], []
            best_vector_corrrelation = -1
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
                if refine_z:
                    if init_img is None:
                        strength = 1-0.4*(math.pow(cur_iter/max_iter, 3))
                    else:
                        strength = 0.9-0.4*(math.pow(cur_iter/max_iter, 3))
                else:
                    strength = 1
                if refine_clip:
                    momentum = 0.15*(math.pow(cur_iter/max_iter, 2))
                    noise = int(50-50*(cur_iter/max_iter))
                    # noise = 25
                else:
                    momentum = 0
                    noise = 0
                n_i = max(10, int((n/n_branches)*strength))
                tqdm.write("Strength: {}, Momentum: {}, Noise: {}, N: {}".format(strength, momentum, noise, n_i))
                
                samples = []
                sample_clips = []
                origin_clips = []
                for i in range(n_branches):
                    branch_clip = slerp(cur_clip, iter_clips[i], momentum)
                    origin_clips += [branch_clip] * n_i
                    samples += self.generateNSamples(image=iter_images[i], 
                                                    c_i=branch_clip,  
                                                    n=n_i,  
                                                    strength=strength,
                                                    noise_level=noise)
                if cur_iter > 0:
                    if not best_image not in iter_images:
                        tqdm.write("Adding best image and clip to branches!")
                        branch_clip = cur_clip
                        origin_clips += [branch_clip] * n_i
                        samples += self.generateNSamples(image=best_image, 
                                                        c_i=branch_clip,
                                                        n=n_i,  
                                                        strength=strength,
                                                        noise_level=noise)
                for image in samples:
                    sample_clips.append(self.R.encode_image_raw(image))
                        
                combined_preds = torch.zeros((len(self.EncType), len(samples), beta.shape[1])).to(self.device)
                # if(self.ae):
                #     autoencoded_betas = torch.zeros((len(self.EncType), beta.shape[1]))
                for c, mType in enumerate(self.EncType):
                    if mType == "images" or mType == "alexnet_encoder_sub1":
                        combined_preds[c] = self.EncModels[c].predict(samples).to(self.device)
                    elif mType == "c_img_uc":
                        combined_preds[c] = self.EncModels[c].predict(torch.stack(sample_clips)[:,0,:]).to(self.device)
                # combined_preds = combined_preds.to(self.device)
                if custom_weighting:
                    repeated_weights = self.encoderWeights.view(self.encoderWeights.shape[0], 1, self.encoderWeights.shape[1]).repeat(1, combined_preds.shape[1], 1)
                    beta_primes = torch.sum(repeated_weights * combined_preds, dim=0)
                    # print("REPEATED WEIGHTS SHAPE: {}".format(repeated_weights.shape))
                    # beta_primes = torch.where(repeated_weights[0]>0.5, combined_preds[0], combined_preds[1])
                else:
                    beta_primes = torch.mean(combined_preds, dim=0)
                beta_primes = beta_primes.moveaxis(0, 1).to(self.device)
                scores = []
                PeC = PearsonCorrCoef(num_outputs=beta_primes.shape[1]).to(self.device) 
                for i in range(beta.shape[0]):
                    xDup = beta[i].repeat(beta_primes.shape[1], 1).moveaxis(0, 1).to(self.device)
                    score = PeC(xDup, beta_primes)
                    scores.append(score)
                scores = torch.mean(torch.stack(scores), dim=0)
                cur_var = float(torch.var(scores))
                topn_pearson = torch.topk(scores, n_branches)
                cur_vector_corrrelation = float(torch.max(scores))
                if(self.log):
                    wandb.log({'Brain encoding pearson correlation_{}'.format(mType): cur_vector_corrrelation, 'score variance_{}'.format(mType): cur_var})
                tqdm.write("Type: {}, VC: {}, Var: {}".format(mType, cur_vector_corrrelation, cur_var))
                best_im_index = int(torch.argmax(scores))
                images.append(samples[best_im_index])
                clips.append(origin_clips[best_im_index])
                iter_scores.append(cur_vector_corrrelation)
                var_scores.append(float(torch.var(scores)))
                for i in range(n_branches):
                    if refine_z:
                        iter_images[i] = samples[int(topn_pearson.indices[i])]
                    if refine_clip:
                        iter_clips[i] = sample_clips[int(topn_pearson.indices[i])]
                if cur_vector_corrrelation > best_vector_corrrelation or best_vector_corrrelation == -1:
                    best_vector_corrrelation = cur_vector_corrrelation
                    best_image = samples[best_im_index]
                    if refine_clip:
                        best_clip = sample_clips[best_im_index]
                        cur_clip = slerp(cur_clip, best_clip, momentum)
            
        return best_image, images, clips, iter_scores, var_scores


    
