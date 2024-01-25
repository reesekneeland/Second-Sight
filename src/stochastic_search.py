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
from reconstructor import Reconstructor
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
                self.R = Reconstructor(which='v1.0', fp16=True, device=self.device)     
            
            
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
    # def predict(self, x, mask=None):
        
    #     if(isinstance(x, list)):
    #         img_tensor = torch.zeros((len(x), 425, 425, 3))
    #         for i, sample in enumerate(x):
    #             image = sample.resize((425, 425))
    #             img_tensor[i] = torch.from_numpy(np.array(image)).reshape((425, 425, 3))
    #         x = img_tensor
    #     elif(isinstance(x, torch.Tensor)):
    #         assert 425 in x.shape or 541875 in x.shape,"Tensor of wrong size"
    #         x = x.reshape((x.shape[0], 425, 425, 3))
    #     else:
    #         raise TypeError

    #     combined_preds = self.EncModels[0].predict(x, mask).cpu()
        
    #     return combined_preds

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
                                    c_i=c_i, 
                                    strength=strength)
            images.append(im)
        return images


    def score_samples(self, beta, images, save_path):
        beta_primes = self.EncModels[0].predict(images)
        if(self.log):
            os.makedirs(save_path + "beta_primes/", exist_ok=True)
            os.makedirs(save_path + "images/", exist_ok=True)
            for i, im in enumerate(images):
                im.save(save_path + "images/{}.png".format(i))
            for i in range(beta_primes.shape[0]):
                torch.save(beta_primes[i], "{}/beta_primes/{}.pt".format(save_path, i))
        scores = []
        PeC = PearsonCorrCoef(num_outputs=beta_primes.shape[0]).to(self.device) 
        xDup = beta.repeat(beta_primes.shape[0], 1).moveaxis(0, 1).to(self.device)
        scores = PeC(xDup, beta_primes.moveaxis(0, 1).to(self.device))
        return scores, beta_primes
    
    

    # Main search method
    # beta is a 3*x_size tensor of brain data to use as guidance targets
    # init_img is a pil image to serve as a low level strcutral guesscur_iter
    def search(self, sample_path, beta, c_i, target_variance, init_img=None, mindeye=None):
        with torch.no_grad():
            mindeye_score, mindeye_beta_primes = self.score_samples(beta, [mindeye], save_path=sample_path)
            print(mindeye_score)
            # Initialize search variables
            best_image, best_distribution_score = init_img, -1
            iter_images, iter_scores, var_scores = [], [], []
            pbar = tqdm(total=self.n_iter, desc="Search iterations")
            # Prepare beta as search target
            print("TARGET VARIANCE AE: {:.10f}".format(target_variance))
            
            # Generate iteration 0
            iteration_samples = self.generateNSamples(image=init_img, 
                                                    c_i=c_i,  
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
            iteration_scores, iteration_beta_primes = self.score_samples(beta, iteration_samples, save_path=iter_path)
            
            
            #Update best image and iteration images from iteration 0
            if float(torch.mean(iteration_scores)) > best_distribution_score:
                best_distribution_score = float(torch.mean(iteration_scores))
                best_image = iteration_samples[int(torch.argmax(iteration_scores))]
            iter_scores.append(float(best_distribution_score))
            iter_images.append(best_image)
            bp_var = bootstrap_variance(iteration_beta_primes)
            var_scores.append(bp_var)
            tqdm.write("SEARCH VARIANCE BP: {:.10f}, BRAIN CORRELATION: {:.10f}".format(bp_var, float(torch.mean(iteration_scores))))
            best_distribution_params = {
                                        "images":iteration_samples,
                                        "beta_primes": iteration_beta_primes,
                                        "z_img": init_img,
                                        "strength": 0.92}
            pbar.update(1)
            # Target condition, our variance is lower than the target so our distribution is the right width
            if bp_var < target_variance:
                pbar.close()
                return best_image, best_distribution_params, iter_images, iter_scores, var_scores, mindeye_score
        
            # Iteration >0 loop
            for i in range(1, self.n_iter):
                # Initalize parameters for iteration
                strength = 0.92-0.92*(i/self.n_iter) 
                n_i = max(5, int((self.n_samples/self.n_branches)*strength))
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
                                                            n=n_i,  
                                                            strength=strength)
                    #Score batch samples
                    batch_scores, batch_beta_primes = self.score_samples(beta, batch_samples, batch_path)
                    
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
                    bp_var = bootstrap_variance(batch_beta_primes)
                    iteration_vars.append(bp_var)
                    #Update best image and iteration images
                    if torch.mean(batch_scores) > best_distribution_score:
                        best_distribution_score = torch.mean(batch_scores)
                        best_image = batch_samples[int(torch.argmax(batch_scores))]
                        best_distribution_params = {
                                        "images":batch_samples,
                                        "beta_primes": batch_beta_primes,
                                        "z_img": z_seeds[b],
                                        "strength": strength}
                
                # Concatenate scores from each batch to be sorted for seeding the next iteration
                iteration_scores = torch.concat(iteration_scores, dim=0)
                iteration_var = np.mean(np.array(iteration_vars))
                iter_scores.append(float(best_distribution_score))
                iter_images.append(best_image)
                
                
                var_scores.append(iteration_var)
                tqdm.write("SEARCH VARIANCE BP: {:.10f}, BRAIN CORRELATION: {:.10f}".format(bp_var, float(torch.mean(iteration_scores))))
                pbar.update(1)
                # Target condition, our variance is lower than the target so our distribution is the right width
                if iteration_var < target_variance:
                    print("SEARCH BELOW VARIANCE TARGET, EXITING...")
                    break
            pbar.close()
        return best_image, best_distribution_params, iter_images, iter_scores, var_scores, mindeye_score


    
