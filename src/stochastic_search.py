import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import torch
import numpy as np
from PIL import Image
from nsd_access import NSDAccess
from pycocotools.coco import COCO
from utils import *
import math
import wandb
from tqdm import tqdm
from encoder import Encoder
from decoder import Decoder
from decoder_pca import Decoder_PCA
from alexnet_encoder import AlexNetEncoder
from autoencoder import AutoEncoder
from reconstructor import Reconstructor
from random import randrange



def main():
    # os.chdir("/export/raid1/home/kneel027/Second-Sight/")
    # for i in range(10):
    #     print(1.0-0.5*(math.pow(i/10, 2)))
    S0 = StochasticSearch(device="cuda:0",
                          log=False,
                          n_iter=10,
                          n_samples=100,
                          n_branches=4)
    # S1 = StochasticSearch(device="cuda:0",
    #                       log=False,
    #                       n_iter=10,
    #                       n_samples=250,
    #                       n_branches=5)
    # S2 = StochasticSearch(device="cuda:0",
    #                       log=True,
    #                       n_iter=20,
    #                       n_samples=60,
    #                       n_branches=3)
    S0.generateTestSamples(experiment_title="SCS VD PCA LR 10:100:4 0.5 Exponential Strength AE", idx=[i for i in range(0, 20)], mask=[], ae=True, test=False, average=True)
    # S0.generateTestSamples(experiment_title="SCS 10:100:4 best case AlexNet", idx=[i for i in range(0, 10)], mask=[1,2,3,4,5,6,7], ae=False)
    # S0.generateTestSamples(experiment_title="SCS 10:100:4 worst case random", idx=[i for i in range(0, 10)], mask=[1,2,3,4,5,6,7], ae=True)
    # S0.generateTestSamples(experiment_title="SCS 10:100:4 higher strength V1234 AE", idx=[i for i in range(0, 10)], mask=[1,2,3,4], ae=True)
    # S1.generateTestSamples(experiment_title="SCS 10:250:5 HS V1234567 AE", idx=[i for i in range(0, 20)], mask=[1, 2, 3, 4, 5, 6, 7], ae=True)
    # S1.generateTestSamples(experiment_title="SCS 10:250:5 HS V1234567 AE", idx=[i for i in range(20, 40)], mask=[1, 2, 3, 4, 5, 6, 7], ae=True)
    # S1.generateTestSamples(experiment_title="SCS VD ST 10:250:5 HS nsd_general AE", idx=[i for i in range(0, 786)], mask=[], ae=True)
    # S2.generateTestSamples(experiment_title="SCS 20:60:3 higher strength V1234567 AE", idx=[i for i in range(0, 10)], mask=[1, 2, 3, 4, 5, 6, 7], ae=True)
    # S2.generateTestSamples(experiment_title="SCS 20:60:3 higher strength V1 AE", idx=[i for i in range(0, 10)], mask=[1], ae=True)
    # S2.generateTestSamples(experiment_title="SCS 20:60:3 higher strength V1234 AE", idx=[i for i in range(0, 10)], mask=[1, 2, 3, 4], ae=True)

class StochasticSearch():
    def __init__(self, 
                device="cuda:0",
                log=True,
                n_iter=10,
                n_samples=10,
                n_branches=1):
        self.log = log
        self.device = device
        self.n_iter = n_iter
        self.n_samples = n_samples
        self.n_branches = n_branches
        self.R = Reconstructor(device="cuda:0")
        self.Alexnet = AlexNetEncoder()
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

    def generateNSamples(self, image, c_i, c_t, n, strength=1):
        images = []
        for i in tqdm(range(n), desc="Generating samples"):
            images.append(self.R.reconstruct(image=image, 
                                                c_i=c_i, 
                                                c_t=c_t, 
                                                n_samples=1, 
                                                textstrength=0.0,
                                                strength=strength))
        return images

    #clip is a 5x768 clip vector
    #beta is a 3x11838 tensor of brain data to reconstruct
    #cross validate says whether to cross validate between scans
    #n is the number of samples to generate at each iteration
    #max_iter caps the number of iterations it will perform
    def zSearch(self, c_i, c_t, beta, n=10, max_iter=10, n_branches=1, mask=[]):
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
            beta = beta[beta_mask]
        for cur_iter in tqdm(range(max_iter), desc="search iterations"):
            # if(loss_counter > 3):
            #     break
            # strength = 1.0-0.5*(cur_iter/max_iter)
            strength = 1.0-0.5*(math.pow(cur_iter/max_iter, 3))
            n_i = max(10, int(n/n_branches*strength))
            tqdm.write("Strength: " + str(strength) + ", N: " + str(n_i))
            
            samples = []
            for i in range(n_branches):
                samples += self.generateNSamples(image=iter_images[i], 
                                                c_i=c_i, 
                                                c_t=c_t, 
                                                n=n_i,  
                                                strength=strength)
        

            beta_primes = self.Alexnet.predict(samples, mask)
            # beta_primes = beta_primes[:, beta_mask]
            
            beta_primes = beta_primes.moveaxis(0, 1).to(self.device)
            
            xDup = beta.repeat(beta_primes.shape[1], 1).moveaxis(0, 1).to(self.device)
            PeC = PearsonCorrCoef(num_outputs=beta_primes.shape[1]).to(self.device) 
            print(xDup.shape, beta_primes.shape)
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
            # if cur_var < best_var or best_var == -1:
            #     best_var = cur_var
            # else:
            #     loss_counter +=1
            #     tqdm.write("loss counter: " + str(loss_counter))
        return best_image, images, iter_scores, var_scores




    def generateTestSamples(self, experiment_title, idx, mask=[], ae=False, test=True, average=True):    

        os.makedirs("reconstructions/" + experiment_title + "/", exist_ok=True)
        os.makedirs("logs/" + experiment_title + "/", exist_ok=True)
        AE = AutoEncoder(hashNum = "582",
                 lr=0.0000001,
                 vector="alexnet_encoder_sub1", #c_img_0, c_text_0, z_img_mixer
                 encoderHash="579",
                 log=False, 
                 batch_size=750,
                 device=self.device
                )
         # Load data and targets
        if test:
            _, _, _, _, _, _, _, targets_c_i, _, _ = load_nsd(vector="c_img_vd", loader=False, average=True)
            _, _, _, _, _, _, _, targets_c_t, _, _ = load_nsd(vector="c_text_vd", loader=False, average=True)
            _, _, _, x, _, _, _, _, _, trials = load_nsd(vector="c_img_vd", loader=False, average=False, nest=True)
        else:
            _, _, _, _, _, _, targets_c_i, _, _, _ = load_nsd(vector="c_img_vd", loader=False, average=True)
            _, _, _, _, _, _, targets_c_t, _, _, _ = load_nsd(vector="c_text_vd", loader=False, average=True)
            _, _, x, _, _, _, _, _, trials, _ = load_nsd(vector="c_img_vd", loader=False, average=False, nest=True)
        if(ae):
            x_pruned_ae = torch.zeros((len(idx), 11838))
        x_pruned = torch.zeros((len(idx), 11838))
        for i in tqdm(range(len(idx)), desc="Pruning samples"):# and averaging"):
            if(average):
                if(ae):
                    x_pruned_ae[i] = torch.mean(AE.predict(x[i]),dim=0)
                x_pruned[i] = torch.mean(x[i], dim=0)
            else:
                x_pruned[i] = x[i, randrange(0,3)]
                if(ae):
                    x_pruned_ae[i] = AE.predict(x_pruned[i])
        x = x_pruned
        
        # Dc_i = Decoder(hashNum = "634",
        #          vector="c_img_vd", 
        #          log=False, 
        #          device="cuda:0"
        #          )
    
        # Dc_t = Decoder(hashNum = "619",
        #             vector="c_text_vd", 
        #             log=False, 
        #             device="cuda:0"
        #             )
        Dc_i = Decoder_PCA(hashNum = "710",
                 vector="c_img_vd", 
                 log=False, 
                 device="cuda",
                 )
        outputs_c_i = Dc_i.predict(x=x)
        del Dc_i
        Dc_t = Decoder_PCA(hashNum = "712",
                    vector="c_text_vd",
                    log=False, 
                    device="cuda",
                    )
        outputs_c_t = Dc_t.predict(x=x)
        del Dc_t
        
        # Worst Case Random Samples
        # x, _ = load_nsd(vector ="c_img_0", loader = False, split = False)
        # x_param_rand = torch.zeros((len(idx), 11838))
        # for i in tqdm(idx, desc="making random betas"):
        #     for j in range(x_param_rand.shape[1]):
        #         randIndex = randrange(len(x)-len(idx)-2)
        #         x_param_rand[i,j] = x[randIndex,j]
        # x_param = x_param_rand
        
        # Generating predicted and target vectors
        # outputs_c_i = Dc_i.predict(x=x)
        # outputs_c_t = Dc_t.predict(x=x)
        # del Dc_t
        # del Dc_i
        if(ae):
            x = x_pruned_ae
        # Best Case Images
        # gt_images = []
        # for i in idx:
        #     nsdId = param_trials[i]
        #     ground_truth_np_array = self.nsda.read_images([nsdId], show=True)
        #     ground_truth = Image.fromarray(ground_truth_np_array[0])
        #     ground_truth = ground_truth.resize((512, 512), resample=Image.Resampling.LANCZOS)
        #     gt_images.append(ground_truth)
        # A = AlexNetEncoder(device=self.device)
        # x_param = A.predict(gt_images)
   
        PeC = PearsonCorrCoef(num_outputs=len(idx)).to("cpu")
        #Log the CLIP scores
        np.save("logs/" + experiment_title + "/" + "c_img_PeC.npy", np.array(PeC(outputs_c_i[idx].moveaxis(0,1).to("cpu"), targets_c_i[idx].moveaxis(0,1).to("cpu")).detach()))
        np.save("logs/" + experiment_title + "/" + "c_text_PeC.npy", np.array(PeC(outputs_c_t[idx].moveaxis(0,1).to("cpu"), targets_c_t[idx].moveaxis(0,1).to("cpu")).detach()))
        
        for i, val in enumerate(idx):
            os.makedirs("reconstructions/" + experiment_title + "/" + str(i) + "/", exist_ok=True)
            
            if(self.log):
                wandb.init(
                    # set the wandb project where this run will be logged
                    project="StochasticSearch",
                    # track hyperparameters and run metadata
                    config={
                    "experiment": experiment_title,
                    "sample": i,
                    "masks": mask,
                    "n_iter": self.n_iter,
                    "n_samples": self.n_samples
                    }
                )
            reconstructed_output_c = self.R.reconstruct(c_i=outputs_c_i[i], c_t=outputs_c_t[i], strength=1.0)
            reconstructed_target_c = self.R.reconstruct(c_i=targets_c_i[i], c_t=targets_c_t[i], strength=1.0)
            scs_reconstruction, image_list, score_list, var_list = self.zSearch(c_i=outputs_c_i[i], c_t=outputs_c_t[i], beta=x[i], n=self.n_samples, max_iter=self.n_iter, n_branches=self.n_branches, mask=mask)
            
            #log the data to a file
            np.save("logs/" + experiment_title + "/" + str(i) + "_score_list.npy", np.array(score_list))
            np.save("logs/" + experiment_title + "/" + str(i) + "_var_list.npy", np.array(var_list))
            
            # returns a numpy array 
            nsdId = trials[val]
            ground_truth_np_array = self.nsda.read_images([nsdId], show=True)
            ground_truth = Image.fromarray(ground_truth_np_array[0])
            ground_truth = ground_truth.resize((512, 512), resample=Image.Resampling.LANCZOS)
            rows = int(math.ceil(len(image_list)/2 + 2))
            columns = 2
            images = [ground_truth, scs_reconstruction, reconstructed_target_c, reconstructed_output_c]
            captions = ["Ground Truth", "Search Reconstruction", "Ground Truth CLIP", "Decoded CLIP Only"]
            for j in range(len(image_list)):
                images.append(image_list[j])
                captions.append("BC: " + str(round(score_list[j], 3)) + " VAR: " + str(round(var_list[j], 3)))
            print(len(images), len(captions), rows, columns)
            figure = tileImages(experiment_title + ": " + str(i), images, captions, rows, columns)
            if(self.log):
                wandb.finish()
            
            figure.save('reconstructions/' + experiment_title + '/' + str(i) + '.png')
            try:
                count = 0
                for j in range(len(images)):
                    if("BC" in captions[j]):
                        images[j].save('reconstructions/' + experiment_title + "/" + str(i) + "/iter_" + str(count) + '.png')
                        count +=1
                    else:
                        images[j].save('reconstructions/' + experiment_title + "/" + str(i) + "/" + captions[j] + '.png')
            except:
                pass
if __name__ == "__main__":
    main()
