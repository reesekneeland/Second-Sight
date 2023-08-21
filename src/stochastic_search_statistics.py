import os
import sys
import torch
import pandas as pd
from PIL import Image
sys.path.append('src')
from utils import *
from tqdm import tqdm
from autoencoder import AutoEncoder
from torchmetrics import PearsonCorrCoef
from stochastic_search import StochasticSearch
import cv2
import random
from transformers import AutoProcessor, CLIPVisionModelWithProjection
import math
import re
import numpy as np
from skimage import io
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim
import os.path
import torchvision.models as tvmodels
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.transforms as T
import scipy as sp
from scipy.stats import pearsonr,binom,linregress
import clip

class batch_generator_external_images(Dataset):

    def __init__(self, images, net_name= 'CLIP Two-way'):
        self.images = images
        self.net_name = net_name
        
        if self.net_name == 'CLIP Two-way':
            self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        else:
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.num_test = len(images)
        
    def __getitem__(self,idx):
        img = self.images[idx]
        img = T.functional.resize(img,(224,224))
        img = T.functional.to_tensor(img).float()
        img = self.normalize(img)
        return img

    def __len__(self):
        return  self.num_test


class Stochastic_Search_Statistics():
    
    def __init__(self, subject = 1, device = "cuda"):

        self.device=device
        self.subject = subject
        # model_id = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        model_id = "openai/clip-vit-large-patch14"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.visionmodel = CLIPVisionModelWithProjection.from_pretrained(model_id).to(self.device)
        self.PeC = PearsonCorrCoef().to(self.device)
        self.PeC1 = PearsonCorrCoef(num_outputs=1).to(self.device) 
        self.mask_path = "data/preprocessed_data/subject{}/masks/".format(subject)
        self.masks = {0:torch.full((11838,), False),
                        1:torch.load(self.mask_path + "V1.pt"),
                        2:torch.load(self.mask_path + "V2.pt"),
                        3:torch.load(self.mask_path + "V3.pt"),
                        4:torch.load(self.mask_path + "V4.pt"),
                        5:torch.load(self.mask_path + "early_vis.pt"),
                        6:torch.load(self.mask_path + "higher_vis.pt")}  
            
        self.paper_image_indices = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 
                                    64, 65, 66, 67, 68, 69, 71, 72, 73, 74, 75, 77, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 106, 107, 108, 
                                    109, 110, 111, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145,
                                    147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 
                                    182, 183, 184, 185, 186, 188, 189, 190, 191, 192, 193, 194, 195, 196, 198, 199, 200, 201, 202, 203, 205, 206, 207, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 
                                    221, 222, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 256, 257, 
                                    258, 259, 261, 262, 263, 264, 265, 266, 267, 268, 270, 272, 273, 274, 275, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 
                                    297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 
                                    333, 334, 335, 336, 337, 338, 339, 341, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 364, 365, 366, 367, 368, 369, 370, 
                                    371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 393, 394, 395, 396, 397, 398, 400, 401, 402, 403, 404, 406, 407, 408, 
                                    409, 410, 411, 412, 413, 414, 415, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 
                                    447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 481, 482, 
                                    483, 484, 485, 486, 487, 488, 489, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 
                                    521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 547, 548, 549, 551, 552, 553, 554, 555, 556, 557, 
                                    558, 559, 560, 561, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 
                                    594, 595, 596, 597, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 616, 617, 618, 619, 620, 621, 622, 624, 625, 626, 627, 628, 629, 630, 631, 632, 
                                    633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 655, 656, 657, 659, 661, 662, 663, 664, 666, 667, 668, 669, 670, 671, 
                                    672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 694, 695, 696, 698, 699, 700, 701, 702, 703, 704, 705, 706, 708, 709, 
                                    710, 711, 712, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 
                                    747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 782, 783, 
                                    784, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819,
                                    820, 821, 822, 823, 824, 826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 838, 839, 840, 841, 842, 843, 844, 845, 847, 848, 849, 851, 852, 854, 855, 856, 857, 858, 859,
                                    861, 862, 863, 864, 865, 866, 867, 869, 870, 871, 872, 873, 874, 875, 876, 877, 878, 879, 880, 881, 882, 883, 884, 885, 886, 887, 888, 889, 890, 892, 893, 894, 895, 896, 897, 
                                    898, 899, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913, 915, 916, 917, 918, 919, 920, 921, 922, 923, 924, 925, 926, 927, 928, 929, 930, 931, 932, 933, 
                                    934, 936, 937, 938, 939, 940, 941, 942, 943, 944, 945, 947, 948, 949, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 970, 
                                    971, 974, 976, 977, 978, 979, 980, 981]

    def autoencoded_brain_samples(self, subject = 1):
        
        AE = AutoEncoder(config="hybrid",
                        inference=True,
                        subject=subject,
                        device=self.device)
        
        # Load the test samples
        _, _, x_test, _, _, y_test, test_trials = load_nsd(vector="images", subject=subject, loader=False, average=False, nest=True)
        #print(y_test[0].reshape((425,425,3)).numpy().shape)
        # test = Image.fromarray(y_test[0].reshape((425,425,3)).numpy().astype(np.uint8))
        # test.save("/home/naxos2-raid25/ojeda040/local/ojeda040/Second-Sight/logs/test.png")
        
        
        #x_test_ae = torch.zeros((x_test.shape[0], 11838))
        x_test_ae = torch.zeros((x_test.shape[0], x_test.shape[2]))
        
        for i in tqdm(range(x_test.shape[0]), desc = "Autoencoding samples and averaging" ):
            repetitions = []
            for j in range(3):
                if(torch.count_nonzero(x_test[i,j]) > 0):
                    repetitions.append(x_test[i,j])
                
            x_test_ae[i] = torch.mean(AE.predict(torch.stack(repetitions)),dim=0)
        
        return x_test_ae
    
    
    # Return all the different brain masks. 
    def return_all_masks(self):
        return self.masks[1], self.masks[2], self.masks[3], self.masks[4], self.masks[5], self.masks[6]

    # Calcaulate the pearson correlation for 
    def generate_pearson_correlation(self, beta_prime, beta, mask=None):
        
        if(mask is not None):
            beta = beta[mask]
                    
        scores = self.PeC1(beta.to(self.device), beta_prime.to(self.device))
        scores_np = scores.detach().cpu().numpy()
        
        return scores_np
        
    # SSIM metric calculation
    def calculate_ssim(self, ground_truth_path, reconstruction_path):

        ground_truth   = Image.open(ground_truth_path).resize((425, 425))
        reconstruction = Image.open(reconstruction_path).resize((425, 425))
        
        ground_truth = np.array(ground_truth) / 255.0
        reconstruction = np.array(reconstruction) / 255.0

        ground_truth = rgb2gray(ground_truth)
        reconstruction = rgb2gray(reconstruction)

        return ssim(reconstruction, ground_truth, gaussian_weights=True, sigma=1.5, use_sample_covariance=True, data_range=ground_truth.max()-ground_truth.min())
            
        
    # Pixel Correlation Metric 
    def calculate_pixel_correlation(self, ground_truth, reconstruction):
        
        return pixel_correlation(ground_truth, reconstruction)
        
    
    #two_way_prob is the two way identification experiment between the given image and a random search reconstruction of a different sample with respect to the ground truth
    #clip_pearson is the pearson correlation score between the clips of the two given images
    #Sample type controls which of the types of image to pick a random sample between
        #   0 --> 0.png
        #   1 --> 1.png
        #   2 --> 2.png
        #   3 --> 3.png
        #   4 --> 4.png
        #   5 --> Ground Truth
    def calculate_clip_similarity_papaer(self, experiment_name, sample, sampleType=1, subject = 1):
        with torch.no_grad():
            exp_path = "/export/raid1/home/ojeda040/Second-Sight/reconstructions/subject{}/{}/".format(subject, experiment_name)
            
            folders = sorted([int(f.name) for f in os.scandir(exp_path) if f.is_dir() and f.name != 'results'])
            rand_list = [i for i in range(len(folders)) if folders[i] != sample and os.listdir(exp_path + str(folders[i]) + "/")]
            rand_index = random.choice(rand_list)
            sampleTypes = {0: "0.png", 1: "1.png", 2: "2.png", 3: "3.png", 4: "4.png", 5: "Ground Truth.png"}
            random_image = Image.open(exp_path + str(folders[rand_index]) + "/" + sampleTypes[sampleType])
            image = Image.open(exp_path + str(sample) + "/" + sampleTypes[sampleType])
            ground_truth = Image.open(exp_path + str(sample) + "/Ground Truth.png")
            
            inputs = self.processor(images=[ground_truth, image, random_image], return_tensors="pt", padding=True).to(self.device)
            outputs = self.visionmodel(**inputs)
            
            gt_feature = outputs.image_embeds[0].reshape((768))
            reconstruct_feature = outputs.image_embeds[1].reshape((768))
            clip_cosine_sim = torch.nn.functional.cosine_similarity(gt_feature, reconstruct_feature, dim=0)
            rand_image_feature = outputs.image_embeds[2].reshape((768))
            rand_image_feature /= rand_image_feature.norm(dim=-1, keepdim=True)
            gt_feature /= gt_feature.norm(dim=-1, keepdim=True)
            reconstruct_feature /= reconstruct_feature.norm(dim=-1, keepdim=True)
            
            loss = (torch.stack([gt_feature @ reconstruct_feature, gt_feature @ rand_image_feature]) *100)
            two_way_prob = loss.softmax(dim=0)[0]
            clip_pearson = self.PeC(gt_feature.flatten(), reconstruct_feature.flatten())
        return float(two_way_prob), float(clip_pearson), float(clip_cosine_sim)
        
    # clip_pearson is the pearson correlation score between the clips of the two given images
    def calculate_clip_cosine_sim(self, ground_truth, prediction):
        with torch.no_grad():
            inputs = self.processor(images=[ground_truth, prediction], return_tensors="pt", padding=True).to(self.device)
            outputs = self.visionmodel(**inputs)
            
            gt_feature = outputs.image_embeds[0].reshape((768))
            reconstruct_feature = outputs.image_embeds[1].reshape((768))
            clip_cosine_sim = torch.nn.functional.cosine_similarity(gt_feature, reconstruct_feature, dim=0)
        return float(clip_cosine_sim)
    

    
    def pairwise_corr_all(self, ground_truth, predictions):
        r = np.corrcoef(ground_truth, predictions)     #cosine_similarity(ground_truth, predictions)#
        r = r[:len(ground_truth), len(ground_truth):]  # rows: groundtruth, columns: predicitons
       
        # congruent pairs are on diagonal
        congruents = np.diag(r)
        
        # for each column (predicition) we should count the number of rows (groundtruth) that the value is lower than the congruent (e.g. success).
        success = r < congruents
        success_cnt = np.sum(success, 0)
        
        # note: diagonal of 'success' is always zero so we can discard it. That's why we divide by len-1
        perf = np.mean(success_cnt) / (len(ground_truth)-1)
        p = 1 - binom.cdf(perf*len(ground_truth)*(len(ground_truth)-1), len(ground_truth)*(len(ground_truth)-1), 0.5)
        
        return perf, p


    # CNN Metrics
    def net_metrics(self, images):
        
        print(len(images))
        feat_list_dict = {}
        global feat_list
        feat_list = []
        def fn(module, inputs, outputs):
            feat_list.append(outputs.cpu().numpy())

        net_list = [
            ('Inception V3','avgpool'),
            ('CLIP Two-way','final'),
            ('AlexNet 2',2),
            ('AlexNet 5',5),
            ('AlexNet 7',7),
            ('EffNet-B','avgpool'),
            ('SwAV','avgpool')
            ]

        device = 0
        net = None
        batchsize=64

        for (net_name,layer) in net_list:
            feat_list = []
            print(net_name,layer)
            dataset = batch_generator_external_images(images)
            loader = DataLoader(dataset,batchsize,shuffle=False)
            
            if net_name == 'Inception V3': # SD Brain uses this
                net = tvmodels.inception_v3(pretrained=True)
                if layer == 'avgpool':
                    net.avgpool.register_forward_hook(fn) 
                elif layer == 'lastconv':
                    net.Mixed_7c.register_forward_hook(fn)
                    
            elif 'AlexNet' in net_name:
                net = tvmodels.alexnet(pretrained=True)
                if layer==2:
                    net.features[4].register_forward_hook(fn)
                elif layer==5:
                    net.features[11].register_forward_hook(fn)
                elif layer==7:
                    net.classifier[5].register_forward_hook(fn)
                    
            elif net_name == 'CLIP Two-way':
                model, _ = clip.load("ViT-L/14", device=self.device)
                net = model.visual
                net = net.to(torch.float32)
                if layer==7:
                    net.transformer.resblocks[7].register_forward_hook(fn)
                elif layer==12:
                    net.transformer.resblocks[12].register_forward_hook(fn)
                elif layer=='final':
                    net.register_forward_hook(fn)
            
            elif net_name == 'EffNet-B':
                net = tvmodels.efficientnet_b1(weights=True)
                net.avgpool.register_forward_hook(fn) 
                
            elif net_name == 'SwAV':
                net = torch.hub.load('facebookresearch/swav:main', 'resnet50')
                net.avgpool.register_forward_hook(fn) 
            net.eval()
            net.to(self.device)    
            
            with torch.no_grad():
                for i,x in enumerate(loader):
                    x = x.to(self.device)
                    _ = net(x)
                    
              
            feat_list = np.concatenate(feat_list)
    
            file_name = '{}'.format(net_name)
            feat_list_dict[file_name] = feat_list
            
        return feat_list_dict
    
    
    # Grab the image indicies to create calculations on. 
    def image_indices(self, experiment_name, subject = 1):
        
        # Directory path
        dir_path = "Second-Sight-Archive/reconstructions/subject{}/{}".format(self.subject, experiment_name)
        
        # Grab the list of files
        files = []
        for path in os.listdir(dir_path):
            
            # check if current path is a file
            # if os.path.isfile(os.path.join(dir_path, path)):
            files.append(path)
        
        # Get just the image number and then sort the list. 
        indicies = []
        for i in range(len(files)):
            indicies.append(int(re.search(r'\d+', files[i]).group()))
       
        indicies.sort()
        
        return indicies
        
    def create_beta_primes(self, experiment_name):
        
        folder_image_set = []
        ground_truth = []
        library_reconstruction = []
        #folders = ["best_distribution"]
        folders = ["vdvae_distribution", "clip_distribution", "clip+vdvae_distribution", "best_distribution"]
        directory_path = "output/{}/subject{}/".format(experiment_name, self.subject)
        
        existing_path = directory_path + "/22/clip_distribution/0_beta_prime.pt"
        
        if(not os.path.exists(existing_path)):
        
            SCS = StochasticSearch(modelParams=["gnet", "clip"], subject=self.subject, device=self.device)
            
            # List of image numbers created. 
            idx = self.image_indices(experiment_name, subject = self.subject)
            
            # Append rows to an empty DataFrame
            for i in tqdm(idx, desc="creating beta primes"):
                
                ground_truth_path = directory_path + str(i) + "/Ground Truth.png"
                ground_truth_image = Image.open(ground_truth_path)
                ground_truth.append(ground_truth_image)
                
                library_reconstruction_path = directory_path + str(i) + "/library_reconstruction.png"
                library_reconstruction_image = Image.open(library_reconstruction_path)
                library_reconstruction.append(library_reconstruction_image)
                
                for folder in folders:
                    
                    # Create the path
                    images_path = directory_path + str(i) + "/" + folder + "/images/"
                    beta_primes_path = directory_path + str(i) + "/" + folder + "/beta_primes/"
                    os.makedirs(beta_primes_path, exist_ok=True)
                    for filename in os.listdir(images_path): 
                        reconstruction_image = Image.open(images_path + filename)
                        folder_image_set.append(reconstruction_image)
                        
                    beta_primes = SCS.predict(folder_image_set)
                    
                    for j in range(beta_primes.shape[0]):
                        torch.save(beta_primes[j], "{}{}.pt".format(beta_primes_path, j))
                        
                    folder_image_set = []
                
                ground_truth_beta_prime = SCS.predict(ground_truth)
                torch.save(ground_truth_beta_prime[0], "{}/ground_truth_beta_prime.pt".format(directory_path + str(i)))
                ground_truth = []
                
                library_reconstruction_beta_prime = SCS.predict(library_reconstruction)
                torch.save(library_reconstruction_beta_prime[0], "{}/library_reconstruction_beta_prime.pt".format(directory_path + str(i)))
                library_reconstruction = []
                
    def calculate_brain_predictions(self, path, brain_mask_V1, brain_mask_V2, brain_mask_V3, brain_mask_V4, 
                                    brain_mask_early_visual, brain_mask_higher_visual, beta_sample):
        
        # Calculate brain predictions
        brain_prediction_nsd_general        = torch.load(path)
        brain_prediction_V1                 = brain_prediction_nsd_general[brain_mask_V1]
        brain_prediction_V2                 = brain_prediction_nsd_general[brain_mask_V2]
        brain_prediction_V3                 = brain_prediction_nsd_general[brain_mask_V3]
        brain_prediction_V4                 = brain_prediction_nsd_general[brain_mask_V4]
        brain_prediction_early_visual       = brain_prediction_nsd_general[brain_mask_early_visual]
        brain_prediction_higher_visual      = brain_prediction_nsd_general[brain_mask_higher_visual]
        
        # Pearson correlations for each reconstruction region
        pearson_correlation_V1              = float(self.generate_pearson_correlation(brain_prediction_V1, beta_sample, brain_mask_V1))
        pearson_correlation_V2              = float(self.generate_pearson_correlation(brain_prediction_V2, beta_sample, brain_mask_V2))
        pearson_correlation_V3              = float(self.generate_pearson_correlation(brain_prediction_V3, beta_sample, brain_mask_V3))
        pearson_correlation_V4              = float(self.generate_pearson_correlation(brain_prediction_V4, beta_sample, brain_mask_V4))
        pearson_correlation_early_visual    = float(self.generate_pearson_correlation(brain_prediction_early_visual, beta_sample, brain_mask_early_visual))
        pearson_correlation_higher_visual   = float(self.generate_pearson_correlation(brain_prediction_higher_visual, beta_sample, brain_mask_higher_visual))
        pearson_correlation_nsd_general     = float(self.generate_pearson_correlation(brain_prediction_nsd_general, beta_sample))
        
        return pearson_correlation_V1, pearson_correlation_V2, pearson_correlation_V3, pearson_correlation_V4, pearson_correlation_early_visual, pearson_correlation_higher_visual, pearson_correlation_nsd_general
    
    
    def create_dataframe_brain_predictions(self, experiment_name, logging = False):
        
        # Path to the folder
        directory_path = "/home/naxos2-raid25/ojeda040/local/ojeda040/Second-Sight-Archive/reconstructions/subject{}/{}/".format(self.subject, experiment_name)
        dataframe_path = "/home/naxos2-raid25/ojeda040/local/ojeda040/Second-Sight-Archive/reconstructions/subject{}/dataframes/".format(self.subject)
        
        os.makedirs(dataframe_path, exist_ok=True)
        
        # Create betas if needed
        #self.create_beta_primes(experiment_name)
        
        # List of image numbers created. 
        # idx = self.image_indices(experiment_name, subject = self.subject)
        
        # print("IDX: ", len(idx), idx)
        idx = self.paper_image_indices
        print("IDX: ", len(idx), idx)
        
        # Load Average Brain Samples
        _, _, beta_samples, _, _, _, _ = load_nsd(vector="images", subject=self.subject, loader=False, average=True, nest=False)
        
        # Grab the necessary brain masks
        brain_mask_V1, brain_mask_V2, brain_mask_V3, brain_mask_V4, brain_mask_early_visual, brain_mask_higher_visual = self.return_all_masks()
        
        # Create an Empty DataFrame
        # Object With column names only
        # Sample Indicator: 
            #   0 --> Ground Truth
            #   1 --> VDVAE Distribution        (Decoded Distribution)
            #   2 --> Clip Distrubituon         (Decoded CLIP Only)
            #   3 --> Clip Distrubituon + VDVAE (Decoded CLIP + VDVAE)
            #   4 --> iter_0
            #   5 --> iter_1
            #   6 --> iter_2
            #   7 --> iter_3
            #   8 --> iter_4
            #   9 --> iter_5
            #   11 --> Search Reconstruction
            #   12 --> Library Reconstruction
        df = pd.DataFrame(columns = ['ID', 'Sample Count', 'Batch Number', 'Sample Indicator', 'Strength', 'Brain Correlation V1', 'Brain Correlation V2', 
                                     'Brain Correlation V3', 'Brain Correlation V4', 'Brain Correlation Early Visual', 'Brain Correlation Higher Visual',
                                     'Brain Correlation NSD General', 'SSIM', 'Pixel Correlation', 'CLIP Cosine', 'CLIP Two-way', 'AlexNet 2', 
                                     'AlexNet 5', 'AlexNet 7', 'Inception V3', 'EffNet-B', 'SwAV' ])
        
        # Sample count. 
        sample_count = 0
        
        # Dataframe index count. 
        df_row_num = 0
        
        # Images per folder for net metrics.
        folder_images = []
        
        # Folders in the directory and sample number for dataframe operations
        folders = {}
        if(logging):
            folders = {"vdvae_distribution" : 1, "clip_distribution" : 2, "clip+vdvae_distribution" : 3, "iter_0" : 4, "iter_1" : 5 , "iter_2" : 6, "iter_3" : 7, "iter_4" : 8, "iter_5" : 9}
        else:
            folders = {"best_distribution": 10}
        
        # Append rows to an empty DataFrame
        for i in tqdm(idx, desc="creating dataframe rows"):
            
            # Ground Truth Image
            ground_truth_path = directory_path + str(i) + '/' + 'Ground Truth.png'
            ground_truth = Image.open(ground_truth_path)
            
            # Search Reconstruction 
            search_reconstruction_path = directory_path + str(i) + '/' + 'Search Reconstruction.png'
            search_reonstruction = Image.open(search_reconstruction_path)
            
            # Library Reconstruction
            library_reconstruction_path = directory_path + str(i) + '/' + 'Library Reconstruction.png'
            library_reconstruction = Image.open(library_reconstruction_path)
            
            for folder, sample_number in folders.items():
                
                print("In folder: ", folder)
                
                # Create the path
                path = directory_path + str(i) + "/" + folder + "/"
            
                if("iter" in folder):
                    
                    batch_number = torch.load(path + "/best_batch_index.pt")
                    
                    # Find out if this is the iter that the search reconstruction was taken from. 
                    iter_path           = directory_path + str(i) + '/' + folder + '.png'
                    ssim_iter           = self.calculate_ssim(iter_path, search_reconstruction_path)
                    
                    for filename in os.listdir(path + "/batch_" + str(int(batch_number))): 
                        
                        if(".pt" in filename):
                            continue
                    
                        # Reconstruction path
                        reconstruction_path = path + '/batch_' + str(int(batch_number)) + '/' + filename
                        
                        # Reconstruction image
                        reconstruction = Image.open(reconstruction_path)
                        folder_images.append(reconstruction)
                        
                        # Calculate the strength at that reconstruction iter image. 
                        strength = 0.92-0.3*(math.pow((sample_count + 1)/ 6, 3))
                        
                        # Pearson correlation for each region of the brain. 
                        pearson_correlation_V1, pearson_correlation_V2, pearson_correlation_V3, pearson_correlation_V4, pearson_correlation_early_visual, pearson_correlation_higher_visual, pearson_correlation_nsd_general = self.calculate_brain_predictions(path + '/batch_' + str(int(batch_number)) + "/" + str(sample_count) + "_beta_prime.pt", 
                                                                                            brain_mask_V1, brain_mask_V2, brain_mask_V3, brain_mask_V4, brain_mask_early_visual,
                                                                                            brain_mask_higher_visual, beta_samples[i])
                        
                        row = pd.DataFrame({'ID' : str(i), 'Sample Count' : str(sample_count), 'Batch Number' : str(int(batch_number)),  'Sample Indicator' : str(sample_number), 'Strength' : str(round(strength, 10)), 
                                            'Brain Correlation V1' : str(round(pearson_correlation_V1, 10)), 'Brain Correlation V2' : str(round(pearson_correlation_V2, 10)), 'Brain Correlation V3' : str(round(pearson_correlation_V3 , 10)), 
                                            'Brain Correlation V4' : str(round(pearson_correlation_V4, 10)), 'Brain Correlation Early Visual' : str(round(pearson_correlation_early_visual , 10)), 
                                            'Brain Correlation Higher Visual' : str(round(pearson_correlation_higher_visual, 10)), 'Brain Correlation NSD General' : str(round(pearson_correlation_nsd_general, 10))},  index=[df_row_num])
                                
                        # Add the row to the dataframe
                        df = pd.concat([df, row])
                        
                        # Iterate the counts
                        sample_count += 1
                        df_row_num += 1
                    
                else: 
                    for filename in os.listdir(path): 
                        
                        if(".pt" in filename):
                            continue
                        
                        # Reconstruction path
                        reconstruction_path = path + filename
                        
                        # Reconstruction image
                        reconstruction = Image.open(reconstruction_path)
                        folder_images.append(reconstruction)
                        
                        # Calculate the strength at that reconstruction iter image. 
                        strength = 0.92-0.3*(math.pow((sample_count + 1)/ 6, 3))
                        
                        # Pearson correlation for each region of the brain. 
                        pearson_correlation_V1, pearson_correlation_V2, pearson_correlation_V3, pearson_correlation_V4, pearson_correlation_early_visual, pearson_correlation_higher_visual, pearson_correlation_nsd_general = self.calculate_brain_predictions(path + str(sample_count) + "_beta_prime.pt", 
                                                                                            brain_mask_V1, brain_mask_V2, brain_mask_V3, brain_mask_V4,
                                                                                            brain_mask_early_visual, brain_mask_higher_visual, beta_samples[i])
                        
                        row = pd.DataFrame({'ID' : str(i), 'Sample Count' : str(sample_count), 'Sample Indicator' : str(sample_number), 'Strength' : str(round(strength, 10)), 'Brain Correlation V1' : str(round(pearson_correlation_V1, 10)),
                                            'Brain Correlation V2' : str(round(pearson_correlation_V2, 10)), 'Brain Correlation V3' : str(round(pearson_correlation_V3, 10)), 
                                            'Brain Correlation V4' : str(round(pearson_correlation_V4, 10)), 'Brain Correlation Early Visual' : str(round(pearson_correlation_early_visual, 10)),
                                            'Brain Correlation Higher Visual' : str(round(pearson_correlation_higher_visual, 10)), 'Brain Correlation NSD General' : str(round(pearson_correlation_nsd_general, 10))},  index=[df_row_num])
                                
                        # Add the row to the dataframe
                        df = pd.concat([df, row])
                        
                        # Iterate the counts
                        sample_count += 1
                        df_row_num += 1
                    
                # Reset the sample_count for the next folder. 
                sample_count = 0 
            
            # Make data frame row for library reconstruction Image
            
            # Pearson correlation for each region of the brain. 
            pearson_correlation_V1_library , pearson_correlation_V2_library , pearson_correlation_V3_library , pearson_correlation_V4_library , pearson_correlation_early_visual_library , pearson_correlation_higher_visual_library , pearson_correlation_nsd_general_library  = self.calculate_brain_predictions(directory_path + str(i) + "/library_reconstruction_beta_prime.pt", 
                                                                                                                                                                                                                    brain_mask_V1, brain_mask_V2, brain_mask_V3, brain_mask_V4, brain_mask_early_visual,
                                                                                                                                                                                                                    brain_mask_higher_visual, beta_samples[i])
            
            row_library_reoncstruction = pd.DataFrame({'ID' : str(i), 'Sample Indicator' : "12", 'Strength' : str(round(1, 10)), 'Brain Correlation V1' : str(round(pearson_correlation_V1_library, 10)),
                                            'Brain Correlation V2' : str(round(pearson_correlation_V2_library, 10)), 'Brain Correlation V3' : str(round(pearson_correlation_V3_library, 10)), 
                                            'Brain Correlation V4' : str(round(pearson_correlation_V4_library, 10)), 'Brain Correlation Early Visual' : str(round(pearson_correlation_early_visual_library, 10)),
                                            'Brain Correlation Higher Visual' : str(round(pearson_correlation_higher_visual_library, 10)), 'Brain Correlation NSD General' : str(round(pearson_correlation_nsd_general_library, 10))}, index=[df_row_num])
            df_row_num += 1
            folder_images.append(library_reconstruction)
            df = pd.concat([df, row_library_reoncstruction])
            
            # Pearson correlation for each region of the brain. 
            pearson_correlation_V1_gt, pearson_correlation_V2_gt, pearson_correlation_V3_gt, pearson_correlation_V4_gt, pearson_correlation_early_visual_gt, pearson_correlation_higher_visual_gt, pearson_correlation_nsd_general_gt  = self.calculate_brain_predictions(directory_path + str(i) + "/ground_truth_beta_prime.pt", 
                                                                                brain_mask_V1, brain_mask_V2, brain_mask_V3, brain_mask_V4, brain_mask_early_visual,
                                                                                brain_mask_higher_visual, beta_samples[i])
            
            row_ground_truth = pd.DataFrame({'ID' : str(i), 'Sample Indicator' : "0", 'Strength' : str(round(strength, 10)), 'Brain Correlation V1' : str(round(pearson_correlation_V1_gt, 10)),
                                            'Brain Correlation V2' : str(round(pearson_correlation_V2_gt, 10)), 'Brain Correlation V3' : str(round(pearson_correlation_V3_gt, 10)), 
                                            'Brain Correlation V4' : str(round(pearson_correlation_V4_gt, 10)), 'Brain Correlation Early Visual' : str(round(pearson_correlation_early_visual_gt, 10)),
                                            'Brain Correlation Higher Visual' : str(round(pearson_correlation_higher_visual_gt, 10)), 'Brain Correlation NSD General' : str(round(pearson_correlation_nsd_general_gt, 10))}, index=[df_row_num])
            df_row_num += 1
            folder_images.append(ground_truth)
            df = pd.concat([df, row_ground_truth])
                
            # Reset the sample_count for the next folder. 
            sample_count = 0 
            folder_images = []
                                           
                        
        print(df.shape)
        print(df)
        df.to_csv(dataframe_path + "statistics_df_" + experiment_name + "_" + str(len(idx)) +  "_brain_correlation_only_all_noae.csv")
        
    def create_dataframe(self, experiment_name, logging = False):
        # Path to the folder
        directory_path = "Second-Sight-Archive/reconstructions/subject{}/{}".format(self.subject, experiment_name)
        dataframe_path = "Second-Sight-Archive/reconstructions/subject{}/dataframes/".format(self.subject)
        
        os.makedirs(dataframe_path, exist_ok=True)
        
        # Create betas if needed
        self.create_beta_primes(experiment_name)
        
        # List of image numbers created. 
        idx = self.image_indices(experiment_name, subject = self.subject)
        print("IDX: ", len(idx), idx)
        # Autoencoded avearged brain samples 
        # beta_samples = self.autoencoded_brain_samples(subject=self.subject)
        _, _, beta_samples, _, _, _, _ = load_nsd(vector="images", subject=self.subject, loader=False, average=True, nest=False)
        
        # Grab the necessary brain masks
        brain_mask_V1, brain_mask_V2, brain_mask_V3, brain_mask_V4, brain_mask_early_visual, brain_mask_higher_visual = self.return_all_masks()
        
        # Create an Empty DataFrame
        # Object With column names only
        # Sample Indicator: 
            #   0 --> Ground Truth
            #   1 --> VDVAE Distribution        (Decoded Distribution)
            #   2 --> Clip Distrubituon         (Decoded CLIP Only)
            #   3 --> Clip Distrubituon + VDVAE (Decoded CLIP + VDVAE)
            #   4 --> iter_0
            #   5 --> iter_1
            #   6 --> iter_2
            #   7 --> iter_3
            #   8 --> iter_4
            #   9 --> iter_5
            #   10 --> Best Distribution
            #   11 --> Search Reconstruction
            #   12 --> Library Reconstruction
        df = pd.DataFrame(columns = ['ID', 'Sample Count', 'Batch Number', 'Sample Indicator', 'Strength', 'Brain Correlation V1', 'Brain Correlation V2', 
                                     'Brain Correlation V3', 'Brain Correlation V4', 'Brain Correlation Early Visual', 'Brain Correlation Higher Visual',
                                     'Brain Correlation NSD General', 'SSIM', 'Pixel Correlation', 'CLIP Cosine', 'CLIP Two-way', 'AlexNet 2', 
                                     'AlexNet 5', 'AlexNet 7', 'Inception V3', 'EffNet-B', 'SwAV' ])
        
        # Sample count. 
        sample_count = 0
        
        # Dataframe index count. 
        df_row_num = 0
        
        # Images per folder for net metrics.
        folder_images = []
        
        # Folders in the directory and sample number for dataframe operations
        folders = {}
        if(logging):
            folders = {"vdvae_distribution" : 1, "clip_distribution" : 2, "clip+vdvae_distribution" : 3, "iter_0" : 4, "iter_1" : 5 , "iter_2" : 6, "iter_3" : 7, "iter_4" : 8, "iter_5" : 9, "best_distribution": 10}
        else:
            folders = {"best_distribution": 10}
        
        # Append rows to an empty DataFrame
        for i in tqdm(idx, desc="creating dataframe rows"):
            
            # Ground Truth Image
            ground_truth_path = directory_path + str(i) + '/' + 'Ground Truth.png'
            ground_truth = Image.open(ground_truth_path)
            
            # Search Reconstruction 
            search_reconstruction_path = directory_path + str(i) + '/' + 'Search Reconstruction.png'
            search_reonstruction = Image.open(search_reconstruction_path)
            
            # Library Reconstruction
            library_reconstruction_path = directory_path + str(i) + '/' + 'library_reconstruction.png'
            library_reconstruction = Image.open(library_reconstruction_path)
            
            for folder, sample_number in folders.items():
                
                print("In folder: ", folder)
                
                # Create the path
                path = directory_path + str(i) + "/" + folder
            
                if("iter" in folder):
                    
                    batch_number = torch.load(path + "/best_batch_index.pt")
                    
                    # Find out if this is the iter that the search reconstruction was taken from. 
                    iter_path           = directory_path + str(i) + '/' + folder + '.png'
                    ssim_iter           = self.calculate_ssim(iter_path, search_reconstruction_path)
                    
                    for filename in os.listdir(path + "/batch_" + str(int(batch_number))): 
                        
                        if(".pt" in filename):
                            continue
                        
                        if(sample_count == 5):
                            break
                    
                        # Reconstruction path
                        reconstruction_path = path + '/batch_' + str(int(batch_number)) + '/' + filename
                        
                        # Reconstruction image
                        reconstruction = Image.open(reconstruction_path)
                        folder_images.append(reconstruction)
                        
                        # Pix Corr metrics calculation
                        pix_corr = self.calculate_pixel_correlation(ground_truth, reconstruction)
                        
                        # SSIM metrics calculation
                        ssim        = self.calculate_ssim(ground_truth_path, reconstruction_path)
                        
                        # CLIP metrics calculation
                        clip_cosine_sim = self.calculate_clip_cosine_sim(ground_truth, reconstruction)
                        
                        # Calculate the strength at that reconstruction iter image. 
                        strength = 0.92-0.3*(math.pow((sample_count + 1)/ 6, 3))
                        
                        # Pearson correlation for each region of the brain. 
                        pearson_correlation_V1, pearson_correlation_V2, pearson_correlation_V3, pearson_correlation_V4, pearson_correlation_early_visual, pearson_correlation_higher_visual, pearson_correlation_nsd_general = self.calculate_brain_predictions(path + '/batch_' + str(int(batch_number)) + "/" + str(sample_count) + "_beta_prime.pt", 
                                                                                            brain_mask_V1, brain_mask_V2, brain_mask_V3, brain_mask_V4, brain_mask_early_visual,
                                                                                            brain_mask_higher_visual, beta_samples[i])
                        
                        row = pd.DataFrame({'ID' : str(i), 'Sample Count' : str(sample_count), 'Batch Number' : str(int(batch_number)),  'Sample Indicator' : str(sample_number), 'Strength' : str(round(strength, 10)), 
                                            'Brain Correlation V1' : str(round(pearson_correlation_V1, 10)), 'Brain Correlation V2' : str(round(pearson_correlation_V2, 10)), 'Brain Correlation V3' : str(round(pearson_correlation_V3 , 10)), 
                                            'Brain Correlation V4' : str(round(pearson_correlation_V4, 10)), 'Brain Correlation Early Visual' : str(round(pearson_correlation_early_visual , 10)), 
                                            'Brain Correlation Higher Visual' : str(round(pearson_correlation_higher_visual, 10)), 'Brain Correlation NSD General' : str(round(pearson_correlation_nsd_general, 10)),
                                            'SSIM' : str(round(ssim, 10)), 'Pixel Correlation' : str(round(pix_corr, 10)), 'CLIP Cosine' : str(round(clip_cosine_sim, 10))},  index=[df_row_num])
                                
                        # Add the row to the dataframe
                        df = pd.concat([df, row])
                        
                        # Iterate the counts
                        sample_count += 1
                        df_row_num += 1
                    
                else: 
                    for filename in os.listdir(path + "/images/"): 
                        
                        if(sample_count == 5):
                            break
                        
                        # Reconstruction path
                        reconstruction_path = path + '/images/' + filename
                        
                        # Reconstruction image
                        reconstruction = Image.open(reconstruction_path)
                        folder_images.append(reconstruction)
                        
                        # Pix Corr metrics calculation
                        pix_corr = self.calculate_pixel_correlation(ground_truth, reconstruction)
                        
                        # SSIM metrics calculation
                        ssim    = self.calculate_ssim(ground_truth_path, reconstruction_path)
                        
                        # CLIP metrics calculation
                        clip_cosine_sim = self.calculate_clip_cosine_sim(ground_truth, reconstruction)
                        
                        # Calculate the strength at that reconstruction iter image. 
                        strength = 0.92-0.3*(math.pow((sample_count + 1)/ 6, 3))
                        
                        # Pearson correlation for each region of the brain. 
                        pearson_correlation_V1, pearson_correlation_V2, pearson_correlation_V3, pearson_correlation_V4, pearson_correlation_early_visual, pearson_correlation_higher_visual, pearson_correlation_nsd_general = self.calculate_brain_predictions(path + "/beta_primes/" + str(sample_count) + ".pt", 
                                                                                            brain_mask_V1, brain_mask_V2, brain_mask_V3, brain_mask_V4,
                                                                                            brain_mask_early_visual, brain_mask_higher_visual, beta_samples[i])
                        
                        row = pd.DataFrame({'ID' : str(i), 'Sample Count' : str(sample_count), 'Sample Indicator' : str(sample_number), 'Strength' : str(round(strength, 10)), 'Brain Correlation V1' : str(round(pearson_correlation_V1, 10)),
                                            'Brain Correlation V2' : str(round(pearson_correlation_V2, 10)), 'Brain Correlation V3' : str(round(pearson_correlation_V3, 10)), 
                                            'Brain Correlation V4' : str(round(pearson_correlation_V4, 10)), 'Brain Correlation Early Visual' : str(round(pearson_correlation_early_visual, 10)),
                                            'Brain Correlation Higher Visual' : str(round(pearson_correlation_higher_visual, 10)), 'Brain Correlation NSD General' : str(round(pearson_correlation_nsd_general, 10)),
                                            'SSIM' : str(round(ssim, 10)), 'Pixel Correlation' : str(round(pix_corr, 10)), 'CLIP Cosine' : str(round(clip_cosine_sim, 10))},  index=[df_row_num])
                                
                        # Add the row to the dataframe
                        df = pd.concat([df, row])
                        
                        # Iterate the counts
                        sample_count += 1
                        df_row_num += 1
                    
                # Reset the sample_count for the next folder. 
                sample_count = 0 
                        
            # Make dataframe row for search reconstruction
            pix_corr_search = self.calculate_pixel_correlation(ground_truth, search_reonstruction)
            ssim_search = self.calculate_ssim(ground_truth_path, search_reconstruction_path)
            clip_cosine_sim_search = self.calculate_clip_cosine_sim(ground_truth, reconstruction)
            row_search = pd.DataFrame({'ID' : str(i), 'Sample Indicator' : "11", 'SSIM' : str(round(ssim_search, 10)), 'Pixel Correlation' : str(round(pix_corr_search, 10)),  'CLIP Cosine' : str(round(clip_cosine_sim_search, 10))}, index=[df_row_num])
            df_row_num += 1
            folder_images.append(search_reonstruction)
            df = pd.concat([df, row_search])
            
            # Make data frame row for library reconstruction Image
            pix_corr_library = self.calculate_pixel_correlation(ground_truth, library_reconstruction)
            ssim_library = self.calculate_ssim(ground_truth_path, library_reconstruction_path)
            clip_cosine_sim_library = self.calculate_clip_cosine_sim(ground_truth, reconstruction)
            
            # Pearson correlation for each region of the brain. 
            pearson_correlation_V1_library , pearson_correlation_V2_library , pearson_correlation_V3_library , pearson_correlation_V4_library , pearson_correlation_early_visual_library , pearson_correlation_higher_visual_library , pearson_correlation_nsd_general_library  = self.calculate_brain_predictions(directory_path + str(i) + "/library_reconstruction_beta_prime.pt", 
                                                                                brain_mask_V1, brain_mask_V2, brain_mask_V3, brain_mask_V4, brain_mask_early_visual,
                                                                                brain_mask_higher_visual, beta_samples[i])
            
            row_library_reoncstruction = pd.DataFrame({'ID' : str(i), 'Sample Indicator' : "12", 'Strength' : str(round(1, 10)), 'Brain Correlation V1' : str(round(pearson_correlation_V1_library, 10)),
                                            'Brain Correlation V2' : str(round(pearson_correlation_V2_library, 10)), 'Brain Correlation V3' : str(round(pearson_correlation_V3_library, 10)), 
                                            'Brain Correlation V4' : str(round(pearson_correlation_V4_library, 10)), 'Brain Correlation Early Visual' : str(round(pearson_correlation_early_visual_library, 10)),
                                            'Brain Correlation Higher Visual' : str(round(pearson_correlation_higher_visual_library, 10)), 'Brain Correlation NSD General' : str(round(pearson_correlation_nsd_general_library, 10)),
                                            'SSIM' : str(round(ssim_library, 10)), 'Pixel Correlation' : str(round(pix_corr_library, 10)), 'CLIP Cosine' : str(round(clip_cosine_sim_library, 10))}, index=[df_row_num])
            df_row_num += 1
            folder_images.append(library_reconstruction)
            df = pd.concat([df, row_library_reoncstruction])
            
            # Make data frame row for ground truth Image
            clip_cosine_sim_gt = self.calculate_clip_cosine_sim(ground_truth, reconstruction)
            
            # Pearson correlation for each region of the brain. 
            pearson_correlation_V1_gt, pearson_correlation_V2_gt, pearson_correlation_V3_gt, pearson_correlation_V4_gt, pearson_correlation_early_visual_gt, pearson_correlation_higher_visual_gt, pearson_correlation_nsd_general_gt  = self.calculate_brain_predictions(directory_path + str(i) + "/ground_truth_beta_prime.pt", 
                                                                                brain_mask_V1, brain_mask_V2, brain_mask_V3, brain_mask_V4, brain_mask_early_visual,
                                                                                brain_mask_higher_visual, beta_samples[i])
            
            row_ground_truth = pd.DataFrame({'ID' : str(i), 'Sample Indicator' : "0", 'Strength' : str(round(strength, 10)), 'Brain Correlation V1' : str(round(pearson_correlation_V1_gt, 10)),
                                            'Brain Correlation V2' : str(round(pearson_correlation_V2_gt, 10)), 'Brain Correlation V3' : str(round(pearson_correlation_V3_gt, 10)), 
                                            'Brain Correlation V4' : str(round(pearson_correlation_V4_gt, 10)), 'Brain Correlation Early Visual' : str(round(pearson_correlation_early_visual_gt, 10)),
                                            'Brain Correlation Higher Visual' : str(round(pearson_correlation_higher_visual_gt, 10)), 'Brain Correlation NSD General' : str(round(pearson_correlation_nsd_general_gt, 10)),
                                            'CLIP Cosine' : str(round(clip_cosine_sim_gt, 10))}, index=[df_row_num])
            df_row_num += 1
            folder_images.append(ground_truth)
            df = pd.concat([df, row_ground_truth])
            
            # Calculate CNN metrics
            # net_predictions:
            #   Key:     Net Name
            #   Value:   Array of predicted values 
            net_predictions = self.net_metrics(folder_images)
            
            # Grab the key value pair in the dictionary. 
            for net_name, feature_list in net_predictions.items(): 

                # Iterate over the list of predictions
                for sample in range(feature_list.shape[0]):
                    
                    # Add the prediction at it's respected index to the dataframe. 
                    df.at[((df_row_num - (feature_list.shape[0])) + sample), net_name]  =  feature_list[sample].flatten().tolist()
                
            # Reset the sample_count for the next folder. 
            sample_count = 0 
            folder_images = []
                                           
                        
        print(df.shape)
        print(df)
        df.to_csv(dataframe_path + "statistics_df_" + experiment_name + "_" + str(len(idx)) +  ".csv")
    
    
def main():
    
    SCS = Stochastic_Search_Statistics(subject = 1, device="cuda:1")
    SCS.create_dataframe_brain_predictions("Final Run: SCS UC LD 6:100:4 Dual Guided clip_iter", logging = True)
        
if __name__ == "__main__":
    main()