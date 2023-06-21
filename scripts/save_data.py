# Only GPU's in use
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
import torch
from torchmetrics.functional import pearson_corrcoef
from torch.autograd import Variable
import numpy as np
from nsd_access import NSDAccess
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch.nn as nn
from PIL import Image
from pycocotools.coco import COCO
sys.path.append('src')
from utils import *
import copy
from tqdm import tqdm
import nibabel as nib

# _, _, _, _, _, _, _ = load_nsd(vector="c_img_uc", subject=1, loader=False, average=False)
# _, _, _, _, _, _, _ = load_nsd(vector="c_img_uc", subject=2, loader=False, average=False)
# _, _, _, _, _, _, _ = load_nsd(vector="c_img_uc", subject=5, loader=False, average=False)
# _, _, _, _, _, _, _ = load_nsd(vector="c_img_uc", subject=7, loader=False, average=False)
# vdvae_73k = torch.load("/home/naxos2-raid25/kneel027/home/kneel027/nsd_local/preprocessed_data/z_vdvae_73k.pt")
# torch.save(torch.mean(vdvae_73k, dim=0), "/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/vdvae/train_mean.pt")
# torch.save(torch.std(vdvae_73k, dim=0), "/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/vdvae/train_std.pt")
# vdvae_27k = torch.load("/home/naxos2-raid25/kneel027/home/kneel027/nsd_local/preprocessed_data/subject1/z_vdvae.pt")
# for i in range(vdvae_73k.shape[0]):
#     print(i, torch.sum(torch.count_nonzero(vdvae_73k[i])), torch.sum(torch.count_nonzero(vdvae_27k[i])))
# process_raw_tensors(vector="z_vdvae")
# process_masks(subject=1, big=True)
# subjects = [7]
# for subject in subjects:
#     # create_whole_region_unnormalized(subject=subject, big=True)
#     # create_whole_region_normalized(subject=subject, big=True)
#     # process_data(subject=subject, vector="c_img_uc")
#     # process_data(subject=subject, vector="images")
#     # process_data(subject=subject, vector="z_vdvae")
#     process_masks(subject=subject, big=False)
#     process_masks(subject=subject, big=True)

# mask_path = "masks/subject1/nsdgeneral_big.nii.gz"
# # mask_path = "/export/raid1/home/styvesg/data/nsd/masks/subj01/func1pt8mm/brainmask_inflated_1.0.nii"
# # mask_path_v = "/home/naxos2-raid25/kneel027/home/kneel027/home/styvesg/data/nsd/masks/subj01/func1pt8mm/roi/prf-visualrois.nii.gz"
# nsd_general = nib.load(mask_path).get_fdata()
# nsd_general = np.nan_to_num(nsd_general)#.astype(bool)
# nsd_general = np.where(nsd_general==1.0, True, False)
# # nsd_general = np.where(nsd_general==1.0, True, False)
# print("NSD_GENERAL S1: ", np.unique(nsd_general, return_counts=True))

# visual_rois = nib.load(mask_path_v).get_fdata()
# V1L = np.where(visual_rois==1.0, True, False)
# V1R = np.where(visual_rois==2.0, True, False)
# V1 = torch.from_numpy(V1L[nsd_general] + V1R[nsd_general])
# print("V1 S1: ", np.unique(V1, return_counts=True))

# encoderWeights = torch.load("masks/subject{}/{}_encoder_prediction_weights.pt".format(1, "gnetEncoder_clipEncoder"))
# V1 = torch.load("masks/subject1/V1.pt")
# early_vis = torch.load("masks/subject1/early_vis.pt")
# higher_vis = torch.load("masks/subject1/higher_vis.pt")
# print(torch.mean(encoderWeights[0, V1], dim=0), torch.mean(encoderWeights[1, V1], dim=0))
# print(torch.mean(encoderWeights[0, early_vis], dim=0), torch.mean(encoderWeights[1, early_vis], dim=0))
# print(torch.mean(encoderWeights[0, higher_vis], dim=0), torch.mean(encoderWeights[1, higher_vis], dim=0))
# thresh = encoderWeights[0] > 0.5
# print(torch.sum(thresh))
nsda = NSDAccess('/home/naxos2-raid25/kneel027/home/surly/raid4/kendrick-data/nsd', '/home/naxos2-raid25/kneel027/home/kneel027/nsd_local')
subj1 = nsda.stim_descriptions[(nsda.stim_descriptions['subject1'] != 0) & (nsda.stim_descriptions['shared1000'] == True)]
subj1 = subj1.sort_values(by='subject1_rep0')

# idx = convert_indices(idx=[2, 7, 8, 10, 22, 28, 44, 61, 77, 90, 104, 110, 114, 121, 122, 159, 169, 185, 209, 210, 215, 225, 233, 255, 265, 325, 342, 351, 394, 401, 412, 414, 427, 439, 442, 451, 466, 467, 479, 487, 488, 500, 503, 504, 517, 519, 523, 531, 547, 579, 607, 609, 612, 616, 689, 735, 740, 776, 777, 779, 838, 874, 882, 887, 891, 916, 922, 928, 946, 960, 964, 970])
# print(idx)
# subj_test = nsda.stim_descriptions[(nsda.stim_descriptions['subject1'] != 0) & (nsda.stim_descriptions['shared1000'] == True)]
# sample_count = 0
# index_list = []
# for i in range(subj_test.shape[0]):
#     heldout = True
#     for j in range(3):
#         scanId = subj_test.iloc[i]['subject{}_rep{}'.format(1, j)]
#         if scanId < 27750:
#             heldout = False
#     if heldout == False:
#         nsdId = subj1.iloc[i]['nsdId']
#         img = nsda.read_images([nsdId], show=True)
#         Image.fromarray(img[0]).save("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/logs/shared1000_images_nsdId/" + str(sample_count) + ".png")
#         index_list.append(sample_count)
#         sample_count += 1
#     else:
#         index_list.append(-1)
# for i, index in enumerate(idx):
#     print(i)
#     nsdId = subj1.iloc[i]['nsdId']

#     img = nsda.read_images([nsdId], show=True)
#     Image.fromarray(img[0]).save("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/logs/shared1000_images_nsdId/" + str(i) + ".png")
gt_images = "/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/logs/shared1000_images_nsdId/"
rec_folder = "/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/reconstructions/subject7/Tagaki raw/"
new_folder = "/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/reconstructions/subject7/Tagaki/"
os.makedirs(new_folder, exist_ok=True)
converted_indicies = convert_indices(idx=[i for i in range(1000)], reverse=True)
converted_indicies = remove_heldout_indices(converted_indicies, scanId_sorted=False)
for i in tqdm(range(982)):
    index = converted_indicies[i]
    tqdm.write(str(index))
    os.makedirs(new_folder + str(index) + "/", exist_ok=True)
    gt = Image.open(gt_images + "{}.png".format(str(index)))
    gt.save(new_folder + str(index) + "/Ground Truth.png")
    for j in range(5):
        image = Image.open(rec_folder + "{:05d}_00{}_zc.png".format(i, j))
        image.save(new_folder + str(index) + "/{}.png".format(j))
        
# SAVE CORTICAL CONVOLUTION IMAGES   
# subj_test = nsda.stim_descriptions[(nsda.stim_descriptions['subject1'] != 0) & (nsda.stim_descriptions['shared1000'] == True)]
# # subj_test = subj_test.sort_values(by='subject1_rep0')
# sample_count = 0
# index_list = []
# for i in range(subj_test.shape[0]):
#     heldout = True
#     for j in range(3):
#         scanId = subj_test.iloc[i]['subject{}_rep{}'.format(1, j)]
#         if scanId < 27750:
#             heldout = False
#     if heldout == False:
#         index_list.append(sample_count)
#         sample_count += 1
#     else:
#         index_list.append(-1)
# gt_images = "/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/logs/shared1000_images_nsdId/"
# rec_folder = "/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/reconstructions/subject2/Cortical Convolutions/"
# image_block = np.load(rec_folder + "meshpool_adamw_lr1e-04_dc1e-01_dp5e-01_fd32_ind32_layer3_rw1e-07_ifw0e+00_ofw1e+00_kldw1e-08_ae10_kldse0_vsf1e+00_fixb2f_mupred_imgs.npy")
# print(image_block.shape)
# for index in tqdm(range(1000)):
#     if index_list[index] != -1:
#         i = index_list[index]
#         os.makedirs(rec_folder + str(i) + "/", exist_ok=True)
#         gt = Image.open(gt_images +"{}.png".format(str(i)))
#         gt.save(rec_folder + str(i) + "/Ground Truth.png")
#         # for j in range(5):
#         # print(image_block[index])
#         image = Image.fromarray((image_block[index] * 255).astype(np.uint8))
#         image.save(rec_folder + str(i) + "/0.png")

# rec_folder = "/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/reconstructions/subject1/Mind Diffuser raw/"
# out_folder = "/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/reconstructions/subject1/Mind Diffuser/"
# for i in tqdm(range(982)):
#     os.makedirs(out_folder + str(i) + "/", exist_ok=True)
#     gt = Image.open(gt_images +"{}.png".format(str(i)))
#     gt.save(out_folder + str(i) + "/Ground Truth.png")
#     # for j in range(5):
#     # print(image_block[index])
#     image = Image.open(rec_folder + "{}.png".format(str(i))).convert('RGB')
#     image.save(out_folder + str(i) + "/0.png")