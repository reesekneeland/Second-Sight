import os
import sys
import torch
import numpy as np
from nsd_access import NSDAccess
import pandas as pd
from PIL import Image
from nsd_access import NSDAccess
import torch.nn as nn
from pycocotools.coco import COCO
sys.path.append('src')
from utils import *
from tqdm import tqdm
from reconstructor import Reconstructor
from pearson import PearsonCorrCoef
import cv2
import seaborn as sns
import matplotlib.pylab as plt
from random import randrange
import transformers
from transformers import CLIPTokenizerFast, AutoProcessor, CLIPModel, CLIPVisionModelWithProjection


device="cuda:3"
R = Reconstructor(device=device)

# dog = Image.open("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/dog.png")
# dog_var = Image.open("/home/naxos2-raid25/kneel027/home/kneel027/tester_scripts/dog_vd5.png")

sent1 = "A dog is running in the grass"
sent2 = "A dog is playing catch in the grass"
sent3 = "A surfer is riding a wave"

PeC = PearsonCorrCoef().to(device)
CrossEntropy = nn.CrossEntropyLoss().to(device)
# model_id = "openai/clip-vit-large-patch14"
# processor = AutoProcessor.from_pretrained(model_id)
# model = CLIPModel.from_pretrained(model_id).to(device)
# visionmodel = CLIPVisionModelWithProjection.from_pretrained(model_id).to(device)

# inputs = processor(images=[dog, dog_var], return_tensors="pt", padding=True).to(device)
# outputs = visionmodel(**inputs)

# im1_feature = outputs.image_embeds[0]
# im2_feature = outputs.image_embeds[1]
# im1_feature /= im1_feature.norm(dim=-1, keepdim=True)
# im2_feature /= im2_feature.norm(dim=-1, keepdim=True)

# clip_pearson = PeC(im1_feature.flatten(), im2_feature.flatten())
# clip_loss = nn.functional.mse_loss(im1_feature, im2_feature)
# print("CLIP: ", clip_pearson, clip_loss)

# im1_vd_feature = R.encode_image(dog)
# im2_vd_feature = R.encode_image(dog_var)

sent1_vd_feature = R.encode_text(sent1)
sent2_vd_feature = R.encode_text(sent2)
sent3_vd_feature = R.encode_text(sent3)
print(sent1_vd_feature.shape)
# # z1_pooled = im1_vd_feature[:, 0:1]
# # print("Z1_POOLED: ", z1_pooled.shape)
# # im1_vd_feature_norm = im1_vd_feature# / torch.norm(z1_pooled, dim=-1, keepdim=True)

# # z2_pooled = im2_vd_feature[:, 0:1]
# # im2_vd_feature_norm = im2_vd_feature# / torch.norm(z2_pooled, dim=-1, keepdim=True)

# # print(z1_pooled == im1_vd_feature[:,0,:])
# # # im1_vd_feature_norm = im1_vd_feature_norm[:,0,:]
# # # im2_vd_feature_norm = im2_vd_feature_norm[:,0,:]

vd_pearson = PeC(sent1_vd_feature.flatten(), sent2_vd_feature.flatten())
vd_pearson_bad = PeC(sent1_vd_feature.flatten(), sent3_vd_feature.flatten())
vd_loss = nn.functional.mse_loss(sent1_vd_feature, sent2_vd_feature)
vd_loss_bad = nn.functional.mse_loss(sent1_vd_feature, sent3_vd_feature)
print("VD: ", vd_pearson, vd_loss)
print("VD_bad: ", vd_pearson_bad, vd_loss_bad)

sent1_vd_feature = sent1_vd_feature[:,0,:]
sent2_vd_feature = sent2_vd_feature[:,0,:]
sent3_vd_feature = sent3_vd_feature[:,0,:]

reduced_pearson = PeC(sent1_vd_feature.flatten(), sent2_vd_feature.flatten())
reduced_pearson_bad = PeC(sent1_vd_feature.flatten(), sent3_vd_feature.flatten())
reduced_loss = nn.functional.mse_loss(sent1_vd_feature, sent2_vd_feature)
reduced_loss_bad = nn.functional.mse_loss(sent1_vd_feature, sent3_vd_feature)
print("reduced VD: ", reduced_pearson, reduced_loss)
print("reduced VD_bad: ", reduced_pearson_bad, reduced_loss_bad)


# im1_r_feature = im1_vd_feature[:,0,:]# / torch.norm(im1_vd_feature[:,0,:], dim=-1, keepdim=True)
# im2_r_feature = im2_vd_feature[:,0,:]# / torch.norm(im2_vd_feature[:,0,:], dim=-1, keepdim=True)
# # print(im1_vd_feature)
# reduced_pearson = PeC(im1_r_feature.flatten(), im2_r_feature.flatten())
# reduced_loss = nn.functional.mse_loss(im1_r_feature, im2_r_feature)
# print("REDUCED: ", reduced_pearson, reduced_loss)
# shared_1000_path = "/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/logs/shared1000_images/"
# gt_clips = []
# r_clips = []
# for i in tqdm(range(100)):
#     im = Image.open(shared_1000_path + str(i) + ".png")
#     gt_clip = R.encode_image(im)
#     gt_clips.append(gt_clip)
#     reconstruction = R.reconstruct(c_i=gt_clip, strength=1.0)
#     r_clip = R.encode_image(reconstruction)
#     r_clips.append(r_clip)
# gt_clips = torch.stack(gt_clips)
# r_clips = torch.stack(r_clips)
# print(gt_clips.shape)
# print(r_clips.shape)
# torch.save(gt_clips, "/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/logs/clip_test/gt_clips.pt")
# torch.save(r_clips, "/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/logs/clip_test/r_clips.pt")

# gt_clips = torch.load("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/logs/clip_test/gt_clips.pt")[:,:,0,:].reshape((100, 768))
# r_clips = torch.load("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/logs/clip_test/r_clips.pt")[:,:,0,:].reshape((100, 768))
# gt_clips = torch.load("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/logs/clip_test/gt_clips.pt").reshape((100, 197376))
# r_clips = torch.load("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/logs/clip_test/r_clips.pt").reshape((100, 197376))

# # r_clips = torch.flip(r_clips, dims=[0])
# mse_loss_matrix = torch.zeros((100,100))
# pearson_loss_matrix = torch.zeros((100,100))
# cosine_similarity_matrix = gt_clips @ r_clips.T

# labels = torch.diag(torch.ones((100), dtype=torch.float)).to(device)
# # print(labels.shape, labels[0:10])
# logitsx = nn.functional.softmax(cosine_similarity_matrix, dim=0)
# logitsy = nn.functional.softmax(cosine_similarity_matrix, dim=1)
# # print(labels[0], logits[0])
# # cross_entropy_loss = CrossEntropy(logits, labels)
# # print(cross_entropy_loss)
# # print(logits.shape, logits)
# for i in tqdm(range(100)):
#     for j in range(100):
#         mse_loss_matrix[i][j] = nn.functional.mse_loss(gt_clips[i], r_clips[j])
#         pearson_loss_matrix[i][j] = PeC(gt_clips[i], r_clips[j])
# ax = sns.heatmap(mse_loss_matrix.cpu().numpy())
# plt.title("mse_loss_matrix Normalized (lower is better)")
# plt.ylabel("Ground Truth Clip")
# plt.xlabel("Reconstructed Clip")
# plt.savefig("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/logs/clip_test/mse_loss_matrix.png")
# plt.clf()
# ax = sns.heatmap(pearson_loss_matrix.cpu().numpy())
# plt.title("pearson_loss_matrix Normalized (higher is better)")
# plt.ylabel("Ground Truth Clip")
# plt.xlabel("Reconstructed Clip")
# plt.savefig("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/logs/clip_test/pearson_loss_matrix.png")
# plt.clf()
# ax = sns.heatmap(cosine_similarity_matrix.cpu().numpy())
# plt.title("cosine_similarity_matrix Normalized (higher is better)")
# plt.ylabel("Ground Truth Clip")
# plt.xlabel("Reconstructed Clip")
# plt.savefig("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/logs/clip_test/cosine_loss_matrix.png")
# plt.clf()
# ax = sns.heatmap((logitsx + logitsy).cpu().numpy())
# plt.title("CLIP Probability Normalized (Higher is better)")
# plt.ylabel("Ground Truth Clip")
# plt.xlabel("Reconstructed Clip")
# plt.savefig("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/logs/clip_test/probability_matrix_test.png")
