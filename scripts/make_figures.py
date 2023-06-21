import os, sys
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib as plt
from PIL import Image
sys.path.append('src')
from utils import *
import seaborn as sns
from matplotlib.lines import Line2D
import cv2


def make_iteration_figure(idx, filename):
    base_path = "/export/raid1/home/kneel027/Second-Sight/reconstructions/subject"
    base_path2 = "/Final Run: SCS UC LD 6:100:4 Dual Guided clip_iter/"
    output_path = "/export/raid1/home/kneel027/Second-Sight/reconstructions/comparisons/"
    images = []
    captions = ["Ground Truth", "Second Sight (Ours)", "VDVAE", "CLIP", "CLIP+VDVAE", "Iteration 0", "Iteration 1", "Iteration 2", "Iteration 3", "Iteration 4", "Iteration 5"]
    for index in idx:
        images.append(Image.open("{}1{}{}/Ground Truth.png".format(base_path, base_path2, index)))
        images.append(Image.open("{}1{}{}/Search Reconstruction.png".format(base_path, base_path2, index)))
        images.append(Image.open("{}1{}{}/Decoded VDVAE.png".format(base_path, base_path2, index)))
        images.append(Image.open("{}1{}{}/Decoded CLIP Only.png".format(base_path, base_path2, index)))
        images.append(Image.open("{}1{}{}/Decoded CLIP+VDVAE.png".format(base_path, base_path2, index)))
        for j in range(6):
            images.append(Image.open("{}1{}{}/iter_{}.png".format(base_path, base_path2, index, j)))
        
    output = tileImages("", images, captions, len(idx), 11, useTitle=2, rowCaptions=False, buffer=35, redCol=True)
    output.save("{}{}.png".format(output_path, filename))


def make_results_figure(experiment_title, idx = []):
    exp_path = "/export/raid1/home/kneel027/Second-Sight/reconstructions/" + experiment_title + "/"
    images = []
    captions = []
    rows = len(idx)
    columns = 4
    captions = ["Ground Truth", "Stochastic Search (Ours)", "Decoded CLIP", "Best COCO Image"]
    for index in idx:
        sample_path = exp_path + str(index) + "/"
        images.append(Image.open(sample_path + "/Ground Truth.png"))
        images.append(Image.open(sample_path + "/Search Reconstruction.png"))
        images.append(Image.open(sample_path + "/Decoded CLIP Only.png"))
        images.append(Image.open(sample_path + "/Library Reconstruction.png"))
        
    output = tileImages("", images, captions, rows, columns, useTitle=2, rowCaptions=False)
    os.makedirs(exp_path + "results", exist_ok=True)
    output.save(exp_path + "results/Results.png")
    
def make_subject_figures(idx, filename):
    base_path = "/export/raid1/home/kneel027/Second-Sight/reconstructions/subject"
    base_path2 = "/Final Run: SCS UC LD 6:100:4 Dual Guided clip_iter/"
    output_path = "/export/raid1/home/kneel027/Second-Sight/reconstructions/comparisons/"
    
    images = []
    for index in tqdm(idx):
        images.append(Image.open("{}1{}{}/Ground Truth.png".format(base_path, base_path2, index)))
        images.append(Image.open("{}1{}{}/Search Reconstruction.png".format(base_path, base_path2, index)))
        images.append(Image.open("{}2{}{}/Search Reconstruction.png".format(base_path, base_path2, index)))
        images.append(Image.open("{}5{}{}/Search Reconstruction.png".format(base_path, base_path2, index)))
        images.append(Image.open("{}7{}{}/Search Reconstruction.png".format(base_path, base_path2, index)))
    captions = ["Ground Truth", "Subject 1", "Subject 2", "Subject 5", "Subject 7"]
    output = tileImages("", images, captions, len(idx), 5, useTitle=2, rowCaptions=False, buffer=35, redCol=True)
    output.save("{}{}.png".format(output_path, filename))
        
        
# make_results_figure("SCS UC 10:250:5 0.6 Exp3 AE Fixed copy", idx=[46, 0, 15, 52])
# make_results_figure("SCS UC 10:250:5 0.6 Exp3 AE Fixed copy", idx=[11, 8, 25, 41])
make_iteration_figure([90, 115, 122, 130, 159, 186, 488, 740], "Iteration Comparison 1")
# make_subject_figures([257, 265, 267, 298, 304, 312, 319, 320, 322, 332, 339, 348, 349, 361, 369], "appendix3tall")
# make_subject_figures([372, 427, 442, 451, 467, 500, 504, 531, 609, 616, 776, 779, 838, 882, 887], "appendix4tall")
