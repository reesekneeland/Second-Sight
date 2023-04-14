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


def make_iteration_figure(experiment_title, i = []):
    exp_path = "/export/raid1/home/kneel027/Second-Sight/reconstructions/" + experiment_title + "/"
    images = []
    captions = []
    rows = len(i)
    columns = 13
    captions = ["Ground Truth", "Stochastic Search (Ours)", "Decoded CLIP", "Iteration 1", "Iteration 2", "Iteration 3", 
                "Iteration 4", "Iteration 5", "Iteration 6", "Iteration 7", "Iteration 8", "Iteration 9", "Iteration 10"]
    for index in i:
        sample_path = exp_path + str(index) + "/"
        images.append(Image.open(sample_path + "/Ground Truth.png"))
        images.append(Image.open(sample_path + "/Search Reconstruction.png"))
        images.append(Image.open(sample_path + "/Decoded CLIP Only.png"))
        for j in range(10):
            images.append(Image.open(sample_path + "/iter_" + str(j) + ".png"))
        
    output = tileImages("", images, captions, rows, columns, useTitle=2, rowCaptions=False)
    os.makedirs(exp_path + "results", exist_ok=True)
    output.save(exp_path + "results/Search Iterations.png")
        

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
        
# make_results_figure("SCS UC 10:250:5 0.6 Exp3 AE Fixed copy", idx=[46, 0, 15, 52])
make_results_figure("SCS UC 10:250:5 0.6 Exp3 AE Fixed copy", idx=[11, 8, 25, 41])
# make_iteration_figure("SCS UC 10:250:5 0.6 Exp3 AE Fixed copy", i=[13, 28, 55, 21])