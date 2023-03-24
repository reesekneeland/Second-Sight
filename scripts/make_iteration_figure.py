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


def make_figure(experiment_title, i = []):
    exp_path = "/export/raid1/home/kneel027/Second-Sight/reconstructions/" + experiment_title + "/"
    paths = []
    images = []
    captions = []
    rows = len(i)
    columns = 13
    for index in i:
        sample_path = exp_path + str(index) + "/"
        images.append(Image.open(sample_path + "/Ground Truth.png"))
        captions.append("Ground Truth")
        images.append(Image.open(sample_path + "/Search Reconstruction.png"))
        captions.append("Final Reconstruction")
        images.append(Image.open(sample_path + "/Decoded CLIP Only.png"))
        captions.append("CLIP Only")
        for j in range(10):
            images.append(Image.open(sample_path + "/iter_" + str(j) + ".png"))
            captions.append("Iter " + str(j))
        
    output = tileImages("Search Iterations", images, captions, rows, columns)
    output.save(exp_path + "Search Iterations.png")
        
make_figure("SCS VD PCA LR 10:100:4 0.4 Exp AE", i=[0,1,2,3])
