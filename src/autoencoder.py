# Only GPU's in use
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
import torch
from torch.autograd import Variable
from torch.optim import Adam
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch.nn as nn
from torchmetrics.functional import pearson_corrcoef
from pycocotools.coco import COCO
import h5py
from utils import *
import wandb
import copy
from tqdm import tqdm
from pearson import PearsonCorrCoef



class encoder(torch.nn.modules):
    def __init__(self):
        super(encoder, self).__init__()
        
    self.relu = nn.ReLU()
    self.linear = nn.Linear(11838, 5000)
    self.linear2 = nn.Linear(5000, 1000)
    
    def forward(self, x):
        y_pred = self.relu(self.linear(x))
        y_pred = self.linear2(y_pred)
        return y_pred
        
class decoder(torch.nn.modules):
    def __init__(self):
        super(decoder, self).__init__()      
    
    self.relu = nn.ReLU()
    self.linear = nn.Linear(1000, 5000)
    self.linear2 = nn.Linear(5000, 11838) 
    
    def forward(self, x):
        y_pred = self.relu(self.linear(x))
        y_pred = self.linear2(y_pred)
        return y_pred

    
# Pytorch model class for Linear regression layer Neural Network
class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        
        self.firstlayer  = encoder()
        self.secondlayer = decoder() 
        self.relu = nn.ReLU()
        
    def forward(self, x):
        y_pred = self.firstlayer(x)
        y_pred = self.secondlayer(y_pred)
        return y_pred
    
    