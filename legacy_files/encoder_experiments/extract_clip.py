import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
import torch
import numpy as np

vector="c_combined"
prep_path = "/export/raid1/home/kneel027/nsd_local/preprocessed_data/"
x = torch.load(prep_path + "x/whole_region_11838.pt").requires_grad_(False)
y  = torch.load(prep_path + vector + "/vector.pt").requires_grad_(False)
y = y.reshape((y.shape[0], 5, 768))
c_img_0 = torch.zeros((y.shape[0]), y.shape[-1])
c_text_0 = torch.zeros((y.shape[0]), y.shape[-1])
print(c_img_0.shape)

for i in range(y.shape[0]):
    c_img_0[i] = y[i,0,:]
    c_text_0[i] = y[i,1,:]

torch.save(c_img_0, "c_img_0.pt")
torch.save(c_text_0, "c_text_0.pt")