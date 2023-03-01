import sys
import os
import struct
import time
import numpy as np
import scipy.io as sio
from scipy import ndimage as nd
from scipy import misc
from glob import glob
import h5py
import pickle
import math
import matplotlib.pyplot as plt
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
import nibabel as nib
from nsd_access import NSDAccess
import torch
from tqdm import tqdm
from pearson import PearsonCorrCoef


prep_path = "/export/raid1/home/kneel027/nsd_local/preprocessed_data/"
latent_path = "/export/raid1/home/kneel027/Second-Sight/latent_vectors/"

# First URL: This is the original read-only NSD file path (The actual data)
# Second URL: Local files that we are adding to the dataset and need to access as part of the data
# Object for the NSDAccess package
nsda = NSDAccess('/home/naxos2-raid25/kneel027/home/surly/raid4/kendrick-data/nsd', '/home/naxos2-raid25/kneel027/home/kneel027/nsd_local')


        
def get_hash():
    with open('/export/raid1/home/kneel027/Second-Sight/hash','r') as file:
        h = file.read()
    file.close()
    return str(h)

def update_hash():
    with open('/export/raid1/home/kneel027/Second-Sight/hash','r+') as file:
        h = int(file.read())
        new_h = f'{h+1:03d}'
        file.seek(0)
        file.write(new_h)
        file.truncate()      
    file.close()
    return str(new_h)

def get_last_token(s, tokens={'@': list, '.': dict}):
    l,name,entry,t = 2**31,'','',None
    for tok,toktype in tokens.items():
        ss = s.split(tok)
        if len(ss)>1 and len(ss[-1])<l:
            l = len(ss[-1])
            entry = ss[-1]
            name = tok.join(ss[:-1])
            t = toktype
    return name, entry, t


def has_token(s, tokens=['@', '.']):
    isin = False
    for tok in tokens:
        if tok in s:
            isin = True
    return isin
    
def extend_list(l, i, v):
    if len(l)<i+1:
        l += [None,]*(i+1-len(l))
    l[i] = v
    return l

def flatten_dict(base, append=''):
    '''flatten nested dictionary and lists'''
    flat = {}
    for k,v in base.items():
        if type(v)==dict:
            flat.update(flatten_dict(v, '%s%s.'%(append,k)))
        elif type(v)==list:
            flat.update(flatten_dict({'%s%s@%d'%(append,k,i): vv for i,vv in enumerate(v)}))
        else:
            flat['%s%s'%(append,k)] = v
    return flat

def embed_dict(fd):
    d = {}
    for k,v in fd.items():
        name, entry, ty = get_last_token(k, {'@': list, '.': dict})
        if ty==list:
            if name in d.keys():
                d[name] = extend_list(d[name], int(entry), v)
            else:
                d[name] = extend_list([], int(entry), v)
        elif ty==dict:
            if name in d.keys():
                d[name].update({entry: v})
            else:
                d[name] = {entry: v}
        else:
            if k in d.keys():
                d[k].update(v)
            else:
                d[k] = v   
    return embed_dict(d) if has_token(''.join(d.keys()), tokens=['@', '.']) else d


# Main data loader, 
# Loader = True
#    - Returns the train and test data loader
# Loader = False
#    - Returns the x_train, x_val, x_test, y_train, y_val, y_test

def load_nsd(vector, batch_size=375, num_workers=16, loader=True, split=True, ae=False, encoderModel=None, average=False, return_trial=False):
    if(ae):
        x = torch.load(prep_path + "x_encoded/" + encoderModel + "/" + "vector.pt").requires_grad_(False)
        y = torch.load(prep_path + "x/whole_region_11838.pt").requires_grad_(False)
    else:
        x = torch.load(prep_path + "x/whole_region_11838.pt").requires_grad_(False)
        y = torch.load(prep_path + vector + "/vector.pt").requires_grad_(False)
    
    if(not split): 
        return x, y
    
    else: 
        x_train, x_val, x_voxelSelection, x_thresholdSelection, x_test = [], [], [], [], []
        y_train, y_val, y_voxelSelection, y_thresholdSelection, y_test = [], [], [], [], []
        subj1_train = nsda.stim_descriptions[(nsda.stim_descriptions['subject1'] != 0) & (nsda.stim_descriptions['shared1000'] == False)]
        subj1_test = nsda.stim_descriptions[(nsda.stim_descriptions['subject1'] != 0) & (nsda.stim_descriptions['shared1000'] == True)]
        # Loads the raw tensors into a Dataset object

        # TensorDataset takes in two tensors of equal size and then maps 
        # them to one dataset. 
        # x is the brain data 
        # y are the vectors
        # train_i, test_i, val_i, voxelSelection_i, thresholdSelection_i = 0,0,0,0,0
        alexnet_stimuli_ordering  = []
        test_trials = []
        for i in tqdm(range(7500), desc="loading training samples"):
            if(average==True):
                avx = []
                avy = []
                for j in range(3):
                    scanId = subj1_train.iloc[i]['subject1_rep' + str(j)]
                
                    if(scanId < 27750):
                        avx.append(x[scanId-1])
                        avy.append(y[scanId-1])
                if(len(avx)>0):
                    avx = torch.stack(avx)
                    x_train.append(torch.mean(avx, dim=0))
                    y_train.append(avy[0])
            else:
                for j in range(3):
                    scanId = subj1_train.iloc[i]['subject1_rep' + str(j)]
                    if(scanId < 27750):
                        x_train.append(x[scanId-1])
                        y_train.append(y[scanId-1])
                        
                        
        for i in tqdm(range(7500, 9000), desc="loading validation samples"):
            if(average==True):
                avx = []
                avy = []
                for j in range(3):
                    scanId = subj1_train.iloc[i]['subject1_rep' + str(j)]
                
                    if(scanId < 27750):
                        avx.append(x[scanId-1])
                        avy.append(y[scanId-1])
                if(len(avx)>0):
                    avx = torch.stack(avx)
                    x_val.append(torch.mean(avx, dim=0))
                    y_val.append(avy[0])
            
            else:
                for j in range(3):
                    scanId = subj1_train.iloc[i]['subject1_rep' + str(j)]
                    if(scanId < 27750):
                        x_val.append(x[scanId-1])
                        y_val.append(y[scanId-1])
                        
        
        for i in range(200):
            if(average==True):
                avx = []
                avy = []
                for j in range(3):
                    scanId = subj1_train.iloc[i]['subject1_rep' + str(j)]
                
                    if(scanId < 27750):
                        avx.append(x[scanId-1])
                        avy.append(y[scanId-1])
                if(len(avx)>0):
                    avx = torch.stack(avx)
                    x_voxelSelection.append(torch.mean(avx, dim=0))
                    y_voxelSelection.append(avy[0])
            else:
                for j in range(3):
                    scanId = subj1_test.iloc[i]['subject1_rep' + str(j)]
                    if(scanId < 27750):
                        if(return_trial): 
                            x_test.append(x[scanId-1])
                        x_voxelSelection.append(x[scanId-1])
                        y_voxelSelection.append(y[scanId-1])
                        alexnet_stimuli_ordering.append(i)
                    
        for i in range(200, 400):
            if(average==True):
                avx = []
                avy = []
                for j in range(3):
                    scanId = subj1_train.iloc[i]['subject1_rep' + str(j)]
                
                    if(scanId < 27750):
                        avx.append(x[scanId-1])
                        avy.append(y[scanId-1])
                if(len(avx)>0):
                    avx = torch.stack(avx)
                    x_thresholdSelection.append(torch.mean(avx, dim=0))
                    y_thresholdSelection.append(avy[0])
            else:
                for j in range(3):
                    scanId = subj1_test.iloc[i]['subject1_rep' + str(j)]
                    if(scanId < 27750):
                        if(return_trial): 
                            x_test.append(x[scanId-1])
                        x_thresholdSelection.append(x[scanId-1])
                        y_thresholdSelection.append(y[scanId-1])
                        alexnet_stimuli_ordering.append(i)
                    
        for i in range(400, 1000):
            nsdId = subj1_train.iloc[i]['nsdId']
            if(average==True):
                avx = []
                avy = []
                for j in range(3):
                    scanId = subj1_train.iloc[i]['subject1_rep' + str(j)]
                    if(scanId < 27750):
                        avx.append(x[scanId-1])
                        avy.append(y[scanId-1])
                if(len(avx)>0):
                    avx = torch.stack(avx)
                    x_test.append(torch.mean(avx, dim=0))
                    y_test.append(avy[0])
                    test_trials.append(nsdId)
            else:
                for j in range(3):
                    scanId = subj1_test.iloc[i]['subject1_rep' + str(j)]
                    if(scanId < 27750):
                        x_test.append(x[scanId-1])
                        y_test.append(y[scanId-1])
                        test_trials.append(nsdId)
                        alexnet_stimuli_ordering.append(i)
        x_train = torch.stack(x_train).to("cpu")
        x_val = torch.stack(x_val).to("cpu")
        x_voxelSelection = torch.stack(x_voxelSelection).to("cpu")
        x_thresholdSelection = torch.stack(x_thresholdSelection).to("cpu")
        x_test = torch.stack(x_test).to("cpu")
        y_train = torch.stack(y_train)
        y_val = torch.stack(y_val)
        y_voxelSelection = torch.stack(y_voxelSelection)
        y_thresholdSelection = torch.stack(y_thresholdSelection)
        y_test = torch.stack(y_test)
        print("shapes: ", x_train.shape, x_val.shape, x_voxelSelection.shape, x_thresholdSelection.shape, x_test.shape, y_train.shape, y_val.shape, y_voxelSelection.shape, y_thresholdSelection.shape, y_test.shape, len(test_trials))

        if(loader):
            trainset = torch.utils.data.TensorDataset(x_train, y_train)
            valset = torch.utils.data.TensorDataset(x_val, y_val)
            voxelset = torch.utils.data.TensorDataset(x_voxelSelection, y_voxelSelection)
            thresholdset = torch.utils.data.TensorDataset(x_thresholdSelection, y_thresholdSelection)
            testset = torch.utils.data.TensorDataset(x_test, y_test)
            # Loads the Dataset into a DataLoader
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
            valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
            voxelloader = torch.utils.data.DataLoader(voxelset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
            threshloader = torch.utils.data.DataLoader(thresholdset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
            testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
            return trainloader, valloader, voxelloader, threshloader, testloader
        else:
            if(return_trial): 
                return x_train, x_val, x_voxelSelection, x_thresholdSelection, x_test, y_train, y_val, y_voxelSelection, y_thresholdSelection, y_test, alexnet_stimuli_ordering, test_trials
            else:
                return x_train, x_val, x_voxelSelection, x_thresholdSelection, x_test, y_train, y_val, y_voxelSelection, y_thresholdSelection, y_test, test_trials


def load_cc3m(vector, modelId, batch_size=1500, num_workers=16):
    x_path = latent_path + modelId + "/cc3m_batches/"
    y_path = prep_path + vector + "/cc3m_batches/"
    size = 2819140
    x = torch.zeros((size, 11838)).requires_grad_(False)
    if(vector == "z" or vector == "z_img_mixer"):
        y = torch.zeros((size, 16384)).requires_grad_(False)
    elif(vector == "c_img_0" or vector == "c_text_0"):
        y = torch.zeros((size, 768)).requires_grad_(False)
    
    for i in tqdm(range(124), desc="loading cc3m"):
        y[i*22735:i*22735+22735] = torch.load(y_path + str(i) + ".pt").requires_grad_(False)
        x[i*22735:i*22735+22735] = torch.load(x_path + str(i) + ".pt").requires_grad_(False)
    val_split = int(size*0.7)
    test_split = int(size*0.9)
    x_train = x[:val_split]
    y_train = y[:val_split]
    x_val = x[val_split:test_split]
    y_val = y[val_split:test_split]
    x_test = x[test_split:]
    y_test = y[test_split:]

    trainset = torch.utils.data.TensorDataset(x_train, y_train)
    valset = torch.utils.data.TensorDataset(x_val, y_val)
    testset = torch.utils.data.TensorDataset(x_test, y_test)
    # Loads the Dataset into a DataLoader
    trainLoader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valLoader = torch.utils.data.DataLoader(valset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    testLoader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return trainLoader, valLoader, testLoader
    
    


def create_whole_region_unnormalized(whole=False):
    
    if(whole):
        data = 41
        file = "x/whole_region_11838_unnormalized_all.pt"
        whole_region = torch.zeros((30000, 11838))
    else:
        data = 38
        file = "x/whole_region_11838_unnormalized.pt"
        whole_region = torch.zeros((27750, 11838))

    nsd_general = nib.load("/export/raid1/home/kneel027/Second-Sight/masks/brainmask_nsdgeneral_1.0.nii").get_fdata()
    print(nsd_general.shape)

    nsd_general_mask = np.nan_to_num(nsd_general)
    nsd_mask = np.array(nsd_general_mask.reshape((699192,)), dtype=bool)
        
    # Loads the full collection of beta sessions for subject 1
    for i in tqdm(range(1,data), desc="Loading Voxels"):
        beta = nsda.read_betas(subject='subj01', 
                            session_index=i, 
                            trial_index=[], # Empty list as index means get all 750 scans for this session (trial --> scan)
                            data_type='betas_fithrf_GLMdenoise_RR',
                            data_format='func1pt8mm')

        # Reshape the beta trails to be flattened. 
        beta = beta.reshape((699192, 750))

        for j in range(750):

            # Grab the current beta trail. 
            curScan = beta[:, j]
            
            single_scan = torch.from_numpy(curScan)

            # Discard the unmasked values and keeps the masked values. 
            whole_region[j + (i-1)*750] = single_scan[nsd_mask]
            
    # Save the tensor
    torch.save(whole_region, prep_path + file)
    
    
def create_whole_region_normalized(whole=False):
    
    if(whole):
        #whole_region_norm = torch.zeros((27750, 11838))
        whole_region_norm_z = torch.zeros((30000, 11838))
        whole_region = torch.load(prep_path + "x/whole_region_11838_unnormalized_all.pt")
                
        # Normalize the data using Z scoring method for each voxel
        for i in range(whole_region.shape[1]):
            voxel_mean, voxel_std = torch.mean(whole_region[:, i]), torch.std(whole_region[:, i])  
            normalized_voxel = (whole_region[:, i] - voxel_mean) / voxel_std
            whole_region_norm_z[:, i] = normalized_voxel
            

        # Normalize the data by dividing all elements by the max of each voxel
        # whole_region_norm = whole_region / whole_region.max(0, keepdim=True)[0]

        # Save the tensor
        torch.save(whole_region_norm_z, prep_path + "x/whole_region_11838_all.pt")
        #torch.save(whole_region_norm, prep_path + "x/whole_region_11838_old_norm.pt")
    
    else:
        #whole_region_norm = torch.zeros((27750, 11838))
        whole_region_norm_z = torch.zeros((27750, 11838))
        whole_region = torch.load(prep_path + "x/whole_region_11838_unnormalized.pt")
                
        # Normalize the data using Z scoring method for each voxel
        for i in range(whole_region.shape[1]):
            voxel_mean, voxel_std = torch.mean(whole_region[:, i]), torch.std(whole_region[:, i])  
            normalized_voxel = (whole_region[:, i] - voxel_mean) / voxel_std
            whole_region_norm_z[:, i] = normalized_voxel
            

        # Normalize the data by dividing all elements by the max of each voxel
        # whole_region_norm = whole_region / whole_region.max(0, keepdim=True)[0]

        # Save the tensor
        torch.save(whole_region_norm_z, prep_path + "x/whole_region_11838.pt")
        #torch.save(whole_region_norm, prep_path + "x/whole_region_11838_old_norm.pt")
    
    
def process_data(vector):
    
    if(vector == "z" or vector == "z_img_mixer"):
        vec_target = torch.zeros((27750, 16384))
        datashape = (1, 16384)
    elif(vector == "c"):
        vec_target = torch.zeros((27750, 1536))
        datashape = (1, 1536)
    elif(vector == "c_prompt"):
        vec_target = torch.zeros((27750, 78848))
        datashape = (1, 78848)
    elif(vector == "c_combined" or vector == "c_img_mixer"):
        vec_target = torch.zeros((27750, 3840))
        datashape = (1, 3840)

    # Loading the description object for subejct1
    
    subj1x = nsda.stim_descriptions[nsda.stim_descriptions['subject1'] != 0]

    for i in tqdm(range(0,27750), desc="vector loader"):
        
        # Flexible to both Z and C tensors depending on class configuration
        
        # TODO: index the column of this table that is apart of the 1000 test set. 
        # Do a check here. Do this in get_data
        # If the sample is part of the held out 1000 put it in the test set otherwise put it in the training set. 
        index = int(subj1x.loc[(subj1x['subject1_rep0'] == i+1) | (subj1x['subject1_rep1'] == i+1) | (subj1x['subject1_rep2'] == i+1)].nsdId)
        vec_target[i] = torch.reshape(torch.load("/export/raid1/home/kneel027/nsd_local/nsddata_stimuli/tensors/" + vector + "/" + str(index) + ".pt"), datashape)

    torch.save(vec_target, prep_path + vector + "/vector.pt")
    
def process_data_full(vector):
    
    if(vector == "z" or vector == "z_img_mixer"):
        # vec_target = torch.zeros((2819140, 16384))
        # vec_target2 = None
        datashape = (1,16384)
    elif(vector == "c_img_0" or vector == "c_text_0"):
        # vec_target = torch.zeros((2819140, 768))
        # vec_target2 = None
        datashape = (1,768)
    elif(vector == "c_combined"):
        vec_target = torch.zeros((73000, 768))
        vec_target2 = torch.zeros((73000, 768))
        datashape = (1,768)

    # Flexible to both Z and C tensors depending on class configuration
    # if vec_target2 is not None:
    #     for i in tqdm(range(73000), desc="vector loader"):
    #         full_vec = torch.load("/export/raid1/home/kneel027/nsd_local/nsddata_stimuli/tensors/" + vector + "/" + str(i) + ".pt")[:,0]
    #         full_vec2 = torch.load("/export/raid1/home/kneel027/nsd_local/nsddata_stimuli/tensors/" + vector + "/" + str(i) + ".pt")[:,1]
    #         vec_target[i] = full_vec.reshape(datashape)
    #         vec_target2[i] = full_vec2.reshape(datashape)
    #     torch.save(vec_target, prep_path + "c_img_0/vector_73k.pt")
    #     torch.save(vec_target2, prep_path + "c_text_0/vector_73k.pt")
    # else:
    for i in tqdm(range(124), desc="batched vector loader"):
        vec_target = torch.zeros((22735, datashape[1]))
        for j in range(22735):
            full_vec = torch.load("/home/naxos2-raid25/kneel027/home/kneel027/nsd_local/cc3m/tensors/" + vector + "/" + str(i*22735 + j) + ".pt")
            vec_target[j] = full_vec.reshape(datashape)
        torch.save(vec_target, prep_path + vector + "/cc3m_batches/" + str(i) + ".pt")
    
    
def extract_dim(vector, dim):
    
    if(vector == "z"):
        vec_target = torch.zeros((27750, 16384))
        datashape = (1, 16384)
    elif(vector == "c"):
        vec_target = torch.zeros((27750, 1536))
        datashape = (1, 1536)
    elif(vector == "c_prompt"):
        vec_target = torch.zeros((27750, 78848))
        datashape = (1, 78848)
    elif(vector == "c_combined" or vector == "c_img_mixer"):
        vec_target = torch.zeros((27750, 768))
        datashape = (1, 768)

    # Loading the description object for subejct1
    
    subj1x = nsda.stim_descriptions[nsda.stim_descriptions['subject1'] != 0]

    for i in tqdm(range(0,27750), desc="vector loader"):
        
        # Flexible to both Z and C tensors depending on class configuration
        
        # TODO: index the column of this table that is apart of the 1000 test set. 
        # Do a check here. Do this in get_data
        # If the sample is part of the held out 1000 put it in the test set otherwise put it in the training set. 
        index = int(subj1x.loc[(subj1x['subject1_rep0'] == i+1) | (subj1x['subject1_rep1'] == i+1) | (subj1x['subject1_rep2'] == i+1)].nsdId)
        full_vec = torch.load("/export/raid1/home/kneel027/nsd_local/nsddata_stimuli/tensors/" + vector + "/" + str(index) + ".pt")
        reduced_dim = full_vec[:,dim]
        vec_target[i] = torch.reshape(reduced_dim, datashape)

    torch.save(vec_target, prep_path + vector + "_" + str(dim) + "/vector.pt")
    
    
    
def grab_samples(vector, threshold, hashNum):
    
    whole_region = torch.load(prep_path + "x/whole_region_11838_old_norm.pt") 
    mask = np.load("/export/raid1/home/kneel027/Second-Sight/masks/" + hashNum + "_" + vector + "2voxels_pearson_thresh" + threshold + ".npy")
    new_len = np.count_nonzero(mask)
    target = torch.zeros((27750, new_len))
    for i in tqdm(range(27750), desc=(vector + " masking")): 
       
        # Indexing into the sample and then using y_mask to grab the correct samples. 
        target[i] = whole_region[i][torch.from_numpy(mask)]
    torch.save(target, prep_path + "x/" + vector + "_2voxels_pearson_thresh" + threshold + ".pt")

def compound_loss(pred, target):
        alpha = 0.9
        mse = nn.MSELoss()
        cs = nn.CosineSimilarity()
        loss = alpha * mse(pred, target) + (1 - alpha) * (1- torch.mean(cs(pred, target)))
        return loss
    
def format_clip(c):
    if(len(c.shape)<2):
        c = c.reshape((1,768))
    c_combined = []
    for i in range(c.shape[0]):
        c_combined.append(c[i].reshape((1,768)).to("cuda"))
    
    for j in range(5-c.shape[0]):
        c_combined.append(torch.zeros((1, 768), device="cuda"))
    
    c_combined = torch.cat(c_combined, dim=0).unsqueeze(0)
    c_combined = c_combined.tile(1, 1, 1)
    return c_combined

def tileImages(title, images, captions, h, w):
    bigH = 576 * h
    bigW = 512 * w 
    canvas = Image.new('RGB', (bigW, bigH+96), color='white')
    font = ImageFont.truetype("arial.ttf", 36)
    titleFont = ImageFont.truetype("arial.ttf", 48)
    textLabeler = ImageDraw.Draw(canvas)
    _, _, w, h = textLabeler.textbbox((0, 0), title, font=titleFont)
    textLabeler.text(((bigW-w)/2, 24), title, font=titleFont, fill='black')
    label = Image.new(mode="RGBA", size=(512,64), color="white")
    count = 0
    for j in range(96, bigH, 576):
        for i in range(0, bigW, 512):
            canvas.paste(images[count], (i,j))
            canvas.paste(label, (i, j+512))
            textLabeler.text((i+32, j+520), captions[count], font=font, fill='black')
            count+=1
    return canvas
