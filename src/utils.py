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
import PIL.Image as Image
import nibabel as nib
from nsd_access import NSDAccess
import torch
from tqdm import tqdm
from pearson import PearsonCorrCoef


prep_path = "/export/raid1/home/kneel027/nsd_local/preprocessed_data/"

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
def load_data(vector, batch_size=375, num_workers=16, loader=True, split=True):
    
    y = torch.load(prep_path + vector + "/vector.pt").requires_grad_(False)
    x = torch.load(prep_path + "x/whole_region_11838_old_norm.pt").requires_grad_(False)
    x_train = torch.zeros((20480, 11838))
    x_val = torch.zeros((4500, 11838))
    x_test = torch.zeros((2770, 11838))
    y_train = torch.zeros((20480, y.shape[1]))
    y_val = torch.zeros((4500, y.shape[1]))
    y_test = torch.zeros((2770, y.shape[1]))
    print("shapes", x.shape, y.shape)
    subj1x = nsda.stim_descriptions[nsda.stim_descriptions['subject1'] != 0]
    
    # Loads the raw tensors into a Dataset object
    
    # TensorDataset takes in two tensors of equal size and then maps 
    # them to one dataset. 
    # x is the brain data 
    # y are the vectors
    train_i, test_i, val_i = 0,0,0
    trueCount = 0
    for i in range(x.shape[0]):
        test_sample = bool(subj1x.loc[(subj1x['subject1_rep0'] == i+1) | (subj1x['subject1_rep1'] == i+1) | (subj1x['subject1_rep2'] == i+1), "shared1000"].item())
        # print(test_sample)
        # test_sample=True
        if(test_sample):
            trueCount+=1
        if(test_sample):
            x_test[test_i] = x[i]
            y_test[test_i] = y[i]
            test_i +=1
        elif train_i<20480:
            x_train[train_i] = x[i]
            y_train[train_i] = y[i]
            train_i+=1
        else:
            x_val[val_i] = x[i]
            y_val[val_i] = y[i]
            val_i +=1

    if(loader):
        trainset = torch.utils.data.TensorDataset(x_train, y_train)
        testset = torch.utils.data.TensorDataset(x_val, y_val)
        
        # Loads the Dataset into a DataLoader
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        return trainloader, testloader
    else:
        print("shapes: ", x_train.shape, x_val.shape, x_test.shape, y_train.shape, y_val.shape, y_test.shape)
        return x_train, x_val, x_test, y_train, y_val, y_test

# Loads the data and puts it into a DataLoader
def get_data_decoder(vector, threshold=0.2, batch_size=375, num_workers=16, loader=True):
        
    y = torch.load(prep_path + vector + "/vector.pt").requires_grad_(False)
    x  = torch.load(prep_path + "x/" + vector + "_2voxels_pearson_thresh" + str(threshold) + ".pt").requires_grad_(False)
    x_train = x[:25500]
    x_test = x[25500:27750]
    y_train = y[:25500]
    y_test = y[25500:27750]
    print("shapes", x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    
    # Loads the raw tensors into a Dataset object
    
    # TensorDataset takes in two tensors of equal size and then maps 
    # them to one dataset. 
    # x is the brain data 
    # y are the captions
    if(loader):
        trainset = torch.utils.data.TensorDataset(x_train, y_train)
        testset = torch.utils.data.TensorDataset(x_test, y_test)
        
        # Loads the Dataset into a DataLoader
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        return trainloader, testloader
    else:
        return x_train, x_test, y_train, y_test


# Loads the data and puts it into a DataLoader
def get_data_encoder(vector, threshold=0.2, batch_size=375, num_workers=16, loader=True):
        
    
    # Clip data
    x  = torch.load(prep_path + vector + "/vector.pt").requires_grad_(False)

    # Brain data
    y = torch.load(prep_path + "x/whole_region_11838_old_norm.pt").requires_grad_(False)

    x_train = x[:25500]
    x_test = x[25500:27750]
    y_train = y[:25500]
    y_test = y[25500:27750]
    print("shapes", x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    
    # Loads the raw tensors into a Dataset object
    # TensorDataset takes in two tensors of equal size and then maps 
    # them to one dataset. 
    if(loader):
        trainset = torch.utils.data.TensorDataset(x_train, y_train)
        testset = torch.utils.data.TensorDataset(x_test, y_test)
        
        # Loads the Dataset into a DataLoader
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        return trainloader, testloader
    else:
        return x_train, x_test, y_train, y_test

def load_data_masked(vector):
        
        # Loads the preprocessed data
        prep_path = "/export/raid1/home/kneel027/nsd_local/preprocessed_data/"
        x = torch.load(prep_path + "x/whole_region_11838_old_norm.pt").requires_grad_(False)
        # x=torch.load("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/encoder_experiments/c_text_0.pt")
        y  = torch.load(prep_path + vector + "/vector.pt").requires_grad_(False)
        print(x.shape, y.shape)
        x_train = x[:25500]
        x_test = x[25500:27750]
        y_train = y[:25500]
        y_test = y[25500:27750]
        
        return x_train, x_test, y_train, y_test
    
def create_whole_region_unnormalized():
    
    whole_region = torch.load(prep_path + "x/whole_region_11838_unnormalized.pt")
    nsd_general = nib.load("/export/raid1/home/kneel027/Second-Sight/masks/brainmask_nsdgeneral_1.0.nii").get_data()
    print(nsd_general.shape)

    nsd_general_mask = np.nan_to_num(nsd_general)
    nsd_mask = np.array(nsd_general_mask.reshape((699192,)), dtype=bool)
        
    # Loads the full collection of beta sessions for subject 1
    for i in tqdm(range(1,38), desc="Loading Voxels"):
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
    torch.save(whole_region, prep_path + "x/whole_region_11838_unnormalized.pt")
    
    
def create_whole_region_normalized():
    
    whole_region_norm = torch.zeros((27750, 11838))
    whole_region_norm_z = torch.zeros((27750, 11838))
    whole_region = torch.load(prep_path + "x/whole_region_11838_unnormalized.pt")
            
    # Normalize the data using Z scoring method for each voxel
    for i in range(whole_region.shape[1]):
        voxel_mean, voxel_std = torch.mean(whole_region[:, i]), torch.std(whole_region[:, i])  
        normalized_voxel = (whole_region[:, i] - voxel_mean) / voxel_std
        whole_region_norm_z[:, i] = normalized_voxel
        

    # Normalize the data by dividing all elements by the max of each voxel
    whole_region_norm = whole_region / whole_region.max(0, keepdim=True)[0]

    # Save the tensor
    torch.save(whole_region_norm, prep_path + "x/whole_region_11838_old_norm.pt")
    
def normalization_test():
    
    whole_region_norm = torch.zeros((27750, 11838))
    whole_region_norm_z = torch.zeros((27750, 11838))
    whole_region = torch.load(prep_path + "x/whole_region_11838_unnormalized.pt")

    unnormalized = torch.var(whole_region, dim=0)
    print("unnormalized: ", unnormalized.shape, unnormalized)
    plt.figure(0)
    plt.hist(whole_region[0].numpy(), bins=40, log=True)
        #plt.yscale('log')
    plt.savefig("/export/raid1/home/kneel027/Second-Sight/charts/norm_testing_unnormalized.png")
    # TODO: Look at the betas before z score and max. 
    # Whitening: Normalize the output c and z vector and then unnormalize them before going back into the decoder. 
    # Learn the noramiziation maybe a two layer network
        # With a relu use two a positive one and a negative one
    # Action list
    #   Look at the actual effect of these normalization
    #   Look at the histogram of beta values and then apply normalization
    #   Add simple non linearities to improve decoders
    #   Start with the encoding model and then do bayes theory inversion with prior information (Clip to brain)
    #         - Create priors on the image  
    #         - Select the 100 tops ones that can 
    #         - Get an enormus library of clip vectos 
    #  
            
    # Normalize the data using Z scoring method for each voxel
    for i in range(whole_region.shape[1]):
        voxel_mean, voxel_std = torch.mean(whole_region[:, i]), torch.std(whole_region[:, i])  
        normalized_voxel = (whole_region[:, i] - voxel_mean) / voxel_std
        whole_region_norm_z[:, i] = normalized_voxel
        
    normalized_z = torch.mean(whole_region_norm_z, dim=0)
    print("normalized_z: ", normalized_z.shape, normalized_z)
    plt.figure(1)
    plt.hist(whole_region_norm_z[0].numpy(), bins=40, log=True)
        #plt.yscale('log')
    plt.savefig("/export/raid1/home/kneel027/Second-Sight/charts/norm_testing_z.png")
    # Normalize the data by dividing all elements by the max of each voxel
    whole_region_norm = whole_region / whole_region.max(0, keepdim=True)[0]
    normalized_max = torch.mean(whole_region_norm, dim=0)
    print("normalized_max: ", normalized_max.shape, normalized_max)
    plt.figure(2)
    plt.hist(whole_region_norm[0].numpy(), bins=40, log=True)
        #plt.yscale('log')
    plt.savefig("/export/raid1/home/kneel027/Second-Sight/charts/norm_testing_max.png")
    voxel_dir = "/export/raid1/home/styvesg/data/nsd/voxels/"
    voxel_data_set = h5py.File(voxel_dir+'voxel_data_V1_4_part1.h5py', 'r')
    voxel_data_dict = embed_dict({k: np.copy(d) for k,d in voxel_data_set.items()})
    voxel_data_set.close()
    voxel_data = voxel_data_dict['voxel_data']["1"]
    g_normalized = torch.mean(torch.from_numpy(voxel_data), dim=0)
    plt.figure(3)
    print("g_normalized: ", g_normalized.shape, g_normalized)
    plt.hist(voxel_data[0], bins=40, log=True)
        #plt.yscale('log')
    plt.savefig("/export/raid1/home/kneel027/Second-Sight/charts/norm_testing_ghislain.png")
    # Save the tensor
    # torch.save(whole_region_norm, prep_path + "x/whole_region_11838.pt")
    
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

def predictVector(model, vector, x):
        prep_path = "/export/raid1/home/kneel027/nsd_local/preprocessed_data/"
        latent_path = "/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/latent_vectors/"
        y = torch.load(prep_path + vector + "/vector_cc3m.pt").requires_grad_(False)
        x_preds = torch.zeros()
        x_preds = torch.load(latent_path + model + "/" + "cc3m_brain_preds.pt").requires_grad_(False)
        # y = y.detach()
        # x_preds = x_preds.detach()
        PeC = PearsonCorrCoef(num_outputs=x_preds.shape[0])
        out = torch.zeros((x.shape[0], y.shape[1]))
        for i in tqdm(range(x.shape[0]), desc="scanning library for " + vector):
            xDup = x[i].repeat(x_preds.shape[0], 1).moveaxis(0, 1)
            x_preds_t = x_preds.moveaxis(0, 1)
            # Pearson correlation
            # pearson = torch.zeros((73000,))
            print(x_preds_t.shape, xDup.shape)
            pearson = PeC(xDup, x_preds_t)
            print("pearson shape: ", pearson.shape)
            out[i] = y[pearson.argmax(dim=0)]
            print("max of pred: ", out[i].max())
        torch.save(out, latent_path + model + "/" + vector + "_cc3m_library_preds.pt")
        return out


def predictVector_cc3m(model, vector, x):
        if(vector == "c_img_0" or vector == "c_text_0"):
            datasize = 768
        elif(vector == "z_img_mixer"):
            datasize = 16384
        prep_path = "/export/raid1/home/kneel027/nsd_local/preprocessed_data/"
        latent_path = "/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/latent_vectors/"
        # y = torch.load(prep_path + vector + "/vector_cc3m.pt").requires_grad_(False)

        # y = y.detach()
        # x_preds = x_preds.detach()
        PeC = PearsonCorrCoef(num_outputs=22735).to("cuda")
        outputPeC = PearsonCorrCoef(num_outputs=620).to("cuda")
        loss = nn.MSELoss(reduction='none')
        out = torch.zeros((x.shape[0], 5, datasize))
        for i in tqdm(range(x.shape[0]), desc="scanning library for " + vector):
            xDup = x[i].repeat(22735, 1).moveaxis(0, 1).to("cuda")
            batch_max_x = torch.zeros((620, x.shape[1])).to("cuda")
            batch_max_y = torch.zeros((620, datasize)).to("cuda")
            for batch in tqdm(range(124), desc="batching sample"):
                y = torch.load(prep_path + vector + "/cc3m_batches/" + str(batch) + ".pt").to("cuda")
                # y_2 = torch.load(prep_path + vector + "/cc3m_batches/" + str(2*batch + 1) + ".pt").to("cuda")
                # y = torch.concat([y_1, y_2])
                x_preds = torch.load(latent_path + model + "/cc3m_batches/" + str(batch) + ".pt")
                # x_preds_2 = torch.load(latent_path + model + "/cc3m_batches/" + str(2*batch + 1) + ".pt")
                # x_preds_t = torch.concat([x_preds_1, x_preds_2]).moveaxis(0, 1).to("cuda")
                x_preds_t = x_preds.moveaxis(0, 1).to("cuda")
                # Pearson correlation
                # pearson = torch.zeros((73000,))
                # print(x_preds_t.shape, xDup.shape)
                # pearson = PeC(xDup, x_preds_t)
                L2 = torch.mean(loss(xDup, x_preds_t), dim=0)
                # print("pearson shape: ", pearson.shape)
                # print("L2 shape: ", L2.shape)
                top5_ind = torch.topk(L2, 5).indices
                for j, index in enumerate(top5_ind):
                    batch_max_x[5*batch + j] = x_preds_t[:,index].to("cuda")
                    batch_max_y[5*batch + j] = y[index].to("cuda")
            xDupOut = x[i].repeat(620, 1).moveaxis(0, 1).to("cuda")
            batch_max_x = batch_max_x.moveaxis(0, 1).to("cuda")
            print(xDupOut.shape, batch_max_x.shape)
            # outPearson = outputPeC(xDupOut, batch_max_x)
            outL2 = torch.mean(loss(xDupOut, batch_max_x), dim=0)
            top5_ind_out = torch.topk(outL2, 5).indices
            for j, index in enumerate(top5_ind_out):
                    out[i, j] = batch_max_y[index] 
            print("max of pred: ", out[i].max())
        torch.save(out, latent_path + model + "/" + vector + "_cc3m_library_preds_MSE.pt")
        return out
    
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