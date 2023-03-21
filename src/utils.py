import sys
import os
import struct
import time
import numpy as np
import scipy.io as sio
from scipy import ndimage as nd
from scipy.special import erf
import matplotlib.pyplot as plt
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import nibabel as nib
from nsd_access import NSDAccess
import torch
from tqdm import tqdm
from pearson import PearsonCorrCoef
import pickle as pk


prep_path = "/export/raid1/home/kneel027/nsd_local/preprocessed_data/"
latent_path = "/export/raid1/home/kneel027/Second-Sight/latent_vectors/"

# First URL: This is the original read-only NSD file path (The actual data)
# Second URL: Local files that we are adding to the dataset and need to access as part of the data
# Object for the NSDAccess package
nsda = NSDAccess('/export/raid1/home/surly/raid4/kendrick-data/nsd', '/export/raid1/home/kneel027/nsd_local')


        
def get_hash():
    with open('hash','r') as file:
        h = file.read()
    file.close()
    return str(h)

def update_hash():
    with open('hash','r+') as file:
        h = int(file.read())
        new_h = f'{h+1:03d}'
        file.seek(0)
        file.write(new_h)
        file.truncate()      
    file.close()
    return str(new_h)

# Main data loader, 
# Loader = True
#    - Returns the train and test data loader
# Loader = False
#    - Returns the x_train, x_val, x_test, y_train, y_val, y_test

def load_nsd(vector, batch_size=375, num_workers=16, loader=True, split=True, ae=False, encoderModel=None, average=False, return_trial=False, old_norm=False, nest=False, pca=False):
    if(old_norm):
        region_name = "whole_region_11838_old_norm.pt"
    else:
        region_name = "whole_region_11838.pt"
        
    if(ae):
        x = torch.load(prep_path + "x/" + region_name).requires_grad_(False).to("cpu")
        y = torch.load(prep_path + "x_encoded/" + encoderModel + "/" + "vector.pt").requires_grad_(False).to("cpu")
    else:
        x = torch.load(prep_path + "x/" + region_name).requires_grad_(False)
        y = torch.load(prep_path + vector + "/vector.pt").requires_grad_(False)
        if(pca):
            pca = pk.load(open("masks/pca_" + vector + "_15k.pkl",'rb'))
            y = torch.from_numpy(pca.transform(y.numpy())).to(torch.float32)
    
    if(not split): 
        if(loader):
            dataset = torch.utils.data.TensorDataset(x, y)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
            return dataloader
        else:
            return x, y
    
    else: 
        x_train, x_val, x_param, x_test = [], [], [], []
        y_train, y_val, y_param, y_test = [], [], [], []
        subj1_train = nsda.stim_descriptions[(nsda.stim_descriptions['subject1'] != 0) & (nsda.stim_descriptions['shared1000'] == False)]
        subj1_test = nsda.stim_descriptions[(nsda.stim_descriptions['subject1'] != 0) & (nsda.stim_descriptions['shared1000'] == True)]
        subj1_full = nsda.stim_descriptions[(nsda.stim_descriptions['subject1'] != 0)]
        alexnet_stimuli_order_list = np.where(subj1_full["shared1000"] == True)[0]
        
        # Loads the raw tensors into a Dataset object

        # TensorDataset takes in two tensors of equal size and then maps 
        # them to one dataset. 
        # x is the brain data 
        # y are the vectors
        # train_i, test_i, val_i, voxelSelection_i, thresholdSelection_i = 0,0,0,0,0
        alexnet_stimuli_ordering  = []
        test_trials = []
        param_trials = []
        for i in tqdm(range(7500), desc="loading training samples"):
            nsdId = subj1_train.iloc[i]['nsdId']
            avx = []
            avy = []
            for j in range(3):
                scanId = subj1_train.iloc[i]['subject1_rep' + str(j)]
                if(scanId < 27750):
                    if(average==True):
                        avx.append(x[scanId-1])
                        avy.append(y[scanId-1])
                    else:
                        x_train.append(x[scanId-1])
                        y_train.append(y[scanId-1])
            if(len(avx) > 0):
                avx = torch.stack(avx)
                x_train.append(torch.mean(avx, dim=0))
                y_train.append(avy[0])
                         
        for i in tqdm(range(7500, 9000), desc="loading validation samples"):
            nsdId = subj1_train.iloc[i]['nsdId']
            avx = []
            avy = []
            for j in range(3):
                scanId = subj1_train.iloc[i]['subject1_rep' + str(j)]
                if(scanId < 27750):
                    if average:
                        avx.append(x[scanId-1])
                        avy.append(y[scanId-1])
                    else:
                        x_val.append(x[scanId-1])
                        y_val.append(y[scanId-1])
            if(len(avx) > 0):
                avx = torch.stack(avx)
                x_val.append(torch.mean(avx, dim=0))
                y_val.append(avy[0])
        
        for i in range(200):
            nsdId = subj1_test.iloc[i]['nsdId']
            avx = []
            avy = []
            x_row = torch.zeros((3, 11838))
            for j in range(3):
                scanId = subj1_test.iloc[i]['subject1_rep' + str(j)]
                if(scanId < 27750):
                    if average or nest:
                        avx.append(x[scanId-1])
                        avy.append(y[scanId-1])
                    else:
                        x_param.append(x[scanId-1])
                        y_param.append(y[scanId-1])
                        param_trials.append(nsdId)
                        alexnet_stimuli_ordering.append(alexnet_stimuli_order_list[i])
            if(len(avy)>0):
                if average:
                    avx = torch.stack(avx)
                    x_param.append(torch.mean(avx, dim=0))
                else:
                    for i in range(len(avx)):
                        x_row[i] = avx[i]
                    x_param.append(x_row)
                y_param.append(avy[0])
                param_trials.append(nsdId)
                    
        for i in range(200, 1000):
            nsdId = subj1_test.iloc[i]['nsdId']
            avx = []
            avy = []
            x_row = torch.zeros((3, 11838))
            for j in range(3):
                scanId = subj1_test.iloc[i]['subject1_rep' + str(j)]
                if(scanId < 27750):
                    if average or nest:
                        avx.append(x[scanId-1])
                        avy.append(y[scanId-1])
                    else:
                        x_test.append(x[scanId-1])
                        y_test.append(y[scanId-1])
                        test_trials.append(nsdId)
                        alexnet_stimuli_ordering.append(alexnet_stimuli_order_list[i])
            if(len(avy)>0):
                if average:
                    avx = torch.stack(avx)
                    x_test.append(torch.mean(avx, dim=0))
                else:
                    for i in range(len(avx)):
                        x_row[i] = avx[i]
                    x_test.append(x_row)
                y_test.append(avy[0])
                test_trials.append(nsdId)
        x_train = torch.stack(x_train).to("cpu")
        x_val = torch.stack(x_val).to("cpu")
        x_param = torch.stack(x_param).to("cpu")
        x_test = torch.stack(x_test).to("cpu")
        y_train = torch.stack(y_train)
        y_val = torch.stack(y_val)
        y_param = torch.stack(y_param)
        y_test = torch.stack(y_test)
        print("shapes: ", x_train.shape, x_val.shape, x_param.shape, x_test.shape, y_train.shape, y_val.shape, y_param.shape, y_test.shape)

        if(loader):
            trainset = torch.utils.data.TensorDataset(x_train, y_train)
            valset = torch.utils.data.TensorDataset(x_val, y_val)
            thresholdset = torch.utils.data.TensorDataset(x_param, y_param)
            testset = torch.utils.data.TensorDataset(x_test, y_test)
            # Loads the Dataset into a DataLoader
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
            valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
            paramLoader = torch.utils.data.DataLoader(thresholdset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
            testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
            return trainloader, valloader, paramLoader, testloader
        else:
            if(return_trial): 
                return x_train, x_val, x_param, x_test, y_train, y_val, y_param, y_test, alexnet_stimuli_ordering, param_trials, test_trials
            else:
                return x_train, x_val, x_param, x_test, y_train, y_val, y_param, y_test, param_trials, test_trials


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
    
    


def create_whole_region_unnormalized(whole=False, subject = "subj1"):
    
    #  Subject #1
    #   - NSD General = 11838
    #   - Brain shape = (81, 104, 83)
    #   - Flatten Brain Shape = 699192
    # 
    #  Subject #2
    #   - NSD General = 10325
    #   - Brain Shape = (82, 106, 84)
    #   - Flatten Brain Shape = 730128
    #

    nsd_general = nib.load("masks/" + subject + "/brainmask_nsdgeneral_1.0.nii").get_fdata()
    layer_size = np.sum(nsd_general == True)
    print(nsd_general.shape)
    
    
    if(whole):
        data = 41
        file = subject + "/x/whole_region_" + str(layer_size) + "_unnormalized_all.pt"
        whole_region = torch.zeros((30000, layer_size))
    else:
        data = 38
        file = subject + "/x/whole_region_" + str(layer_size) + "_unnormalized.pt"
        whole_region = torch.zeros((27750, layer_size))

    nsd_general_mask = np.nan_to_num(nsd_general)
    nsd_mask = np.array(nsd_general_mask.flatten(), dtype=bool)
    print(nsd_mask.shape)
        
    # Loads the full collection of beta sessions for subject 1
    for i in tqdm(range(1,data), desc="Loading Voxels"):
        beta = nsda.read_betas(subject='subj02', 
                            session_index=i, 
                            trial_index=[], # Empty list as index means get all 750 scans for this session (trial --> scan)
                            data_type='betas_fithrf_GLMdenoise_RR',
                            data_format='func1pt8mm')

        # Reshape the beta trails to be flattened. 
        beta = beta.reshape((nsd_mask.shape[0], 750))

        for j in range(750):

            # Grab the current beta trail. 
            curScan = beta[:, j]
            
            single_scan = torch.from_numpy(curScan)

            # Discard the unmasked values and keeps the masked values. 
            whole_region[j + (i-1)*750] = single_scan[nsd_mask]
            
    # Save the tensor
    print(whole_region.shape)
    torch.save(whole_region, prep_path + file)
    
    
def create_whole_region_normalized(whole = False, subject = "subj1"):
    
    subjects = {"subj1": 11838,
                "subj2": 10325}
    
    if(whole):

        whole_region = torch.load(prep_path + subject + "/x/whole_region_" + str(subjects[subject]) + "_unnormalized_all.pt")
        
        #whole_region_norm = torch.zeros((30000, subjects[subject]))
        whole_region_norm_z = torch.zeros((30000, subjects[subject]))
                
        # Normalize the data using Z scoring method for each voxel
        for i in range(whole_region.shape[1]):
            voxel_mean, voxel_std = torch.mean(whole_region[:, i]), torch.std(whole_region[:, i])  
            normalized_voxel = (whole_region[:, i] - voxel_mean) / voxel_std
            whole_region_norm_z[:, i] = normalized_voxel
            

        # Normalize the data by dividing all elements by the max of each voxel
        # whole_region_norm = whole_region / whole_region.max(0, keepdim=True)[0]

        # Save the tensor
        torch.save(whole_region_norm_z, prep_path + subject + "/x/whole_region_" + str(subjects[subject]) + "_all.pt")
        #torch.save(whole_region_norm, prep_path + "x/whole_region_" + str(subjects[subject]) + "_old_norm.pt")
    
    else:

        whole_region = torch.load(prep_path + subject + "/x/whole_region_" + str(subjects[subject]) + "_unnormalized.pt")
        
        #whole_region_norm = torch.zeros((27750, subjects[subject]))
        whole_region_norm_z = torch.zeros((27750, subjects[subject]))
                
        # Normalize the data using Z scoring method for each voxel
        for i in range(whole_region.shape[1]):
            voxel_mean, voxel_std = torch.mean(whole_region[:, i]), torch.std(whole_region[:, i])  
            normalized_voxel = (whole_region[:, i] - voxel_mean) / voxel_std
            whole_region_norm_z[:, i] = normalized_voxel
            

        # Normalize the data by dividing all elements by the max of each voxel
        # whole_region_norm = whole_region / whole_region.max(0, keepdim=True)[0]

        # Save the tensor
        torch.save(whole_region_norm_z, prep_path + subject + "/x/whole_region_" + str(subjects[subject]) + ".pt")
        #torch.save(whole_region_norm, prep_path + "x/whole_region_" + str(subjects[subject]) + "_old_norm.pt")
    
    
def process_data(vector="c_combined", image=False, subject = "subj1"):
    
    subjects = {"subj1": "subject1",
                "subj2": "subject2",
                "subj3": "subject3",
                "subj4": "subject4",
                "subj5": "subject5",
                "subj6": "subject6",
                "subj7": "subject7",
                "subj8": "subject8"}
    
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
    elif(vector == "c_img_vd"):
        vec_target = torch.zeros((27750, 197376))
        datashape = (1, 197376)
    elif(vector == "c_text_vd"):
        vec_target = torch.zeros((27750, 59136))
        datashape = (1, 59136)

    # Loading the description object for subejct1
    
    subjx = nsda.stim_descriptions[nsda.stim_descriptions[subjects[subject]] != 0]
    images = []
    for i in tqdm(range(0,27750), desc="vector loader"):
        
        # Flexible to both Z and C tensors depending on class configuration
        
        # TODO: index the column of this table that is apart of the 1000 test set. 
        # Do a check here. Do this in get_data
        # If the sample is part of the held out 1000 put it in the test set otherwise put it in the training set. 
        index = int(subjx.loc[(subjx[subjects[subject] + "_rep0"] == i+1) | (subjx[subjects[subject] + "_rep1"] == i+1) | (subjx[subjects[subject] + "_rep2"] == i+1)].nsdId)
        if(image):
            ground_truth_image_np_array = nsda.read_images([index], show=False)
            ground_truth_PIL = Image.fromarray(ground_truth_image_np_array[0])
            images.append(ground_truth_PIL)
        else:
            vec_target[i] = torch.reshape(torch.load("/export/raid1/home/kneel027/nsd_local/nsddata_stimuli/tensors/" + vector + "/" + str(index) + ".pt"), datashape)
    if(image):
        return images
    else:
        os.makedirs(prep_path + subjects + "/" + vector + "/", exist_ok=True)
        torch.save(vec_target, prep_path + subjects + "/" + vector + "/vector.pt")
    
def process_data_full(vector):
    
    # if(vector == "z" or vector == "z_img_mixer"):
    #     # vec_target = torch.zeros((2819140, 16384))
    #     # vec_target2 = None
    #     datashape = (1,16384)
    # elif(vector == "c_img_0" or vector == "c_text_0"):
    #     # vec_target = torch.zeros((2819140, 768))
    #     # vec_target2 = None
    #     datashape = (1,768)
    # elif(vector == "c_combined"):
    #     vec_target = torch.zeros((73000, 768))
    #     vec_target2 = torch.zeros((73000, 768))
    #     datashape = (1,768)
    if(vector == "c_img_vd"):
        vec_target = torch.zeros((73000, 197376))
        datashape = (1,197376)
    elif(vector == "c_text_vd"):
        vec_target = torch.zeros((73000, 59136))
        datashape = (1,59136)

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
    # for i in tqdm(range(124), desc="batched vector loader"):
    #     vec_target = torch.zeros((22735, datashape[1]))
    #     for j in range(22735):
    #         full_vec = torch.load("/home/naxos2-raid25/kneel027/home/kneel027/nsd_local/cc3m/tensors/" + vector + "/" + str(i*22735 + j) + ".pt")
    #         vec_target[j] = full_vec.reshape(datashape)
    #     torch.save(vec_target, prep_path + vector + "/cc3m_batches/" + str(i) + ".pt")
    for i in tqdm(range(73000), desc="vector loader"):
        full_vec = torch.load("/export/raid1/home/kneel027/nsd_local/nsddata_stimuli/tensors/" + vector + "/" + str(i) + ".pt").reshape(datashape)
        vec_target[i] = full_vec
    torch.save(vec_target, prep_path + vector + "/vector_73k.pt")

def tileImages(title, images, captions, h, w):
    bigH = 576 * h
    bigW = 512 * w 
    canvas = Image.new('RGB', (bigW, bigH+128), color='white')
    line = Image.new('RGB', (bigW, 8), color='black')
    canvas.paste(line, (0,120))
    font = ImageFont.truetype("arial.ttf", 36)
    titleFont = ImageFont.truetype("arial.ttf", 48)
    textLabeler = ImageDraw.Draw(canvas)
    _, _, w, h = textLabeler.textbbox((0, 0), title, font=titleFont)
    textLabeler.text(((bigW-w)/2, 32), title, font=titleFont, fill='black')
    label = Image.new(mode="RGBA", size=(512,64), color="white")
    count = 0
    for j in range(128, bigH, 576):
        for i in range(0, bigW, 512):
            if(count < len(images)):
                canvas.paste(images[count], (i,j))
                canvas.paste(label, (i, j+512))
                textLabeler.text((i+32, j+520), captions[count], font=font, fill='black')
                count+=1
    return canvas

#  Numpy Utility 
def iterate_range(start, length, batchsize):
    batch_count = int(length // batchsize )
    residual = int(length % batchsize)
    for i in range(batch_count):
        yield range(start+i*batchsize, start+(i+1)*batchsize),batchsize
    if(residual>0):
        yield range(start+batch_count*batchsize,start+length),residual 
        
def gaussian_mass(xi, yi, dx, dy, x, y, sigma):
    return 0.25*(erf((xi-x+dx/2)/(np.sqrt(2)*sigma)) - erf((xi-x-dx/2)/(np.sqrt(2)*sigma)))*(erf((yi-y+dy/2)/(np.sqrt(2)*sigma)) - erf((yi-y-dy/2)/(np.sqrt(2)*sigma)))

def make_gaussian_mass(x, y, sigma, n_pix, size=None, dtype=np.float32):
    deg = dtype(n_pix) if size==None else size
    dpix = dtype(deg) / n_pix
    pix_min = -deg/2. + 0.5 * dpix
    pix_max = deg/2.
    [Xm, Ym] = np.meshgrid(np.arange(pix_min,pix_max,dpix), np.arange(pix_min,pix_max,dpix));
    if sigma<=0:
        Zm = np.zeros_like(Xm)
    elif sigma<dpix:
        g_mass = np.vectorize(lambda a, b: gaussian_mass(a, b, dpix, dpix, x, y, sigma)) 
        Zm = g_mass(Xm, -Ym)        
    else:
        d = (2*dtype(sigma)**2)
        A = dtype(1. / (d*np.pi))
        Zm = dpix**2 * A * np.exp(-((Xm-x)**2 + (-Ym-y)**2) / d)
    return Xm, -Ym, Zm.astype(dtype)   
    
def make_gaussian_mass_stack(xs, ys, sigmas, n_pix, size=None, dtype=np.float32):
    stack_size = min(len(xs), len(ys), len(sigmas))
    assert stack_size>0
    Z = np.ndarray(shape=(stack_size, n_pix, n_pix), dtype=dtype)
    X,Y,Z[0,:,:] = make_gaussian_mass(xs[0], ys[0], sigmas[0], n_pix, size=size, dtype=dtype)
    for i in range(1,stack_size):
        _,_,Z[i,:,:] = make_gaussian_mass(xs[i], ys[i], sigmas[i], n_pix, size=size, dtype=dtype)
    return X, Y, Z
        
        
# File Utility
def zip_dict(*args):
    '''
    like zip but applies to multiple dicts with matching keys, returning a single key and all the corresponding values for that key.
    '''
    for a in args[1:]:
        assert (a.keys()==args[0].keys())
    for k in args[0].keys():
        yield [k,] + [a[k] for a in args]
        
# Torch fwRF
def get_value(_x):
    return np.copy(_x.data.cpu().numpy())

def set_value(_x, x):
    if list(x.shape)!=list(_x.size()):
        _x.resize_(x.shape)
    _x.data.copy_(torch.from_numpy(x))
    
def equalize_color(image):
    filt = ImageEnhance.Color(image)
    return filt.enhance(0.8)
