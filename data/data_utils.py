import os
import os.path as op
import nibabel as nb
import numpy as np
import pandas as pd
from scipy.special import erf
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import nibabel as nib
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import h5py
import pickle

def read_images(image_index):
        """read_images reads a list of images, and returns their data

        Parameters
        ----------
        image_index : list of integers
            which images indexed in the 73k format to return
        show : bool, optional
            whether to also show the images, by default False

        Returns
        -------
        numpy.ndarray, 3D
            RGB image data
        """

        sf = h5py.File('data/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5', 'r')
        sdataset = sf.get('imgBrick')
        if show:
            f, ss = plt.subplots(1, len(image_index),
                                 figsize=(6*len(image_index), 6))
            if len(image_index) == 1:
                ss = [ss]
            for s, d in zip(ss, sdataset[image_index]):
                s.axis('off')
                s.imshow(d)
        return sdataset[image_index]

def read_betas(subject, session_index, trial_index=[], data_type='betas_fithrf_GLMdenoise_RR', data_format='fsaverage', mask=None):
        """read_betas read betas from MRI files

        Parameters
        ----------
        subject : str
            subject identifier, such as 'subj01'
        session_index : int
            which session, counting from 1
        trial_index : list, optional
            which trials from this session's file to return, by default [], which returns all trials
        data_type : str, optional
            which type of beta values to return from ['betas_assumehrf', 'betas_fithrf', 'betas_fithrf_GLMdenoise_RR', 'restingbetas_fithrf'], by default 'betas_fithrf_GLMdenoise_RR'
        data_format : str, optional
            what type of data format, from ['fsaverage', 'func1pt8mm', 'func1mm'], by default 'fsaverage'
        mask : numpy.ndarray, if defined, selects 'mat' data_format, needs volumetric data_format
            binary/boolean mask into mat file beta data format.

        Returns
        -------
        numpy.ndarray, 2D (fsaverage) or 4D (other data formats)
            the requested per-trial beta values
        """
        data_folder = 'data/nsddata_betas/ppdata/{}/{}/{}'.format(subject, data_format, data_type)

        si_str = str(session_index).zfill(2)

        out_data = nb.load(
            op.join(data_folder, f'betas_session{si_str}.nii.gz')).get_fdata()

        if len(trial_index) == 0:
            trial_index = slice(0, out_data.shape[-1])

        return out_data[..., trial_index]

def create_whole_region_unnormalized(subject = 1, include_heldout=False):
    if include_heldout:
        file = "data/preprocessed_data/subject{}/nsd_general_unnormalized_large.pt".format(subject)
        numScans = {1: 40, 2: 40, 3:32, 4: 30, 5:40, 6:32, 7:40, 8:30}
    else:
        file = "data/preprocessed_data/subject{}/nsd_general_unnormalized.pt".format(subject)
        numScans = {1: 37, 2: 37, 3:32, 4: 30, 5:37, 6:32, 7:37, 8:30}
    nsd_general = nib.load("data/nsddata/ppdata/subj0{}/func1pt8mm/roi/nsdgeneral.nii.gz".format(subject)).get_fdata()
    nsd_general = np.nan_to_num(nsd_general)
    nsd_general = np.where(nsd_general==1.0, True, False)
        
    layer_size = np.sum(nsd_general == True)
    os.makedirs("data/preprocessed_data/subject{}/".format(subject), exist_ok=True)
    
    data = numScans[subject]
    whole_region = torch.zeros((750*data, layer_size))

    nsd_general_mask = np.nan_to_num(nsd_general)
    nsd_mask = np.array(nsd_general_mask.flatten(), dtype=bool)
    
    # Loads the full collection of beta sessions for subject 1
    for i in tqdm(range(1,data+1), desc="Loading raw scanning session data"):
        beta = read_betas(subject="subj0" + str(subject), 
                                session_index=i, 
                                trial_index=[], # Empty list as index means get all 750 scans for this session (trial --> scan)
                                data_type="betas_fithrf_GLMdenoise_RR",
                                data_format='func1pt8mm')
            
        # Reshape the beta trails to be flattened. 
        beta = beta.reshape((nsd_mask.shape[0], beta.shape[3]))

        for j in range(beta.shape[1]):

            # Grab the current beta trail. 
            curScan = beta[:, j]
            
            # One scan session. 
            single_scan = torch.from_numpy(curScan)

            # Discard the unmasked values and keeps the masked values. 
            whole_region[j + (i-1)*beta.shape[1]] = single_scan[nsd_mask]
            
    # Save the tensor into the data directory. 
    torch.save(whole_region, file)

def create_whole_region_normalized(subject = 1, include_heldout=False):
    if include_heldout:
        file = "data/preprocessed_data/subject{}/nsd_general_large.pt".format(subject)
        whole_region = torch.load("data/preprocessed_data/subject{}/nsd_general_unnormalized_large.pt".format(subject))
        numScans = {1: 40, 2: 40, 3:32, 4: 30, 5:40, 6:32, 7:40, 8:30}
    else:
        file = "data/preprocessed_data/subject{}/nsd_general.pt".format(subject)
        whole_region = torch.load("data/preprocessed_data/subject{}/nsd_general_unnormalized.pt".format(subject))
        numScans = {1: 37, 2: 37, 3:32, 4: 30, 5:37, 6:32, 7:37, 8:30}
    
    whole_region_norm = torch.zeros_like(whole_region)
    
    stim_descriptions = pd.read_csv('data/nsddata/experiments/nsd/nsd_stim_info_merged.csv', index_col=0)
    subj_train = stim_descriptions[(stim_descriptions['subject{}'.format(subject)] != 0) & (stim_descriptions['shared1000'] == False)]
    train_ids = []
    for i in range(subj_train.shape[0]):
        for j in range(3):
            scanID = subj_train.iloc[i]['subject{}_rep{}'.format(subject, j)] - 1
            if scanID < numScans[subject]*750:
                train_ids.append(scanID)
    normalizing_data = whole_region[torch.tensor(train_ids)]
    print(normalizing_data.shape, whole_region.shape)
    # Normalize the data using Z scoring method for each voxel
    for i in range(normalizing_data.shape[1]):
        voxel_mean, voxel_std = torch.mean(normalizing_data[:, i]), torch.std(normalizing_data[:, i])  
        normalized_voxel = (whole_region[:, i] - voxel_mean) / voxel_std
        whole_region_norm[:, i] = normalized_voxel

    # Save the tensor of normalized data
    torch.save(whole_region_norm, file)
    #save to mindeye folder
    with h5py.File(f"/home/naxos2-raid25/kneel027/home/kneel027/MindEyeV2/new_betas/betas_all_subj0{subject}_fp32_renorm.hdf5", "w") as f:
        f.create_dataset("betas", data=whole_region_norm.numpy())
    
    # # Delete NSD unnormalized file after the normalized data is created. 
    # if(os.path.exists("data/preprocessed_data/subject{}/nsd_general_unnormalized.pt".format(subject))):
    #     os.remove("data/preprocessed_data/subject{}/nsd_general_unnormalized.pt".format(subject))

def create_whole_region_imagery_unnormalized(subject = 1, mask=True):
    
    nsd_general = nib.load("data/nsddata/ppdata/subj0{}/func1pt8mm/roi/nsdgeneral.nii.gz".format(subject)).get_fdata()
    nsd_general = np.nan_to_num(nsd_general)
    nsd_general = np.where(nsd_general==1.0, True, False)
        
    layer_size = np.sum(nsd_general == True)
    os.makedirs("data/preprocessed_data/subject{}/".format(subject), exist_ok=True)

    
    whole_region = np.zeros((720, layer_size))

    nsd_general_mask = np.nan_to_num(nsd_general)
    nsd_mask = np.array(nsd_general_mask.flatten(), dtype=bool)
    
    beta_file = "data/nsddata_betas/ppdata/subj0{}/func1pt8mm/nsdimagerybetas_fithrf/betas_nsdimagery.nii.gz".format(subject)

    imagery_betas = nib.load(beta_file).get_fdata()

    imagery_betas = imagery_betas.transpose((3,0,1,2))
    if mask:
        whole_region = torch.from_numpy(imagery_betas.reshape((len(imagery_betas), -1))[:,nsd_general.flatten()].astype(np.float32))
        file = "data/preprocessed_data/subject{}/nsd_imagery_unnormalized.pt".format(subject)
    else:
        whole_region = torch.from_numpy(imagery_betas.reshape((len(imagery_betas), -1)).astype(np.float32))
        file = "data/preprocessed_data/subject{}/nsd_imagery_unnormalized_unmasked.pt".format(subject)
    
    torch.save(whole_region, file)
    return whole_region

def create_whole_region_imagery_normalized(subject = 1, mask=True):
    img_stim_file = "data/nsddata_stimuli/stimuli/nsd/nsdimagery_stimuli.pkl3"
    ex_file = open(img_stim_file, 'rb')
    imagery_dict = pickle.load(ex_file)
    ex_file.close()
    exps = imagery_dict['exps']
    cues = imagery_dict['cues']
    meta_cond_idx = {
        'visA': np.arange(len(exps))[exps=='visA'],
        'visB': np.arange(len(exps))[exps=='visB'],
        'visC': np.arange(len(exps))[exps=='visC'],
        'imgA_1': np.arange(len(exps))[exps=='imgA_1'],
        'imgA_2': np.arange(len(exps))[exps=='imgA_2'],
        'imgB_1': np.arange(len(exps))[exps=='imgB_1'],
        'imgB_2': np.arange(len(exps))[exps=='imgB_2'],
        'imgC_1': np.arange(len(exps))[exps=='imgC_1'],
        'imgC_2': np.arange(len(exps))[exps=='imgC_2'],
        'attA': np.arange(len(exps))[exps=='attA'],
        'attB': np.arange(len(exps))[exps=='attB'],
        'attC': np.arange(len(exps))[exps=='attC'],
    }
    if mask:
        whole_region = torch.load("data/preprocessed_data/subject{}/nsd_imagery_unnormalized.pt".format(subject))
    else:
        whole_region = torch.load("data/preprocessed_data/subject{}/nsd_imagery_unnormalized_unmasked.pt".format(subject))
    whole_region = whole_region / 300.
    whole_region_norm = torch.zeros_like(whole_region)
            
    # Normalize the data using Z scoring method for each voxel
    for c,idx in meta_cond_idx.items():
        whole_region_norm[idx] = zscore(whole_region[idx])

    # Save the tensor of normalized data
    if mask:
        torch.save(whole_region_norm, "data/preprocessed_data/subject{}/nsd_imagery.pt".format(subject))
        # Delete NSD unnormalized file after the normalized data is created. 
        if(os.path.exists("data/preprocessed_data/subject{}/nsd_imagery_unnormalized.pt".format(subject))):
            os.remove("data/preprocessed_data/subject{}/nsd_imagery_unnormalized.pt".format(subject))
    else:
        torch.save(whole_region_norm, "data/preprocessed_data/subject{}/nsd_imagery_unmasked.pt".format(subject))
        # Delete NSD unnormalized file after the normalized data is created. 
        if(os.path.exists("data/preprocessed_data/subject{}/nsd_imagery_unnormalized_unmasked.pt".format(subject))):
            os.remove("data/preprocessed_data/subject{}/nsd_imagery_unnormalized_unmasked.pt".format(subject))
    
    
    

def process_masks(subject=1):
    
    os.makedirs("data/preprocessed_data/subject{}/masks".format(subject), exist_ok=True)
    nsd_general = nib.load("data/nsddata/ppdata/subj0{}/func1pt8mm/roi/nsdgeneral.nii.gz".format(subject)).get_fdata()
    nsd_general = np.nan_to_num(nsd_general)
    nsd_general = np.where(nsd_general==1.0, True, False)
    visual_rois = nib.load("data/nsddata/ppdata/subj0{}/func1pt8mm/roi/prf-visualrois.nii.gz".format(subject)).get_fdata()
    
    V1L = np.where(visual_rois==1.0, True, False)
    V1R = np.where(visual_rois==2.0, True, False)
    V1 = torch.from_numpy(V1L[nsd_general] + V1R[nsd_general])
    V2L = np.where(visual_rois==3.0, True, False)
    V2R = np.where(visual_rois==4.0, True, False)
    V2 = torch.from_numpy(V2L[nsd_general] + V2R[nsd_general])
    V3L = np.where(visual_rois==5.0, True, False)
    V3R = np.where(visual_rois==6.0, True, False)
    V3 = torch.from_numpy(V3L[nsd_general] + V3R[nsd_general])
    V4 = np.where(visual_rois==7.0, True, False)
    V4 = torch.from_numpy(V4[nsd_general])
    early_vis = V1 + V2 + V3 + V4
    higher_vis = ~early_vis
    
    torch.save(V1, "data/preprocessed_data/subject{}/masks/V1.pt".format(subject))
    torch.save(V2, "data/preprocessed_data/subject{}/masks/V2.pt".format(subject))
    torch.save(V3, "data/preprocessed_data/subject{}/masks/V3.pt".format(subject))
    torch.save(V4, "data/preprocessed_data/subject{}/masks/V4.pt".format(subject))
    torch.save(early_vis, "data/preprocessed_data/subject{}/masks/early_vis.pt".format(subject))
    torch.save(higher_vis,"data/preprocessed_data/subject{}/masks/higher_vis.pt".format(subject))

    
def process_data(vector="c", include_heldout=False):
    full_vec = torch.load("data/preprocessed_data/{}_73k.pt".format(vector))
    stim_descriptions = pd.read_csv('data/nsddata/experiments/nsd/nsd_stim_info_merged.csv', index_col=0)
    for subject in tqdm(range(1,9), desc="processing data"):
        if include_heldout:
            vecLength = torch.load("data/preprocessed_data/subject{}/nsd_general_large.pt".format(subject)).shape[0]
        else:
            vecLength = torch.load("data/preprocessed_data/subject{}/nsd_general.pt".format(subject)).shape[0]
        
        if(vector == "images"):
            vec_target = torch.zeros((vecLength, 541875))
            datashape = (1, 541875)
            
        elif(vector == "c"):
            vec_target = torch.zeros((vecLength, 1024))
            datashape = (1,1024)
            
        elif(vector == "z_vdvae"):
            vec_target = torch.zeros((vecLength, 91168))
            datashape = (1, 91168)
        
        # Loading the description object for subejcts
        subj = "subject" + str(subject)
        
        subjx = stim_descriptions[stim_descriptions[subj] != 0]
        
        for i in tqdm(range(0,vecLength), desc="arranging {} training data for subject{}".format(vector, subject)):
            index = int(subjx.loc[(subjx[subj + "_rep0"] == i+1) | (subjx[subj + "_rep1"] == i+1) | (subjx[subj + "_rep2"] == i+1)].nsdId)
            vec_target[i] = full_vec[index].reshape(datashape)
        
        torch.save(vec_target, "data/preprocessed_data/subject{}/{}.pt".format(subject, vector))
    
    
def process_raw_tensors(vector):
    
    # Intialize the vector variables 
    if(vector == "c"):
        vec_target = torch.zeros((73000, 1024))
        datashape = (7300,1024)
    elif(vector == "images"):
        vec_target = torch.zeros((73000, 541875))
        datashape = (7300, 541875)
    elif(vector == "z_vdvae"):
        vec_target = torch.zeros((73000, 91168))
        datashape = (7300, 91168)

    # Create the 73000 tensor block of vector data
    for i in tqdm(range(10), desc="concatenating tensors"):
        full_vec = torch.load("data/preprocessed_data/{}_{}.pt".format(vector, i)).reshape(datashape)
        vec_target[(i * 7300) : (i * 7300) + 7300] = full_vec
        
    # Save the 73000 tensor block
    torch.save(vec_target, "data/preprocessed_data/{}_73k.pt".format(vector))
    
    # Delete duplicate files after the tensor block of all images has been created. 
    if(os.path.exists("data/preprocessed_data/{}_73k.pt".format(vector))):
        for i in tqdm(range(10), desc="deleting duplicate data"):
            os.remove("data/preprocessed_data/{}_{}.pt".format(vector, i))

def zip_dict(*args):
    '''
    like zip but applies to multiple dicts with matching keys, returning a single key and all the corresponding values for that key.
    '''
    for a in args[1:]:
        assert (a.keys()==args[0].keys())
    for k in args[0].keys():
        yield [k,] + [a[k] for a in args]

def zscore(x, mean=None, stddev=None, return_stats=False):
    if mean is not None:
        m = mean
    else:
        m = torch.mean(x, axis=0, keepdims=True)
    if stddev is not None:
        s = stddev
    else:
        s = torch.std(x, axis=0, keepdims=True)
    if return_stats:
        return (x - m)/(s+1e-6), m, s
    else:
        return (x - m)/(s+1e-6)

def format_imagery_stimuli():
    path  = "data/nsddata_stimuli/stimuli/imagery_images/"
    img_tensor = torch.zeros((18, 541875))
    for i in range(18):
        im = Image.open(path + "{}.png".format(i))
        im = im.resize((425, 425))
        im = np.array(im).flatten()
        img_tensor[i] = torch.from_numpy(im)
    return img_tensor

def search_and_open_image(directory, img_identifier):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if img_identifier in file:
                image_path = os.path.join(root, file)
                image = Image.open(image_path)
                return image
    return None