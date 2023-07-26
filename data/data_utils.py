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

def read_images(image_index, show=False):
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
        data_folder = 'nsddata_betas/ppdata/{}/{}/{}'.format(subject, data_format, data_type)

        si_str = str(session_index).zfill(2)

        out_data = nb.load(
            op.join(data_folder, f'betas_session{si_str}.nii.gz')).get_data()

        if len(trial_index) == 0:
            trial_index = slice(0, out_data.shape[-1])

        return out_data[..., trial_index]

def create_whole_region_unnormalized(subject = 1):
    
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
    numScans = {1: 37, 2: 37, 3:32, 4: 30, 5:37, 6:32, 7:37, 8:30}
    nsd_general = nib.load("data/nsddata/ppdata/subj0{}/func1pt8mm/roi/nsdgeneral.nii.gz".format(subject)).get_fdata()
    nsd_general = np.nan_to_num(nsd_general)
    nsd_general = np.where(nsd_general==1.0, True, False)
        
    layer_size = np.sum(nsd_general == True)
    os.makedirs("data/preprocessed_data/subject{}/".format(subject), exist_ok=True)
    
    data = numScans[subject]
    file = "data/preprocessed_data/subject{}/nsd_general_unnormalized.pt".format(subject)
    whole_region = torch.zeros((750*data, layer_size))

    nsd_general_mask = np.nan_to_num(nsd_general)
    nsd_mask = np.array(nsd_general_mask.flatten(), dtype=bool)
    
    # Loads the full collection of beta sessions for subject 1
    for i in tqdm(range(1,data+1), desc="Loading Voxels"):
        beta = read_betas(subject="subj0" + str(subject), 
                                session_index=i, 
                                trial_index=[], # Empty list as index means get all 750 scans for this session (trial --> scan)
                                data_type='betas_fithrf_GLMdenoise_RR',
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
    print("SUBJECT {} WHOLE REGION SHAPE: {}".format(subject, whole_region.shape))
    torch.save(whole_region, file)
    
    
def create_whole_region_normalized(subject = 1):
    
    whole_region = torch.load("data/preprocessed_data/subject{}/nsd_general_unnormalized.pt".format(subject))
    whole_region_norm = torch.zeros_like(whole_region)
            
    # Normalize the data using Z scoring method for each voxel
    for i in range(whole_region.shape[1]):
        voxel_mean, voxel_std = torch.mean(whole_region[:, i]), torch.std(whole_region[:, i])  
        normalized_voxel = (whole_region[:, i] - voxel_mean) / voxel_std
        whole_region_norm[:, i] = normalized_voxel

    # Save the tensor of normailized data
    torch.save(whole_region_norm, "data/preprocessed_data/subject{}/nsd_general.pt".format(subject))
    
    # Delete NSD unnormalized file after the normalized data is created. 
    if(os.path.exists("data/preprocessed_data/subject{}/nsd_general_unnormalized.pt".format(subject))):
        os.remove("data/preprocessed_data/subject{}/nsd_general_unnormalized.pt".format(subject))
    
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
    
    print("V1: ", np.unique(V1, return_counts=True))
    print("V2: ", np.unique(V2, return_counts=True))
    print("V3: ", np.unique(V3, return_counts=True))
    print("V4: ", np.unique(V4, return_counts=True))
    print("Early Vis: ", np.unique(early_vis, return_counts=True))
    print("Higher Vis: ", np.unique(higher_vis, return_counts=True))

    
def process_data(vector="c_i", subject = 1):
    
    vecLength = torch.load("data/preprocessed_data/subject{}/nsd_general.pt".format(subject)).shape[0]
    print("VECTOR LENGTH: ", vecLength)
    
    if(vector == "images"):
        vec_target = torch.zeros((vecLength, 541875))
        datashape = (1, 541875)
        
    elif(vector == "c_i"):
        vec_target = torch.zeros((vecLength, 1024))
        datashape = (1,1024)
        
    elif(vector == "z_vdvae"):
        vec_target = torch.zeros((vecLength, 91168))
        datashape = (1, 91168)
        
    print(vec_target.shape)
    
    # Loading the description object for subejcts
    subj = "subject" + str(subject)
    stim_descriptions = pd.read_csv('data/nsddata/experiments/nsd/nsd_stim_info_merged.pkl', index_col=0)
    subjx = stim_descriptions[stim_descriptions[subj] != 0]
    full_vec = torch.load("data/preprocessed_data/{}_73k.pt".format(vector))
    
    for i in tqdm(range(0,vecLength), desc="vector loader subject{}".format(subject)):
        index = int(subjx.loc[(subjx[subj + "_rep0"] == i+1) | (subjx[subj + "_rep1"] == i+1) | (subjx[subj + "_rep2"] == i+1)].nsdId)
        vec_target[i] = full_vec[index].reshape(datashape)
    
    torch.save(vec_target, "data/preprocessed_data/subject{}/{}.pt".format(subject, vector))
    
    
def process_raw_tensors(vector):
    
    # Intialize the vector variables 
    if(vector == "c_i"):
        vec_target = torch.zeros((73000, 1024))
        datashape = (7300,1024)
    elif(vector == "images"):
        vec_target = torch.zeros((73000, 541875))
        datashape = (7300, 541875)
    elif(vector == "z_vdvae"):
        vec_target = torch.zeros((73000, 91168))
        datashape = (7300, 91168)

    # Create the 73000 tensor block of vector data
    for i in tqdm(range(10), desc="vector loader"):
        full_vec = torch.load("data/preprocessed_data/{}_{}.pt".format(vector, i)).reshape(datashape)
        vec_target[(i * 7300) : (i * 7300) + 7300] = full_vec
        
    # Save the 73000 tensor block
    torch.save(vec_target, "data/preprocessed_data/{}_73k.pt".format(vector))
    
    # Delete duplicate files after the tensor block of all images has been created. 
    if(os.path.exists("data/preprocessed_data/{}_73k.pt".format(vector))):
        for i in tqdm(range(10), desc="delete duplicate data"):
            os.remove("data/preprocessed_data/{}_{}.pt".format(vector, i))