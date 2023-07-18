import os
import numpy as np
from scipy.special import erf
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import nibabel as nib
from nsd_access import NSDAccess
import torch
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim



def create_whole_region_unnormalized(subject = 1, whole=False, big=False):
    
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
    numScans = {1: 40, 2: 40, 3:32, 4: 30, 5:40, 6:32, 7:40, 8:30}
    if big:
        nsd_general = nib.load("masks/subject{}/nsdgeneral_big.nii.gz".format(subject)).get_fdata()
        nsd_general = np.nan_to_num(nsd_general)
        nsd_general = np.where(nsd_general==1.0, True, False)
    else:
        nsd_general = nib.load("masks/subject{}/brainmask_nsdgeneral_1.0.nii".format(subject)).get_fdata()
        
    layer_size = np.sum(nsd_general == True)
    print(nsd_general.shape)
    os.makedirs(prep_path + "subject{}/".format(subject), exist_ok=True)
    
    if(whole):
        data = numScans[subject]
        file = "subject{}/nsd_general_unnormalized_all.pt".format(subject)
        whole_region = torch.zeros((750*data, layer_size))
    else:
        data = numScans[subject]-3
        if big:
            file = "subject{}/nsd_general_unnormalized_big.pt".format(subject)
        else:
            file = "subject{}/nsd_general_unnormalized.pt".format(subject)
        whole_region = torch.zeros((750*data, layer_size))

    nsd_general_mask = np.nan_to_num(nsd_general)
    nsd_mask = np.array(nsd_general_mask.flatten(), dtype=bool)
    print(nsd_mask.shape)
    # Loads the full collection of beta sessions for subject 1
    for i in tqdm(range(1,data+1), desc="Loading Voxels"):
        beta = nsda.read_betas(subject="subj0" + str(subject), 
                                session_index=i, 
                                trial_index=[], # Empty list as index means get all 750 scans for this session (trial --> scan)
                                data_type='betas_fithrf_GLMdenoise_RR',
                                data_format='func1pt8mm')
            
        # Reshape the beta trails to be flattened. 
        beta = beta.reshape((nsd_mask.shape[0], beta.shape[3]))

        for j in range(beta.shape[1]):

            # Grab the current beta trail. 
            curScan = beta[:, j]
            
            single_scan = torch.from_numpy(curScan)

            # Discard the unmasked values and keeps the masked values. 
            whole_region[j + (i-1)*beta.shape[1]] = single_scan[nsd_mask]
            
    # Save the tensor
    print("SUBJECT {} WHOLE REGION SHAPE: {}".format(subject, whole_region.shape))
    torch.save(whole_region, prep_path + file)
    
    
def create_whole_region_normalized(subject = 1, whole = False, big=False):
    
    if(whole):

        whole_region = torch.load(prep_path + "subject{}/nsd_general_unnormalized_all.pt".format(subject))
    else:
        if big:
            whole_region = torch.load(prep_path + "subject{}/nsd_general_unnormalized_big.pt".format(subject))
        else:
            whole_region = torch.load(prep_path + "subject{}/nsd_general_unnormalized.pt".format(subject))
    whole_region_norm = torch.zeros_like(whole_region)
            
    # Normalize the data using Z scoring method for each voxel
    for i in range(whole_region.shape[1]):
        voxel_mean, voxel_std = torch.mean(whole_region[:, i]), torch.std(whole_region[:, i])  
        normalized_voxel = (whole_region[:, i] - voxel_mean) / voxel_std
        whole_region_norm[:, i] = normalized_voxel
    # Normalize the data by dividing all elements by the max of each voxel
    # whole_region_norm = whole_region / whole_region.max(0, keepdim=True)[0]

        # Save the tensor
    if(whole):
        torch.save(whole_region_norm, prep_path + "subject{}/nsd_general_all.pt".format(subject))
    else:
        if big:
            torch.save(whole_region_norm, prep_path + "subject{}/nsd_general_big.pt".format(subject))
        else:
            torch.save(whole_region_norm, prep_path + "subject{}/nsd_general.pt".format(subject))
    
def process_masks(subject=1, big=False):
    if big:
        nsd_general = nib.load(mask_path + "subject{}/nsdgeneral_big.nii.gz".format(subject)).get_fdata()
        nsd_general = np.nan_to_num(nsd_general)
        nsd_general = np.where(nsd_general==1.0, True, False)
        visual_rois = nib.load(mask_path + "subject{}/prf-visualrois_big.nii.gz".format(subject)).get_fdata()
    else:
        nsd_general = nib.load(mask_path + "subject{}/brainmask_nsdgeneral_1.0.nii".format(subject)).get_fdata()
        nsd_general = np.nan_to_num(nsd_general).astype(bool)
        visual_rois = nib.load(mask_path + "subject{}/prf-visualrois.nii.gz".format(subject)).get_fdata()
    
    
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
    
    flag = ""
    if big:
        flag = "_big"
    torch.save(V1, mask_path + "subject{}/V1{}.pt".format(subject, flag))
    torch.save(V2, mask_path + "subject{}/V2{}.pt".format(subject, flag))
    torch.save(V3, mask_path + "subject{}/V3{}.pt".format(subject, flag))
    torch.save(V4, mask_path + "subject{}/V4{}.pt".format(subject, flag))
    torch.save(early_vis, mask_path + "subject{}/early_vis{}.pt".format(subject, flag))
    torch.save(higher_vis, mask_path + "subject{}/higher_vis{}.pt".format(subject, flag))
    print("V1: ", np.unique(V1, return_counts=True))
    print("V2: ", np.unique(V2, return_counts=True))
    print("V3: ", np.unique(V3, return_counts=True))
    print("V4: ", np.unique(V4, return_counts=True))
    print("Early Vis: ", np.unique(early_vis, return_counts=True))
    print("Higher Vis: ", np.unique(higher_vis, return_counts=True))

    
def process_data(vector="c_img_uc", subject = 1):
    vecLength = torch.load(prep_path + "subject{}/nsd_general.pt".format(subject)).shape[0]
    print("VECTOR LENGTH: ", vecLength)
    if(vector == "images"):
        vec_target = torch.zeros((vecLength, 541875))
        datashape = (1, 541875)
    elif(vector == "c_img_uc"):
        vec_target = torch.zeros((vecLength, 1024))
        datashape = (1,1024)
    elif(vector == "c_text_uc"):
        vec_target = torch.zeros((vecLength, 78848))
        datashape = (1,78848)
    elif(vector == "z_vdvae"):
        vec_target = torch.zeros((vecLength, 91168))
        datashape = (1, 91168)
    print(vec_target.shape)
    # Loading the description object for subejct1
    subj = "subject" + str(subject)
    subjx = nsda.stim_descriptions[nsda.stim_descriptions[subj] != 0]
    full_vec = torch.load("/home/naxos2-raid25/kneel027/home/kneel027/nsd_local/preprocessed_data/{}_73k.pt".format(vector))
    for i in tqdm(range(0,vecLength), desc="vector loader subject{}".format(subject)):
        index = int(subjx.loc[(subjx[subj + "_rep0"] == i+1) | (subjx[subj + "_rep1"] == i+1) | (subjx[subj + "_rep2"] == i+1)].nsdId)
        vec_target[i] = full_vec[index].reshape(datashape)
    
    torch.save(vec_target, prep_path + "subject{}/{}.pt".format(subject, vector))
    
    
def process_raw_tensors(vector):
    
    # Intialize the vector variables 
    if(vector == "c_img_uc"):
        vec_target = torch.zeros((73000, 1024))
        datashape = (1,1024)
    elif(vector == "images"):
        vec_target = torch.zeros((73000, 541875))
        datashape = (1, 541875)
    elif(vector == "z_vdvae"):
        vec_target = torch.zeros((73000, 91168))
        datashape = (1, 91168)

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