import os
import numpy as np
from scipy.special import erf
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import nibabel as nib
from nsd_access import NSDAccess
import torch
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim


prep_path = "/export/raid1/home/kneel027/nsd_local/preprocessed_data/"
latent_path = "/export/raid1/home/kneel027/Second-Sight/latent_vectors/"
mask_path = "/export/raid1/home/kneel027/Second-Sight/masks/"

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

def load_nsd(vector, subject=1, batch_size=64, num_workers=4, loader=True, split=True, ae=False, encoderModel=None, average=False, nest=False, big=True):
    if(big):
        nsd_general = "nsd_general_big.pt"
    else:
        nsd_general = "nsd_general.pt"
    if(ae):
        assert encoderModel is not None
        x = torch.load(prep_path + "subject{}/{}".format(subject, nsd_general)).requires_grad_(False).to("cpu")
        y = torch.load(prep_path + "subject{}/x_encoded/{}".format(subject, encoderModel)).requires_grad_(False).to("cpu")
    else:
        x = torch.load(prep_path + "subject{}/{}".format(subject, nsd_general)).requires_grad_(False)
        y = torch.load(prep_path + "subject{}/{}.pt".format(subject, vector)).requires_grad_(False)
    
    if(not split): 
        if(loader):
            dataset = torch.utils.data.TensorDataset(x, y)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
            return dataloader
        else:
            return x, y
    x_train, x_val, x_test = [], [], []
    y_train, y_val, y_test = [], [], []
    subj_train = nsda.stim_descriptions[(nsda.stim_descriptions['subject{}'.format(subject)] != 0) & (nsda.stim_descriptions['shared1000'] == False)]
    subj_test = nsda.stim_descriptions[(nsda.stim_descriptions['subject{}'.format(subject)] != 0) & (nsda.stim_descriptions['shared1000'] == True)]
    test_trials = []
    split_point = int(subj_train.shape[0]*0.85)
    for i in tqdm(range(split_point), desc="loading training samples"):
        for j in range(3):
            scanId = subj_train.iloc[i]['subject{}_rep{}'.format(subject, j)]
            if(scanId < x.shape[0]):
                x_train.append(x[scanId-1])
                y_train.append(y[scanId-1])
                        
    for i in tqdm(range(split_point, subj_train.shape[0]), desc="loading validation samples"):
        for j in range(3):
            scanId = subj_train.iloc[i]['subject{}_rep{}'.format(subject, j)]
            if(scanId < x.shape[0]):
                x_val.append(x[scanId-1])
                y_val.append(y[scanId-1])
    for i in range(subj_test.shape[0]):
        nsdId = subj_test.iloc[i]['nsdId']
        avx = []
        avy = []
        x_row = torch.zeros((3, x.shape[1]))
        for j in range(3):
            scanId = subj_test.iloc[i]['subject{}_rep{}'.format(subject, j)]
            if(scanId < x.shape[0]):
                if average or nest:
                    avx.append(x[scanId-1])
                    avy.append(y[scanId-1])
                else:
                    x_test.append(x[scanId-1])
                    y_test.append(y[scanId-1])
                    test_trials.append(nsdId)
        if(len(avy)>0):
            # if len(avy) < 3:
                # print("WARNING: Missing data for trial {}".format(i-194))
            if average:
                avx = torch.stack(avx)
                x_test.append(torch.mean(avx, dim=0))
            else:
                for j in range(len(avx)):
                    # if len(avy) < 3:
                        # print("sample: {}, nonzero: {}".format(i-194, torch.sum(torch.count_nonzero(avx[j]))))
                    x_row[j] = avx[j]
                x_test.append(x_row)
            y_test.append(avy[0])
            test_trials.append(nsdId)
    x_train = torch.stack(x_train).to("cpu")
    x_val = torch.stack(x_val).to("cpu")
    x_test = torch.stack(x_test).to("cpu")
    y_train = torch.stack(y_train)
    y_val = torch.stack(y_val)
    y_test = torch.stack(y_test)
    print("shapes: ", x_train.shape, x_val.shape, x_test.shape, y_train.shape, y_val.shape, y_test.shape)
    if(loader):
        trainset = torch.utils.data.TensorDataset(x_train, y_train)
        valset = torch.utils.data.TensorDataset(x_val, y_val)
        testset = torch.utils.data.TensorDataset(x_test, y_test)
        # Loads the Dataset into a DataLoader
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        return trainloader, valloader, testloader
    else:
        return x_train, x_val, x_test, y_train, y_val, y_test, test_trials

def ghislain_stimuli_ordering(subject=1):
    subj_full = nsda.stim_descriptions[(nsda.stim_descriptions['subject{}'.format(subject)] != 0)]
    subj_test = nsda.stim_descriptions[(nsda.stim_descriptions['subject{}'.format(subject)] != 0) & (nsda.stim_descriptions['shared1000'] == True)]
    stimuli_order_list = np.where(subj_full["shared1000"] == True)[0]
    stimuli_ordering  = []
    for i in range(len(stimuli_order_list)):
        for j in range(3):
            scanId = subj_test.iloc[i]['subject1_rep{}'.format(j)]
            if(scanId < 27750):
                stimuli_ordering.append(stimuli_order_list[i])
    return stimuli_ordering


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
    
def get_images(subject=1):
    images = []
    subj = "subject" + str(subject)
    subjx = nsda.stim_descriptions[nsda.stim_descriptions[subj] != 0]
    vecLength = torch.load(prep_path + "subject{}/nsd_general.pt".format(subject)).shape[0]
    for i in tqdm(range(0,vecLength), desc="image loader"):
        index = int(subjx.loc[(subjx[subj + "_rep0"] == i+1) | (subjx[subj + "_rep1"] == i+1) | (subjx[subj + "_rep2"] == i+1)].nsdId)
        ground_truth_image_np_array = nsda.read_images([index], show=False)
        ground_truth_PIL = Image.fromarray(ground_truth_image_np_array[0])
        images.append(ground_truth_PIL)
    return images
    
def process_raw_tensors(vector):
    if(vector == "c_img_uc"):
        vec_target = torch.zeros((73000, 1024))
        datashape = (1,1024)
    elif(vector == "c_text_uc"):
        vec_target = torch.zeros((73000, 78848))
        datashape = (1,78848)
    elif(vector == "images"):
        vec_target = torch.zeros((73000, 541875))
        datashape = (1, 541875)
    elif(vector == "z_vdvae"):
        vec_target = torch.zeros((73000, 91168))
        datashape = (1, 91168)

    for i in tqdm(range(73000), desc="vector loader"):
        full_vec = torch.load("/export/raid1/home/kneel027/nsd_local/nsddata_stimuli/tensors/{}/{}.pt".format(vector, i)).reshape(datashape)
        vec_target[i] = full_vec
    torch.save(vec_target, prep_path + vector + "_73k.pt")
    
def process_x_encoded(Encoder):
    with torch.no_grad():
        modelId = Encoder.hashNum + "_model_" + Encoder.vector + ".pt"
        os.makedirs("latent_vectors/subject{}/{}".format(Encoder.subject, modelId), exist_ok=True)
        coco_full = torch.load("/export/raid1/home/kneel027/nsd_local/preprocessed_data/{}_73k.pt".format(Encoder.vector))
        coco_preds_full = torch.zeros((73000, Encoder.x_size))
        for i in tqdm(range(4), desc="predicting images"):
            coco_preds_full[18250*i:18250*i + 18250] = Encoder.predict(coco_full[18250*i:18250*i + 18250]).cpu()
        pruned_encodings = prune_vector(coco_preds_full)
        torch.save(pruned_encodings, "latent_vectors/subject{}/{}/coco_brain_preds.pt".format(Encoder.subject, modelId))
        
        os.makedirs(prep_path + "subject{}/x_encoded/".format(Encoder.subject), exist_ok=True)
        _, y = load_nsd(vector = Encoder.vector, subject=Encoder.subject, loader = False, split = False)
        outputs = Encoder.predict(y)
        torch.save(outputs, prep_path + "subject{}/x_encoded/{}".format(Encoder.subject, modelId))

#useTitle = 0 means no title at all
#useTitle = 1 means normal centered title at the top
#useTitle = 2 means title uses the captions list for a column wise title
#rewrite this function to be better designed and more general
def tileImages(title=None, images=None, captions=None, h=None, w=None, useTitle=True, rowCaptions=True):
    imW, imH = images[0].size
    bigW = imW * w
    if(rowCaptions):
        bigH = (imH + 64) * h 
        rStep = imH + 64
    else:
        bigH = imH * h
        rStep = imH
    if useTitle:
        hStart = 128
        height = bigH + 128
    else:
        hStart = 0
        height = bigH

    canvas = Image.new('RGB', (bigW, height), color='white')
    font = ImageFont.truetype("arial.ttf", 36)
    titleFont = ImageFont.truetype("arial.ttf", 75)
    textLabeler = ImageDraw.Draw(canvas)
    if useTitle == 1:
        _, _, w, h = textLabeler.textbbox((0, 0), title, font=titleFont)
        textLabeler.text(((bigW-w)/2, 32), title, font=titleFont, fill='black')
    elif useTitle == 2:
        for i in range(w):
            _, _, w, h = textLabeler.textbbox((0, 0), captions[i], font=titleFont)
            textLabeler.text((i*imH + (imW-w)/2, 16), captions[i], font=titleFont, fill='black')
    label = Image.new(mode="RGBA", size=(imW,64), color="white")
    count = 0
    for j in range(hStart, bigH, rStep):
        for i in range(0, bigW, imW):
            if(count < len(images)):
                canvas.paste(images[count].resize((imH, imW), resample=Image.Resampling.LANCZOS), (i,j))
                if(rowCaptions):
                    canvas.paste(label, (i, j+imH))
                    textLabeler.text((i+32, j+imH+8), captions[count], font=font, fill='black')
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


# SCS Performance Metrics

def mse_scs(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

def pixel_correlation(imageA, imageB):
    a = np.array(imageA.resize((425, 425), resample=Image.Resampling.LANCZOS)).flatten()
    b = np.array(imageB.resize((425, 425), resample=Image.Resampling.LANCZOS)).flatten()
    return (np.corrcoef(a,b))[0][1]

#converts a torch tensor of an 425z425vimage into a PIL image and resizes it
def process_image(imageArray, x=768, y=768):
    imageArray = imageArray.reshape((425, 425, 3)).cpu().numpy().astype(np.uint8)
    image = Image.fromarray(imageArray)
    image = image.resize((x, y), resample=Image.Resampling.LANCZOS)
    return image

def prune_vector(x, subject=1):
    subj = nsda.stim_descriptions[nsda.stim_descriptions["subject" + str(subject)] != 0]
    nsdIds = set(subj['nsdId'].tolist())
    
    pruned_x = torch.zeros((73000-len(nsdIds), x.shape[1]))
    count = 0
    for pred in range(73000):
        if pred not in nsdIds:
            pruned_x[count] = x[pred]
            count+=1
    return pruned_x

def get_pruned_indices(subject=1):
    subj = nsda.stim_descriptions[nsda.stim_descriptions["subject" + str(subject)] != 0]
    nsdIds = set(subj['nsdId'].tolist())
    
    pruned_indices = torch.zeros((73000-len(nsdIds), ))
    count = 0
    for pred in range(73000):
        if pred not in nsdIds:
            pruned_indices[count] = pred
            count+=1
    return pruned_indices
    
def slerp(q1, q2, u):
    """Spherical Linear intERPolation."""
    output_shape = q1.shape
    output_type = q1.dtype
    output_device = q1.device
    q1 = q1.flatten().to(dtype=output_type, device=output_device)
    q2 = q2.flatten().to(dtype=output_type, device=output_device)
    cos_theta = torch.dot(q1, q2)
    if cos_theta < 0:
        q1, cos_theta = -q1, -cos_theta

    if cos_theta > 0.9995:
        return torch.lerp(q1, q2, u)

    theta = torch.acos(cos_theta)
    sin_theta = torch.sin(theta)
    a = torch.sin((1 - u) * theta) / sin_theta
    b = torch.sin(u * theta) / sin_theta
    ret = a * q1 + b * q2
    return ret.reshape(output_shape).to(dtype=output_type, device=output_device)


def normalize_vdvae(v):
    latent_mean = torch.load("vdvae/train_mean.pt").to(v.device)
    latent_std = torch.load("vdvae/train_std.pt").to(v.device)

    outputs_vdvae_norm = (v - torch.mean(v, dim=0)) / torch.std(v, dim=0)
    outputs_vdvae_norm = (outputs_vdvae_norm * latent_std) + latent_mean
    outputs_vdvae_norm = outputs_vdvae_norm.reshape((v.shape[0], 1, 91168))
    return outputs_vdvae_norm

# Convert indicides between nsdID sorted and scanID sorted, goes the other way with the reverse flag

def convert_indices(idx, reverse=False, held_out=False):
    df = nsda.stim_descriptions[(nsda.stim_descriptions['subject1']) & (nsda.stim_descriptions['shared1000'] == True)]
    if reverse:
        sorted_df = df.copy()
        df = df.sort_values(by='subject1_rep0')
    else:
        sorted_df = df.sort_values(by='subject1_rep0')
    nsdIds = sorted_df["nsdId"].tolist()
    # print(len(sorted_df))
    output_idx = []
    for i in idx:
        nsd = df.iloc[i].nsdId
        output_idx.append(nsdIds.index(nsd))
    return output_idx

#Remove indices not in heldout 3 scan sessions
def remove_heldout_indices(idx, scanId_sorted=True):
    
    subj_test = nsda.stim_descriptions[(nsda.stim_descriptions['subject1'] != 0) & (nsda.stim_descriptions['shared1000'] == True)]
    if(scanId_sorted):
        subj_test = subj_test.sort_values(by='subject1_rep0')
    sample_count = 0
    index_list = []
    for i in range(subj_test.shape[0]):
        heldout = True
        for j in range(3):
            scanId = subj_test.iloc[i]['subject{}_rep{}'.format(1, j)]
            if scanId < 27750:
                heldout = False
        if heldout == False:
            index_list.append(sample_count)
            sample_count += 1
        else:
            index_list.append(-1)
    converted_indices = []
    for i in idx:
        if index_list[i] != -1:
            converted_indices.append(index_list[i])
    return converted_indices
