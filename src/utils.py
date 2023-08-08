import os
import numpy as np
from scipy.special import erf
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import nibabel as nib
import torch
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import pandas as pd
import h5py
import matplotlib.pyplot as plt

prep_path = "/export/raid1/home/kneel027/nsd_local/preprocessed_data/"
latent_path = "/export/raid1/home/kneel027/Second-Sight/latent_vectors/"
mask_path = "/export/raid1/home/kneel027/Second-Sight/masks/"


# self.stimuli_file = op.join(
#     self.nsd_folder, 'nsddata_stimuli', 'stimuli', 'nsd', 'nsd_stimuli.hdf5')

# self.stimuli_description_file = op.join(
#     self.nsd_folder, 'nsddata', 'experiments', 'nsd', 'nsd_stim_info_merged.csv')

# self.stim_descriptions = pd.read_csv(
#         self.stimuli_description_file, index_col=0)

# self.coco_annotation_file = op.join(
#     self.local_folder, 'nsddata_stimuli', 'stimuli', 'nsd', 'annotations', '{}_{}.json')


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

# Main data loader, 
# Loader = True
#    - Returns the train and test data loader
# Loader = False
#    - Returns the x_train, x_val, x_test, y_train, y_val, y_test

def load_nsd(vector, subject=1, batch_size=64, num_workers=4, loader=True, split=True, ae=False, encoderModel=None, average=False, nest=False):
    nsd_general = "nsd_general.pt"
    if(ae):
        assert encoderModel is not None
        x = torch.load("data/preprocessed_data/subject{}/nsd_general.pt".format(subject)).requires_grad_(False).to("cpu")
        y = torch.load("data/preprocessed_data/subject{}/{}_ae_beta_primes".format(subject, encoderModel)).requires_grad_(False).to("cpu")
    else:
        x = torch.load("data/preprocessed_data/subject{}/nsd_general.pt".format(subject)).requires_grad_(False)
        y = torch.load("data/preprocessed_data/subject{}/{}.pt".format(subject, vector)).requires_grad_(False)
    if(not split): 
        if(loader):
            dataset = torch.utils.data.TensorDataset(x, y)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
            return dataloader
        else:
            return x, y
    x_train, x_val, x_test = [], [], []
    y_train, y_val, y_test = [], [], []
    stim_descriptions = pd.read_csv('data/nsddata/experiments/nsd/nsd_stim_info_merged.csv', index_col=0)
    subj_train = stim_descriptions[(stim_descriptions['subject{}'.format(subject)] != 0) & (stim_descriptions['shared1000'] == False)]
    subj_test = stim_descriptions[(stim_descriptions['subject{}'.format(subject)] != 0) & (stim_descriptions['shared1000'] == True)]
    test_trials = []
    # 
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

    
#useTitle = 0 means no title at all
#useTitle = 1 means normal centered title at the top
#useTitle = 2 means title uses the captions list for a column wise title
#rewrite this function to be better designed and more general
def tileImages(title=None, images=None, captions=None, h=None, w=None, useTitle=True, rowCaptions=True, background_color='white', buffer=0, redCol=False):
    imW, imH = images[0].size
    bigW = (imW + buffer) * w
    if(rowCaptions):
        bigH = (imH + 64 + buffer) * h 
        rStep = imH + 64 + buffer
    else:
        bigH = (imH + buffer) * h
        rStep = imH + buffer
    if useTitle:
        hStart = 128
        height = bigH + 128 + buffer
    else:
        hStart = 0
        height = bigH + buffer
    if redCol:
        bigW += buffer
    cStep = imW + buffer
    hStart += buffer
    bigW += buffer
    bigH += buffer
    canvas = Image.new('RGB', (bigW, height), color=background_color)
    if redCol:
        red = Image.new('RGB', (imW + 2*buffer, bigH+buffer), color='red')
        canvas.paste(red, (0,hStart-buffer))
    font = ImageFont.truetype("arial.ttf", 42)
    titleFont = ImageFont.truetype("arial.ttf", 75)
    textLabeler = ImageDraw.Draw(canvas)
    if useTitle == 1:
        _, _, w, h = textLabeler.textbbox((0, 0), title, font=titleFont)
        textLabeler.text(((bigW-w)/2, 32), title, font=titleFont, fill='black')
    elif useTitle == 2:
        for i in range(w):
            _, _, w, h = textLabeler.textbbox((0, 0), captions[i], font=titleFont)
            textLabeler.text((i*cStep + (imW+buffer-w)/2, 38), captions[i], font=titleFont, fill='black')
    label = Image.new(mode="RGBA", size=(imW,64), color="white")
    count = 0
    for j in range(hStart, bigH-buffer, rStep):
        for i in range(buffer, bigW-buffer, cStep):
            if(count < len(images)):
                canvas.paste(images[count].resize((imH, imW), resample=Image.Resampling.LANCZOS), (i,j))
                if(rowCaptions):
                    canvas.paste(label, (i, j+imH))
                    textLabeler.text((i+16, j+imH+8), captions[count], font=font, fill='black')
                count+=1
    if redCol:
        sub_col = canvas.crop((cStep + buffer, 0, bigW, height))
        buf = Image.new('RGB', (buffer, height), color='white')
        canvas.paste(sub_col, (cStep + buffer + buffer, 0))
        canvas.paste(buf, (cStep + buffer, 0))
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
    stim_descriptions = pd.read_csv('data/nsddata/experiments/nsd/nsd_stim_info_merged.csv', index_col=0)
    subj = stim_descriptions[stim_descriptions["subject" + str(subject)] != 0]
    nsdIds = set(subj['nsdId'].tolist())
    
    pruned_x = torch.zeros((73000-len(nsdIds), x.shape[1]))
    count = 0
    for pred in range(73000):
        if pred not in nsdIds:
            pruned_x[count] = x[pred]
            count+=1
    return pruned_x

def get_pruned_indices(subject=1):
    stim_descriptions = pd.read_csv('data/nsddata/experiments/nsd/nsd_stim_info_merged.csv', index_col=0)
    subj = stim_descriptions[stim_descriptions["subject" + str(subject)] != 0]
    nsdIds = set(subj['nsdId'].tolist())
    
    pruned_indices = torch.zeros((73000-len(nsdIds), ), dtype=int)
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
    stim_descriptions = pd.read_csv('data/nsddata/experiments/nsd/nsd_stim_info_merged.csv', index_col=0)
    df = stim_descriptions[(stim_descriptions['subject1']) & (stim_descriptions['shared1000'] == True)]
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
    stim_descriptions = pd.read_csv('data/nsddata/experiments/nsd/nsd_stim_info_merged.csv', index_col=0)
    subj_test = stim_descriptions[(stim_descriptions['subject1'] != 0) & (stim_descriptions['shared1000'] == True)]
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


def read_images(self, image_index, show=False):
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

    if not hasattr(self, 'stim_descriptions'):
        self.stim_descriptions = pd.read_csv(
            self.stimuli_description_file, index_col=0)

    sf = h5py.File(self.stimuli_file, 'r')
    sdataset = sf.get('imgBrick')
    if show:
        f, ss = plt.subplots(1, len(image_index),
                                figsize=(6*len(image_index), 6))
        if len(image_index) == 1:
            ss = [ss]
        for s, d in zip(ss, sdataset[image_index]):
            s.axis('off')
            s.imshow(d)
        ss.close()
    return sdataset[image_index]


