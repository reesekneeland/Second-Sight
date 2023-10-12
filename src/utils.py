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
import json
import scipy as sp
import pickle

def read_images(image_index):

    sf = h5py.File('data/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5', 'r')
    sdataset = sf.get('imgBrick')
    return sdataset[image_index]

# Main data loader, 
# vector: required parameter for the type of vector you want to load, options are: "c", "images", "z_vdvae"
# subject: required parameter for which subjects data to load, options are: 1, 2, 5, 7
# loader: flag to return dataloaders instead of raw data tensors
# ae: flag to return data used for training the autoencoder
# encoderModel: if ae flag is enabled, this is a required parameter specifying which encoding model's predictions to use for the autoencoder target
# average: flag to determine whether the brain data is averaged across the trial repetitions
# nest: if loader is False, changes the shape of the data structure to keep sample repetitions together
# batch_size: only used if loader is True, determines dataloader batch size
# num_workers: only used if loader is True, determines num_workers for dataloader
def load_nsd(vector, subject, loader=True, ae=False, encoderModel=None, average=False, nest=False, batch_size=64, num_workers=4):
    # If loading autoencoded data, load raw x as brain data (beta) and raw y as encoded brain data (beta prime)
    if(ae):
        assert encoderModel is not None
        x = torch.load("data/preprocessed_data/subject{}/nsd_general.pt".format(subject)).requires_grad_(False).to("cpu")
        y = torch.load("data/preprocessed_data/subject{}/{}_ae_beta_primes.pt".format(subject, encoderModel)).requires_grad_(False).to("cpu")
    # Load raw x as brain data and raw y as the provided vector, either c for a CLIP vector or images 
    else:
        x = torch.load("data/preprocessed_data/subject{}/nsd_general.pt".format(subject)).requires_grad_(False).to("cpu")
        y = torch.load("data/preprocessed_data/subject{}/{}.pt".format(subject, vector)).requires_grad_(False).to("cpu")
    x_train, x_val, x_test = [], [], []
    y_train, y_val, y_test = [], [], []
    # Preparing dataframe to help separate the shared1000 test data
    stim_descriptions = pd.read_csv('data/nsddata/experiments/nsd/nsd_stim_info_merged.csv', index_col=0)
    subj_train = stim_descriptions[(stim_descriptions['subject{}'.format(subject)] != 0) & (stim_descriptions['shared1000'] == False)]
    subj_test = stim_descriptions[(stim_descriptions['subject{}'.format(subject)] != 0) & (stim_descriptions['shared1000'] == True)]
    test_trials = []
    pbar = tqdm(desc="loading samples", total=27749)
    split_point = int(subj_train.shape[0]*0.85)
    # Collect 85% of the non-test data for the training set
    for i in range(split_point):
        for j in range(3):
            scanId = subj_train.iloc[i]['subject{}_rep{}'.format(subject, j)]
            # tqdm.write(str(scanId))
            if(scanId < x.shape[0]):
                x_train.append(x[scanId-1])
                y_train.append(y[scanId-1])
                pbar.update() 
    # Collect 15% of the non-test data for the validation set
    for i in range(split_point, subj_train.shape[0]):
        for j in range(3):
            scanId = subj_train.iloc[i]['subject{}_rep{}'.format(subject, j)]
            if(scanId < x.shape[0]):
                x_val.append(x[scanId-1])
                y_val.append(y[scanId-1])
                pbar.update() 
    # Collect test data
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
                pbar.update() 
        # Setup nested or averaged data structure if flags are passed
        if(len(avy)>0):
            if average:
                avx = torch.stack(avx)
                x_test.append(torch.mean(avx, dim=0))
            else:
                for j in range(len(avx)):
                    x_row[j] = avx[j]
                x_test.append(x_row)
            y_test.append(avy[0])
            test_trials.append(nsdId)
    # Concatenate data into tensors
    x_train = torch.stack(x_train).to("cpu")
    x_val = torch.stack(x_val).to("cpu")
    x_test = torch.stack(x_test).to("cpu")
    y_train = torch.stack(y_train).to("cpu")
    y_val = torch.stack(y_val).to("cpu")
    y_test = torch.stack(y_test).to("cpu")
    tqdm.write("Data Shapes... x_train: {}, x_val: {}, x_test: {}, y_train: {}, y_val: {}, y_test: {}".format(x_train.shape, x_val.shape, x_test.shape, y_train.shape, y_val.shape, y_test.shape))
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

#stimtype: all, simple, complex
#mode: vision, imagery

def load_nsd_mental_imagery(vector, subject, mode, stimtype="all", average=False, nest=False):
    
    img_stim_file = "data/nsddata_stimuli/stimuli/nsd/nsdimagery_stimuli.pkl3"
    ex_file = open(img_stim_file, 'rb')
    imagery_dict = pickle.load(ex_file)
    ex_file.close()
    exps = imagery_dict['exps']
    cues = imagery_dict['cues']
    image_map  = imagery_dict['image_map']
    image_data = imagery_dict['image_data']
    cond_idx = {
    'visionsimple': np.arange(len(exps))[exps=='visA'],
    'visioncomplex': np.arange(len(exps))[exps=='visB'],
    'visionall': np.arange(len(exps))[np.logical_or(exps=='visA', exps=='visB')],
    'imagerysimple': np.arange(len(exps))[np.logical_or(exps=='imgA_1', exps=='imgA_2')],
    'imagerycomplex': np.arange(len(exps))[np.logical_or(exps=='imgB_1', exps=='imgB_2')],
    'imageryall': np.arange(len(exps))[np.logical_or(np.logical_or(exps=='imgA_1', exps=='imgA_2'), np.logical_or(exps=='imgB_1', exps=='imgB_2'))]
    }
    cond_im_idx = {n: [image_map[c] for c in cues[idx]] for n,idx in cond_idx.items()}
    # Load files for subject
    x = torch.load("data/preprocessed_data/subject{}/nsd_imagery.pt".format(subject)).requires_grad_(False).to("cpu")
    y = torch.load("data/preprocessed_data/{}_12.pt".format(vector)).requires_grad_(False).to("cpu")
    
    # Prune down to specific experimental mode/stimuli type
    x = x[cond_idx[mode+stimtype]]

    # Average across trials
    if average:
        x = condition_average(x, cond_im_idx[mode+stimtype])
        x = x.reshape((x.shape[0], 1, x.shape[1]))
    elif nest:
        x_new = torch.zeros((12, 8, x.shape[1]))
        for i in range(12):
            x_new[i] = x[i*8: i*8 + 8]
        x = x_new


    if stimtype == "simple":
        y = y[:6]
    elif stimtype == "complex":
        y = y[6:]
    print(x.shape, y.shape)
    return x, y

# This function is used to assemble the iteration diagrams and other collages of images
#useTitle = 0 means no title at all
#useTitle = 1 means normal centered title at the top
#useTitle = 2 means title uses the captions list for a column wise title
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

# Performance Metrics
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


# -------- Stochastic Search Statistic Helper fucntions -----------

# Converts the CNN metrics for stochastic search statistics 
# into lists allowing for computation.
def column_string_to_list(df):
    
    df_new = df
    
    for index, row in tqdm(df.iterrows(), "creating lists"):
        
        df_new.at[index, 'CLIP Two-way']    = json.loads(row['CLIP Two-way'])
        df_new.at[index, 'AlexNet 2']       = json.loads(row['AlexNet 2'])
        df_new.at[index, 'AlexNet 5']       = json.loads(row['AlexNet 5'])
        df_new.at[index, 'AlexNet 7']       = json.loads(row['AlexNet 7'])
        df_new.at[index, 'Inception V3']    = json.loads(row['Inception V3'])
        df_new.at[index, 'EffNet-B']        = json.loads(row['EffNet-B'])
        df_new.at[index, 'SwAV']            = json.loads(row['SwAV'])
        
    return df_new

# Input: Dataframe containing the samples one type of image
def create_cnn_numpy_array(df):
    cnn_dict = {}
    df = df.reset_index()
    
    alexnet_2       = []
    alexnet_5       = []
    alexnet_7       = []
    clip_two_way    = []
    inception_v3    = []
    effnet_b        = []
    swav            = []
    
    for index, row in df.iterrows():
        
        alexnet_2.append(row['AlexNet 2'])
        alexnet_5.append(np.array(row['AlexNet 5']))
        alexnet_7.append(np.array(row['AlexNet 7']))
        clip_two_way.append(np.array(row['CLIP Two-way']))
        inception_v3.append(np.array(row['Inception V3']))
        effnet_b.append(np.array(row['EffNet-B']))
        swav.append(np.array(row['SwAV']))
    
    cnn_dict['AlexNet 2']      = np.concatenate([alexnet_2])
    cnn_dict['AlexNet 5']      = np.concatenate([alexnet_5])
    cnn_dict['AlexNet 7']      = np.concatenate([alexnet_7])
    cnn_dict['CLIP Two-way']   = np.concatenate([clip_two_way])
    cnn_dict['Inception V3']   = np.concatenate([inception_v3])
    cnn_dict['EffNet-B']       = np.concatenate([effnet_b])
    cnn_dict['SwAV']           = np.concatenate([swav])
    
    return cnn_dict

def pairwise_corr_all(ground_truth, predictions):
    r = np.corrcoef(ground_truth, predictions)      #cosine_similarity(ground_truth, predictions)#
    r = r[:len(ground_truth), len(ground_truth):]   # rows: groundtruth, columns: predicitons
    
    # congruent pairs are on diagonal
    congruents = np.diag(r)
    
    # for each column (predicition) we should count the number of rows (groundtruth) 
    # that the value is lower than the congruent (e.g. success).
    success = r < congruents
    success_cnt = np.sum(success, 0)
    
    # note: diagonal of 'success' is always zero so we can discard it. That's why we divide by len-1
    perf = np.mean(success_cnt) / (len(ground_truth)-1)
    p = 1 - binom.cdf(perf*len(ground_truth)*(len(ground_truth)-1), len(ground_truth)*(len(ground_truth)-1), 0.5)
    
    return perf, p

def compute_cnn_metrics(cnn_metrics_ground_truth, cnn_metrics_reconstructions):
    
    distance_fn = sp.spatial.distance.correlation
    pairwise_corrs = []
    cnn_metrics = {}
    
    for net_name, predictions_np in cnn_metrics_reconstructions.items():
        
        gt_feat = cnn_metrics_ground_truth[net_name]
        
        eval_feat = predictions_np
        num_test = predictions_np.shape[0]
        
        if net_name == 'EffNet-B' or net_name == 'SwAV':
            cnn_metrics[net_name] = np.array([distance_fn(gt_feat[i],eval_feat[i]) for i in range(num_test)]).mean()
            
        else:
            cnn_metrics[net_name] = pairwise_corr_all(gt_feat[:num_test],eval_feat[:num_test])[0]
            
    return cnn_metrics  

def remove_symlink(symlink):
    """Remove a symlink from the file system.

    Parameters
    ----------
    symlink : :obj:`str`
        Symlink to remove.
    """
    # Broken links return False on .exists(), so we need to check .islink() as well
    if not (os.path.islink(symlink) or os.path.exists(symlink)):
        return

    if os.path.isdir(symlink):
        try:
            os.rmdir(symlink)
        except NotADirectoryError:
            os.unlink(symlink)
        except PermissionError:
            raise

    else:
        os.unlink(symlink) 

def condition_average(data, cond):
    idx, idx_count = np.unique(cond, return_counts=True)
    idx_list = [cond==i for i in np.sort(idx)]
    avg_data = torch.zeros((len(idx),data.shape[1]), dtype=torch.float32)
    for i,m in enumerate(idx_list):
        avg_data[i] = torch.mean(data[m], axis=0)
    return avg_data