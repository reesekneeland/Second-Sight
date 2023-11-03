import os
import time
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
from scipy.stats import pearsonr,binom,linregress
from scipy.spatial import distance
import pickle
from itertools import product


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
    #Flag to make compatible with existing SS architectures that expect multiple trials
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
    print(x.shape)
    y = torch.load("data/preprocessed_data/{}_12.pt".format(vector)).requires_grad_(False).to("cpu")
    
    # Prune down to specific experimental mode/stimuli type
    x = x[cond_idx[mode+stimtype]]
    averaged_x, sample_count = condition_average(x, cond_im_idx[mode+stimtype])
    trial_count = int(x.shape[0]/sample_count)
    # Average across trials
    if average:
        x =  averaged_x
        x = x.reshape((x.shape[0], 1, x.shape[1]))
    elif nest:
        x_new = torch.zeros((sample_count, trial_count, x.shape[1]))
        for i in range(sample_count):
            x_new[i] = x[i*trial_count: i*trial_count + trial_count]
        x = x_new


    if stimtype == "simple":
        y = y[:sample_count]
    elif stimtype == "complex":
        y = y[sample_count:]
    print(x.shape, y.shape)
    return x, y

# useTitle = 3 means title uses the captions list for a row wise title
# This function is used to assemble the iteration diagrams and other collages of images
# useTitle = 0 means no title at all
# useTitle = 1 means normal centered title at the top
# useTitle = 2 means title uses the captions list for a column wise title
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
                    textLabeler.text((i+16, j+imH+8), captions[j], font=font, fill='black')
                count+=1
    if redCol:
        sub_col = canvas.crop((cStep + buffer, 0, bigW, height))
        buf = Image.new('RGB', (buffer, height), color='white')
        canvas.paste(sub_col, (cStep + buffer + buffer, 0))
        canvas.paste(buf, (cStep + buffer, 0))
    return canvas

def format_tiled_figure(images, captions, rows, cols, red_line_index=None, buffer=10, mode=0, title=None):
    """
    Assembles a tiled figure of images with optional captions and a red background behind a specified column or row.

    :param images: List of PIL Image objects, ordered row-wise.
    :param captions: List of captions, length and usage depends on mode.
    :param rows: Number of rows in the image grid.
    :param cols: Number of columns in the image grid.
    :param red_line_index: Index of the row or column to highlight with a red background (0-indexed).
    :param buffer: Buffer value in pixels for space between images.
    :param mode: Mode of the figure assembly.
    :param title: Title of the figure, used in mode 1 and mode 3.
    :return: PIL Image object of the assembled figure.
    """
    
    # Find the smallest width and height among all images
    min_width, min_height = min(img.size for img in images)

    # Resize all images to the smallest dimensions
    images = [img.resize((min_width, min_height), Image.ANTIALIAS) for img in images]

    # Font setup
    font_size = 60  # Base font size for readability
    row_caption_font_size = font_size  
    title_font_size = int(1.3 * font_size) 
    title_font = ImageFont.truetype("arial.ttf", title_font_size)
    row_caption_font = ImageFont.truetype("arial.ttf", row_caption_font_size)

    # Calculate dimensions for the entire canvas
    caption_height = row_caption_font_size if mode in [0, 1] else 0
    title_height = int(title_font_size * 1.5) if mode in [1, 3] else 0  # Adjusted to include mode 3
    row_title_width = int(row_caption_font_size * 1.5) if mode == 3 else 0
    extra_buffer_w = buffer if (red_line_index is not None and mode in [0, 1, 2]) else 0
    extra_buffer_h = buffer if (red_line_index is not None and mode == 3) else 0

    # Calculate the total canvas width and height
    total_width = cols * (min_width + buffer) + row_title_width + buffer + extra_buffer_w
    total_height = rows * (min_height + buffer) + title_height + rows * caption_height + buffer + extra_buffer_h

    # Create a new image with a white background
    canvas = Image.new('RGB', (total_width, total_height), color='white')

    # Prepare the drawing context
    draw = ImageDraw.Draw(canvas)

    # Draw the title for modes 1 and 3
    if mode in [1, 3] and title is not None:  # Adjusted to include mode 3
        text_width, text_height = draw.textsize(title, font=title_font)
        draw.text(((total_width - text_width) // 2, (title_height - text_height) // 2), title, font=title_font, fill='black')

    # Draw red background before placing images if a red line index is specified
    if red_line_index is not None:
        if mode in [0, 1, 2]:  # Red column
            red_x = row_title_width + red_line_index * (min_width + buffer)
            red_y = title_height
            red_width = min_width + buffer * 2
            red_height = total_height - title_height
            canvas.paste(Image.new('RGB', (red_width, red_height), color='red'), (red_x, red_y))
        elif mode == 3:  # Red row
            red_x = row_title_width
            red_y = title_height + red_line_index * (min_height + buffer)
            red_width = total_width - row_title_width
            red_height = min_height + buffer * 2
            canvas.paste(Image.new('RGB', (red_width, red_height), color='red'), (red_x, red_y))

    # Insert images into the canvas
    for row in range(rows):
        for col in range(cols):
            idx = row * cols + col
            if idx >= len(images):
                continue

            img = images[idx]
            x = col * (min_width + buffer) + row_title_width + buffer
            y = row * (min_height + buffer) + title_height + buffer

            # Adjust the x position if there is a red column
            if mode in [0, 1, 2] and red_line_index is not None and col > red_line_index:
                x += extra_buffer_w

            # Adjust the y position if there is a red row
            if mode == 3 and red_line_index is not None and row > red_line_index:
                y += extra_buffer_h

            # Paste the image
            canvas.paste(img, (x, y))
    # Draw the vertical text for row titles if mode is 3
    if mode == 3:
        for row, caption in enumerate(captions):
            # Calculate the caption size using the default font
            width, height = row_caption_font.getsize(caption)

            text_image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(text_image)
            draw.text((0, 0), text=caption, font=row_caption_font, fill='black')

            # Rotate the text image to be vertical
            text_image = text_image.rotate(90, expand=1)

            # Calculate the y position for the vertical text
            y = row * (min_height + buffer) + (min_width - width )//2 + title_height
            if row > 0:
                y += buffer

            # Calculate the x position, accounting for the increased text size
            x = 0

            # Paste the rotated text image onto the canvas
            canvas.paste(text_image, (x, y), text_image)

    # Draw captions for each image for modes 0 and 1
    if mode in [0, 1]:
        for idx, caption in enumerate(captions):
            col = idx % cols
            row = idx // cols
            text_width, text_height = draw.textsize(caption, font=row_caption_font)
            x = col * (min_width + buffer) + row_title_width + buffer + (min_width - text_width) // 2
            y = (row + 1) * (min_height + buffer) + title_height - text_height // 2
            draw.text((x, y), caption, font=row_caption_font, fill='black')

    # Draw column titles if mode is 2
    if mode == 2:
        for col, caption in enumerate(captions):
            text_width, text_height = draw.textsize(caption, font=row_caption_font)
            x = col * (min_width + buffer) + row_title_width + buffer + (min_width - text_width) // 2
            y = buffer
            draw.text((x, y), caption, font=row_caption_font, fill='black')

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
    return avg_data, len(idx_count)

# Preprocess beta, remove empty trials
def prepare_betas(beta):
    beta_list = []
    for i in range(beta.shape[0]):
        if(torch.count_nonzero(beta[i]) > 0):
            beta_list.append(beta[i])
    return torch.stack(beta_list)

def bootstrap_variance(data, n_iterations=1000):
    """
    Estimate the variance of an array of 3 values using bootstrapping.

    Parameters:
    - data: List or array-like object containing the data to get the variance of.
    - n_iterations: Number of bootstrap samples to generate.

    Returns:
    - variance_distribution: List of variances from each bootstrap sample.
    """
    if(isinstance(data, torch.Tensor)):
        data = data.cpu().numpy()
        
    rng = np.random.default_rng()
    variance_distribution = []
    
    if len(data.shape) == 1:
        n = len(data)
        for _ in range(n_iterations):
            sample = rng.choice(data, size=n, replace=True)
            # print(sample)
            variance_distribution.append(np.var(sample))

    elif len(data.shape) == 2:
        n_rows = data.shape[0]
        for _ in range(n_iterations):
            sampled_rows = data[rng.choice(n_rows, n_rows, replace=True)]
            variance_distribution.append(np.var(sampled_rows, axis=0).mean())

    else:
        raise ValueError("Input data must be either one-dimensional or two-dimensional.")

    
    return float(np.mean(variance_distribution))



def create_cnn_numpy_array_shared1000(method, subject, low=False):
    if low:
        feature_path = f"output/second_sight_paper/dataframes/{method}/subject{subject}/features_low/"
        cnn_dict_path = f"output/second_sight_paper/dataframes/cnn_dict_{method}_subject{subject}_low.pkl"
    else:
        feature_path = f"output/second_sight_paper/dataframes/{method}/subject{subject}/features/"
        cnn_dict_path = f"output/second_sight_paper/dataframes/cnn_dict_{method}_subject{subject}.pkl"
    cnn_dict = {}
    
    net_list = [
        'Inception V3',
        'CLIP Two-way',
        'AlexNet 1',
        'AlexNet 2',
        'AlexNet 3',
        'AlexNet 4',
        'AlexNet 5',
        'AlexNet 6',
        'AlexNet 7']
    if False:
    # if os.path.isfile(cnn_dict_path):
        print(f'Now Loading cnn_dict for... {method}, subject{subject}, low={low}')
        with open(cnn_dict_path, 'rb') as f:
            cnn_dict = pickle.load(f)
    else:
        for index in range(982):
            if not cnn_dict:
                for net_name in net_list:
                    cnn_dict[net_name] = [np.load(f"{feature_path}{index}/{net_name}.npy")]
            else:
                for net_name in net_list:
                    cnn_dict[net_name].append(np.load(f"{feature_path}{index}/{net_name}.npy"))
        for key, value in cnn_dict.items():
            stacked_array = np.stack(value)
            cnn_dict[key] = stacked_array
        with open(cnn_dict_path,"wb") as f:
            pickle.dump(cnn_dict,f)
    return cnn_dict

def compute_similarity_percentage(ground_truth_features, reconstructed_features, background_features):
    percentages = np.zeros(ground_truth_features.shape[0])  # This will hold the percentage for each feature

    # Compute the base similarity scores between ground truth and reconstructed features
    base_similarities = 1 - distance.cdist(ground_truth_features, reconstructed_features, 'cosine').diagonal()

    # Iterate over each reconstructed feature and corresponding base similarity score
    for i, (feature_reconstructed, base_similarity) in enumerate(zip(reconstructed_features, base_similarities)):
        
        # Compute similarity between the reconstructed feature and background dataset
        similarities = 1 - distance.cdist([feature_reconstructed], background_features, 'cosine')[0]

        # Count how many background samples have a lower similarity score than the base similarity
        count_lower_similarity = np.sum(similarities < base_similarity)

        # Compute the percentage
        percentages[i] = (count_lower_similarity / len(background_features))

    return percentages

# 'percentages' contains the percentage of background samples for each feature in array2
# that have a lower similarity score compared to the base similarity score with array1.

def compute_cnn_metrics_shared1000(cnn_metrics_ground_truth, cnn_metrics_reconstructions, method, subject, low=False):
    cnn_metrics = {}
    net_list = [
        'Inception V3',
        'CLIP Two-way',
        'AlexNet 1',
        'AlexNet 2',
        'AlexNet 3',
        'AlexNet 4',
        'AlexNet 5',
        'AlexNet 6',
        'AlexNet 7']
    
    background_feat_dict = create_cnn_numpy_array_shared1000(method, subject, low)
    
    for net_name in net_list:
        gt_feat = cnn_metrics_ground_truth[net_name]
        eval_feat = cnn_metrics_reconstructions[net_name]
        num_test = eval_feat.shape[0]
        background_feat = background_feat_dict[net_name]
        percent_success = compute_similarity_percentage(gt_feat, eval_feat, background_feat)
        cnn_metrics[net_name] = percent_success
    return cnn_metrics

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
    
    return perf, p, success_cnt / (len(ground_truth)-1)

def compute_cnn_metrics(cnn_metrics_ground_truth, cnn_metrics_reconstructions):
    distance_fn = sp.spatial.distance.correlation
    pairwise_corrs = []
    cnn_metrics = {}
    # print(cnn_metrics_reconstructions)
    for net_name, predictions_np in cnn_metrics_reconstructions.items():
        
        gt_feat = cnn_metrics_ground_truth[net_name]
        
        eval_feat = predictions_np
        num_test = predictions_np.shape[0]
        # print(net_name, predictions_np.shape)
        if net_name == 'EffNet-B' or net_name == 'SwAV':
            cnn_metrics[net_name] = np.array([distance_fn(gt_feat[i],eval_feat[i]) for i in range(num_test)])
            
        else:
            _, _, success_cnt = pairwise_corr_all(gt_feat[:num_test],eval_feat[:num_test])
            cnn_metrics[net_name] = success_cnt
    return cnn_metrics

def create_cnn_numpy_array(df, feature_path):
    cnn_dict = {}
    net_list = [
        'Inception V3',
        'CLIP Two-way',
        'AlexNet 1',
        'AlexNet 2',
        'AlexNet 3',
        'AlexNet 4',
        'AlexNet 5',
        'AlexNet 6',
        'AlexNet 7',
        'EffNet-B',
        'SwAV']
    
    for index, _ in df.iterrows():
        if not cnn_dict:
            for net_name in net_list:
                cnn_dict[net_name] = [np.load(f"{feature_path}{index}/{net_name}.npy")]
        else:
            for net_name in net_list:
                cnn_dict[net_name].append(np.load(f"{feature_path}{index}/{net_name}.npy"))
    for key, value in cnn_dict.items():
        stacked_array = np.stack(value)
        cnn_dict[key] = stacked_array
    return cnn_dict

def get_iter_variance(iter_path, masks):
    var_dict_path = os.path.join(iter_path, 'variance.pkl')
    
    # Check if the variance dictionary file exists
    # if os.path.exists(var_dict_path):
    #     # Load and return the existing variance dictionary
    #     with open(var_dict_path, 'rb') as f:
    #         var_dict = pickle.load(f)
    #     return var_dict

    beta_primes = []
    var_dict = {}
    start = time.time()
    if "iter_0" in iter_path:
        for file in os.listdir(iter_path + "beta_primes/"):
            beta_primes.append(torch.load(f"{iter_path}beta_primes/{file}"))
        beta_primes = torch.stack(beta_primes)
        for mask_name, mask in masks.items():
            var_dict[mask_name] = bootstrap_variance(beta_primes[:, mask], n_iterations=100)
    else:
        for mask_name in masks:
            var_dict[mask_name] = []
        for folder in os.listdir(iter_path):
            if not("best_batch" in folder) and os.path.isdir(iter_path + folder):
                beta_primes = []
                for file in os.listdir(iter_path + folder + "/beta_primes/"):
                    beta_primes.append(torch.load(f"{iter_path}{folder}/beta_primes/{file}"))
                beta_primes = torch.stack(beta_primes)
                for mask_name, mask in masks.items():
                    var_dict[mask_name].append(bootstrap_variance(beta_primes[:, mask], n_iterations=100))
        for mask_name in masks:
            var_dict[mask_name] = np.mean(var_dict[mask_name])
    
    # Save the variance dictionary to a file
    with open(var_dict_path, 'wb') as f:
        pickle.dump(var_dict, f)
    # print(f"Variance calculation took {time.time() - start} seconds")
    return var_dict