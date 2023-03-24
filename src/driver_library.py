import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
import torch
from torch.autograd import Variable
import numpy as np
from PIL import Image
from nsd_access import NSDAccess
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch.nn as nn
from pycocotools.coco import COCO
import h5py
from utils import *
import wandb
import copy
from tqdm import tqdm
from reconstructor import Reconstructor
from autoencoder  import AutoEncoder
from pearson import PearsonCorrCoef, pearson_corrcoef


def main():
    # benchmark_library(encModel="536_model_c_img_0.pt", vector="c_img_0", device="cuda:0", average=True, ae=True, old_norm=True)
    reconstructNImages(experiment_title="coco top 5 VD MLP Encoders", idx=[i for i in range(20)])


def predictVector_cc3m(encModel, vector, x, device="cuda:0"):
        
        if(vector == "c_img_0" or vector == "c_text_0"):
            datasize = 768
        elif(vector == "z_img_mixer"):
            datasize = 16384
        # x = x.to(device)
        prep_path = "/export/raid1/home/kneel027/nsd_local/preprocessed_data/"
        latent_path = "latent_vectors/"
        
        PeC = PearsonCorrCoef(num_outputs=22735).to(device)
        
        out = torch.zeros((x.shape[0], 5, datasize))
        average_pearson = 0
        
        for i in tqdm(range(x.shape[0]), desc="scanning library for " + vector):
            xDup = x[i].repeat(22735, 1).moveaxis(0, 1).to(device)
            scores = torch.zeros((2819141,))
            preds = torch.zeros((2819141,datasize))
            # batch_max_x = torch.zeros((620, x.shape[1]))
            # batch_max_y = torch.zeros((620, datasize))
            for batch in tqdm(range(124), desc="batching sample"):
                y = torch.load(prep_path + vector + "/cc3m_batches/" + str(batch) + ".pt")
                x_preds = torch.load(latent_path + encModel + "/cc3m_batches/" + str(batch) + ".pt")
                # print(x_preds.device)
                x_preds_t = x_preds.moveaxis(0, 1).to(device)
                preds[22735*batch:22735*batch+22735] = y.detach()
                # Pearson correlation
                scores[22735*batch:22735*batch+22735] = PeC(xDup, x_preds_t).detach()
                # Calculating the Average Pearson Across Samples
            top5_pearson = torch.topk(scores, 5)
            average_pearson += torch.mean(top5_pearson.values.detach()) 
            print(top5_pearson.indices, top5_pearson.values, scores[0:5])
            for j, index in enumerate(top5_pearson.indices):
                    out[i, j] = preds[index]
            
        torch.save(out, latent_path + encModel + "/" + vector + "_cc3m_library_preds.pt")
        print("Average Pearson Across Samples: ", (average_pearson / x.shape[0]) ) 
        return out

def predictVector_coco(encModel, vector, x, device="cuda:0"):
        if(vector == "c_img_0" or vector == "c_text_0"):
            datasize = 768
        elif(vector == "z_img_mixer"):
            datasize = 16384
        elif(vector == "c_img_vd"):
            datasize = 197376
        elif(vector == "c_text_vd"):
            datasize = 59136
        prep_path = "/export/raid1/home/kneel027/nsd_local/preprocessed_data/"
        latent_path = "latent_vectors/"
        # Save to latent vectors
        x_preds = torch.zeros((63000, 11838))
        y = torch.zeros((63000, datasize))
        subj1 = nsda.stim_descriptions[nsda.stim_descriptions['subject1'] != 0]
        nsdIds = set(subj1['nsdId'].tolist())
        
        x_preds_full = torch.load(latent_path + encModel + "/coco_brain_preds.pt", map_location=device)
        y_full = torch.load(prep_path + vector + "/vector_73k.pt")
        count = 0
        for pred in range(73000):
            if pred not in nsdIds:
                x_preds[count] = x_preds_full[pred]
                y[count] = y_full[pred]
                count+=1
        PeC = PearsonCorrCoef(num_outputs=21000).to(device)
        
        out = torch.zeros((x.shape[0], 5, datasize))
        average_pearson = 0
        
        for i in tqdm(range(x.shape[0]), desc="scanning library for " + vector):
            # print(torch.sum(torch.count_nonzero(x[i])))
            xDup = x[i].repeat(21000, 1).moveaxis(0, 1).to(device)
            scores = torch.zeros((63000,))
            preds = torch.zeros((63000,datasize))
            # batch_max_x = torch.zeros((620, x.shape[1]))
            # batch_max_y = torch.zeros((620, datasize))
            for batch in range(3):
                y_batch = y[21000*batch:21000*batch+21000]
                x_preds_batch = x_preds[21000*batch:21000*batch+21000]
                x_preds_t = x_preds_batch.moveaxis(0, 1).to(device)
                preds[21000*batch:21000*batch+21000] = y_batch.detach()
                # Pearson correlation
                scores[21000*batch:21000*batch+21000] = PeC(xDup, x_preds_t).detach()
                # print(torch.sum(torch.count_nonzero(xDup)), torch.sum(torch.count_nonzero(x_preds_t)))
                # Calculating the Average Pearson Across Samples
            top5_pearson = torch.topk(scores, 5)
            average_pearson += torch.mean(top5_pearson.values.detach()) 
            # print(top5_pearson.indices, top5_pearson.values, scores[0:5])
                
                # for j, index in enumerate(top5_pearson.indices):
                #     batch_max_x[5*batch + j] = x_preds_t[:,index].detach()
                #     batch_max_y[5*batch + j] = y[index].detach()
                    
                
            # xDupOut = x[i].repeat(620, 1).moveaxis(0, 1).to(device)
            # batch_max_x = batch_max_x.moveaxis(0, 1).to(device)
            # outPearson = outputPeC(xDupOut, batch_max_x).to("cpu")
            # top5_ind_out = torch.topk(outPearson, 5).indices
            for j, index in enumerate(top5_pearson.indices):
                    out[i, j] = preds[index]
            
        torch.save(out, latent_path + encModel + "/" + vector + "_coco_library_preds.pt")
        print("Average Pearson Across Samples: ", (average_pearson / x.shape[0]) ) 
        return out

def predictVector_Alexnet_coco(encModel, vector, x, device="cuda:0"):
    mask_path = "masks/"
    masks = {0:torch.full((11838,), False),
            1:torch.load(mask_path + "V1.pt"),
            2:torch.load(mask_path + "V2.pt"),
            3:torch.load(mask_path + "V3.pt"),
            4:torch.load(mask_path + "V4.pt"),
            5:torch.load(mask_path + "V5.pt"),
            6:torch.load(mask_path + "V6.pt"),
            7:torch.load(mask_path + "V7.pt")}
    if(vector == "c_img_0" or vector == "c_text_0"):
        datasize = 768
    elif(vector == "z_img_mixer"):
        datasize = 16384
    prep_path = "/export/raid1/home/kneel027/nsd_local/preprocessed_data/"
    latent_path = "latent_vectors/"
    # Save to latent vectors
    y = torch.zeros((63000, datasize))
    subj1 = nsda.stim_descriptions[nsda.stim_descriptions['subject1'] != 0]
    nsdIds = set(subj1['nsdId'].tolist())
    
    x_preds = torch.load(latent_path + encModel + "/coco_brain_preds.pt", map_location=device)[:, masks[1]]
    y_full = torch.load(prep_path + vector + "/vector_73k.pt")
    count = 0
    for pred in range(73000):
        if pred not in nsdIds:
            y[count] = y_full[pred]
            count+=1
    PeC = PearsonCorrCoef(num_outputs=21000).to(device)
    # outputPeC = PearsonCorrCoef(num_outputs=620).to(device)
    
    out = torch.zeros((x.shape[0], 5, datasize))
    average_pearson = 0
    
    for i in tqdm(range(x.shape[0]), desc="scanning library for " + vector):
        # print(torch.sum(torch.count_nonzero(x[i])))
        xDup = x[i, masks[1]]
        print(xDup.shape)
        xDup = xDup.repeat(21000, 1).moveaxis(0, 1).to(device)
        scores = torch.zeros((63000,))
        preds = torch.zeros((63000,datasize))
        # batch_max_x = torch.zeros((620, x.shape[1]))
        # batch_max_y = torch.zeros((620, datasize))
        for batch in range(3):
            y_batch = y[21000*batch:21000*batch+21000]
            x_preds_batch = x_preds[21000*batch:21000*batch+21000]
            x_preds_t = x_preds_batch.moveaxis(0, 1).to(device)
            preds[21000*batch:21000*batch+21000] = y_batch.detach()
            # Pearson correlation
            scores[21000*batch:21000*batch+21000] = PeC(xDup, x_preds_t).detach()
            # print(torch.sum(torch.count_nonzero(xDup)), torch.sum(torch.count_nonzero(x_preds_t)))
            # Calculating the Average Pearson Across Samples
        top5_pearson = torch.topk(scores, 5)
        average_pearson += torch.mean(top5_pearson.values.detach()) 
        # print(top5_pearson.indices, top5_pearson.values, scores[0:5])
            
            # for j, index in enumerate(top5_pearson.indices):
            #     batch_max_x[5*batch + j] = x_preds_t[:,index].detach()
            #     batch_max_y[5*batch + j] = y[index].detach()
                
            
        # xDupOut = x[i].repeat(620, 1).moveaxis(0, 1).to(device)
        # batch_max_x = batch_max_x.moveaxis(0, 1).to(device)
        # outPearson = outputPeC(xDupOut, batch_max_x).to("cpu")
        # top5_ind_out = torch.topk(outPearson, 5).indices
        for j, index in enumerate(top5_pearson.indices):
                out[i, j] = preds[index]
        
    torch.save(out, latent_path + encModel + "/" + vector + "_coco_library_preds.pt")
    print("Average Pearson Across Samples: ", (average_pearson / x.shape[0]) ) 
    return out
       
       
            
# Encode latent z (1x4x64x64) and condition c (1x77x1024) tensors into an image
# Strength parameter controls the weighting between the two tensors
def reconstructNImages(experiment_title, idx):
    
    # First URL: This is the original read-only NSD file path (The actual data)
    # Second URL: Local files that we are adding to the dataset and need to access as part of the data
    # Object for the NSDAccess package
    nsda = NSDAccess('/export/raid1/home/surly/raid4/kendrick-data/nsd', '/export/raid1/home/kneel027/nsd_local')
    os.makedirs("reconstructions/" + experiment_title + "/", exist_ok=True)
    # Retriving the ground truth image. 
    subj1 = nsda.stim_descriptions[nsda.stim_descriptions['subject1'] != 0]
    
    # Load in the data
    # Generating predicted and target vectors
    # outputs_c, targets_c = Dc.predict(hashNum=Dc.hashNum, indices=idx)
    # outputs_c_i, targets_c_i = Dc_i.predict(model=c_img_modelId)
    # outputs_c_i = [outputs_c_i[i] for i in idx]
    _, _, x_param, x_test, _, _, y_param_i, y_test_i, param_trials, test_trials = load_nsd(vector="c_img_vd", loader=False, average=True, old_norm=True)
    _, _, _, _, _, _, y_param_t, y_test_t, _, _ = load_nsd(vector="c_text_vd", loader=False, average=True, old_norm=True)
    # AE = AutoEncoder(hashNum = "540",
    #              lr=0.0000001,
    #              vector="c_img_0", #c_img_0, c_text_0, z_img_mixer
    #              encoderHash="536",
    #              log=False, 
    #              batch_size=750,
    #              device="cuda"
    #             )
    # ae_x_test = AE.predict(x_test)
    outputs_c_i = predictVector_coco(encModel="658_model_c_img_vd.pt", vector="c_img_vd", x=x_param[idx].reshape((len(idx),11838)))
    outputs_c_t = predictVector_coco(encModel="660_model_c_text_vd.pt", vector="c_text_vd", x=x_param[idx].reshape((len(idx),11838)))
    
    strength_c = 1
    strength_z = 0
    R = Reconstructor(device="cuda:0")
    for i in tqdm(idx, desc="Generating reconstructions"):
        # index = int(subj1x.loc[(subj1x['subject1_rep0'] == test_i) | (subj1x['subject1_rep1'] == test_i) | (subj1x['subject1_rep2'] == test_i)].nsdId)
        # rootdir = "/home/naxos2-raid25/kneel027/home/kneel027/nsd_local/cc3m/tensors/"
        # outputs_c_i[i] = torch.load(rootdir + "c_img_0/" + str(i) + ".pt")
        # outputs_c_t[i] = torch.load(rootdir + "c_text_0/" + str(i) + ".pt")
        # outputs_z[i] = torch.load(rootdir + "z_img_mixer/" + str(i) + ".pt")
        print(i)
        
        # outputs_z = predictVector_cc3m(encModel="543_model_z_img_mixer.pt", vector="z_img_mixer", x=x_test[i].reshape((1,11838)))
        # outputs_z = predictVector_Alexnet_coco(encModel="alexnet_encoder", vector="z_img_mixer", x=x_test[i].reshape((1,11838)), device="cuda:0")
        # outputs_c_i = outputs_c_i.reshape((5, 768))
        # c_combined = format_clip(outputs_c_i)
        # c_combined_target = format_clip(targets_c_i[i])
        # c_0 = format_clip(outputs_c_i[0])
        # c_1 = format_clip(outputs_c_i[1])
        # c_2 = format_clip(outputs_c_i[2])
        # c_3 = format_clip(outputs_c_i[3])
        # c_4 = format_clip(outputs_c_i[4])
        # outputs_c_i = outputs
        c_i_combined = torch.mean(outputs_c_i[i], dim=0)
        print(c_i_combined.shape)
        ci_0 = outputs_c_i[i,0]
        ci_1 = outputs_c_i[i,1]
        ci_2 = outputs_c_i[i,2]
        ci_3 = outputs_c_i[i,3]
        ci_4 = outputs_c_i[i,4]
        
        # outputs_c_t = outputs_c_t
        c_t_combined = torch.mean(outputs_c_t[i], dim=0)
        print(c_t_combined.shape)
        ct_0 = outputs_c_t[i,0]
        ct_1 = outputs_c_t[i,1]
        ct_2 = outputs_c_t[i,2]
        ct_3 = outputs_c_t[i,3]
        ct_4 = outputs_c_t[i,4]
    
        # Make the c reconstrution images. 
        # reconstructed_output_c = R.reconstruct(c=c_combined, strength=strength_c)
        # reconstructed_target_c = R.reconstruct(c=c_combined_target, strength=strength_c)
        target_c_i = R.reconstruct(c_i=y_param_i[i])
        output_ci = R.reconstruct(c_i=c_i_combined)
        output_ci_0 = R.reconstruct(c_i=ci_0)
        output_ci_1 = R.reconstruct(c_i=ci_1)
        output_ci_2 = R.reconstruct(c_i=ci_2)
        output_ci_3 = R.reconstruct(c_i=ci_3)
        output_ci_4 = R.reconstruct(c_i=ci_4)
        
        target_c_t = R.reconstruct(c_t=y_param_t[i])
        output_ct = R.reconstruct(c_t=c_t_combined)
        output_ct_0 = R.reconstruct(c_t=ct_0)
        output_ct_1 = R.reconstruct(c_t=ct_1)
        output_ct_2 = R.reconstruct(c_t=ct_2)
        output_ct_3 = R.reconstruct(c_t=ct_3)
        output_ct_4 = R.reconstruct(c_t=ct_4)
        
        target_c_c = R.reconstruct(c_i=y_param_i[i], c_t=y_param_t[i])
        output_cc = R.reconstruct(c_i=c_i_combined, c_t=c_t_combined)
        output_cc_0 = R.reconstruct(c_i=ci_0, c_t=ct_0)
        output_cc_1 = R.reconstruct(c_i=ci_1, c_t=ct_1)
        output_cc_2 = R.reconstruct(c_i=ci_2, c_t=ct_2)
        output_cc_3 = R.reconstruct(c_i=ci_3, c_t=ct_3)
        output_cc_4 = R.reconstruct(c_i=ci_4, c_t=ct_4)
       
        
        # returns a numpy array 
        nsdId = param_trials[i]
        ground_truth_np_array = nsda.read_images([nsdId], show=False)
        ground_truth = Image.fromarray(ground_truth_np_array[0])
        ground_truth = ground_truth.resize((512, 512), resample=Image.Resampling.LANCZOS)
        empty = Image.new('RGB', (512, 512), color='white')
        rows = 8
        columns = 3
        images = [ground_truth, empty, empty,
                  target_c_c, target_c_i, target_c_t,
                  output_cc, output_ci, output_ct,
                  output_cc_0, output_ci_0, output_ct_0,
                  output_cc_1, output_ci_1, output_ct_1,
                  output_cc_2, output_ci_2, output_ct_2,
                  output_cc_3, output_ci_3, output_ct_3,
                  output_cc_4, output_ci_4, output_ct_4]
        captions = ["ground_truth", "", "",
                  "target_c_c", "target_c_i", "target_c_t",
                  "output_cc", "output_ci", "output_ct",
                  "output_cc_0", "output_ci_0", "output_ct_0",
                  "output_cc_1", "output_ci_1", "output_ct_1",
                  "output_cc_2", "output_ci_2", "output_ct_2",
                  "output_cc_3", "output_ci_3", "output_ct_3",
                  "output_cc_4", "output_ci_4", "output_ct_4"]
        figure = tileImages(experiment_title + ": " + str(i), images, captions, rows, columns)
        
        figure.save('reconstructions/' + experiment_title + '/' + str(i) + '.png')
        
    
def benchmark_library(encModel, vector, device="cuda:0", average=True, ae=True, old_norm=False):
    print(encModel)
    _, _, _, x_test, _, _, _, target, _, test_trials = load_nsd(vector=vector, loader=False, average=average, old_norm=old_norm)
    # if(not os.path.isfile("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/latent_vectors/" + encModel + "/library_preds_nsd_test.pt")):
    if(ae):
        AE = AutoEncoder(hashNum = "577",
                 lr=0.0000001,
                 vector="c_img_0", #c_img_0, c_text_0, z_img_mixer
                 encoderHash="536",
                 log=False, 
                 batch_size=750,
                 device=device
                )
        x_test = AE.predict(x_test).to("cpu")
    out = predictVector_coco(encModel=encModel, vector=vector, x=x_test, device=device)[:,0]
    # out = predictVector_cc3m(encModel=encModel, vector=vector, x=x_test, device=device)[:,0]
    # torch.save(out, "/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/latent_vectors/" + encModel + "/library_preds_nsd_test_avg.pt")
        
    # else:
        # out = torch.load("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/latent_vectors/" + encModel + "/library_preds_nsd_test.pt", map_location=device)
    
    criterion = nn.MSELoss()
    
    PeC = PearsonCorrCoef(num_outputs=x_test.shape[0]).to(device)
    target = target.to(device)
    out = out.to(device)

    loss = criterion(out, target)
    out = out.moveaxis(0,1).to(device)
    target = target.moveaxis(0,1).to(device)
    pearson_loss = torch.mean(PeC(out, target).detach())
    
    out = out.detach().cpu()
    target = target.detach().cpu()
    PeC = PearsonCorrCoef().to("cpu")
    r = []
    for p in range(out.shape[1]):
        
        # Correlation across voxels for a sample (Taking a column)
        r.append(PeC(out[:,p], target[:,p]))
    r = np.array(r)
    
    print("Vector Correlation: ", float(pearson_loss))
    print("Mean Pearson: ", np.mean(r))
    print("Loss: ", float(loss))
    plt.hist(r, bins=40, log=True)
    plt.savefig("charts/" + encModel + "_pearson_histogram_library_decoder.png")

if __name__ == "__main__":
    main()
