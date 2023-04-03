import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
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
import random
import copy
from tqdm import tqdm
from reconstructor import Reconstructor
from autoencoder  import AutoEncoder
from pearson import PearsonCorrCoef, pearson_corrcoef


def main():
    # benchmark_library(encModel="536_model_c_img_0.pt", vector="c_img_0", device="cuda:0", average=True, ae=True, old_norm=True)
    # reconstructNImages(experiment_title="coco top 5 VD 722 AE Clip Extract", idx=[i for i in range(0, 20)], mask=[], ae=True, test=False, average=True)
    reconstruct_test_samples("SCS VD PCA LR 10:250:5 0.6 Exp3 AE NA", idx=[], test=True, average=True, ae=True)

def predictVector_cc3m(encModel, vector, x, mask=[], device="cuda:0"):
        
        if(vector == "c_img_0" or vector == "c_text_0"):
            datasize = 768
        elif(vector == "z_img_mixer"):
            datasize = 16384
        elif(vector == "images"):
            datasize = 541875
        # x = x.to(device)
        prep_path = "/export/raid1/home/kneel027/nsd_local/preprocessed_data/"
        latent_path = "latent_vectors/"
        
        PeC = PearsonCorrCoef(num_outputs=22735).to(device)
        
        out = torch.zeros((x.shape[0], 5, datasize))
        average_pearson = 0
        
        for i in tqdm(range(x.shape[0]), desc="scanning library for " + vector):
            xDup = x[i].repeat(22735, 1).moveaxis(0, 1).to(device)
            scores = torch.zeros((2819141,))
            # preds = torch.zeros((2819141,datasize))
            # batch_max_x = torch.zeros((620, x.shape[1]))
            # batch_max_y = torch.zeros((620, datasize))
            for batch in tqdm(range(124), desc="batching sample"):
                # y = torch.load(prep_path + vector + "/cc3m_batches/" + str(batch) + ".pt")
                x_preds = torch.load(latent_path + encModel + "/cc3m_batches/" + str(batch) + ".pt")
                # print(x_preds.device)
                x_preds_t = x_preds.moveaxis(0, 1).to(device)
                # preds[22735*batch:22735*batch+22735] = torch.load(prep_path + vector + "/cc3m_batches/" + str(batch) + ".pt")
                # Pearson correlation
                scores[22735*batch:22735*batch+22735] = PeC(xDup, x_preds_t).detach()
                # Calculating the Average Pearson Across Samples
            top5_pearson = torch.topk(scores, 5)
            average_pearson += torch.mean(top5_pearson.values.detach()) 
            print(top5_pearson.indices, top5_pearson.values, scores[0:5])
            for j, index in enumerate(top5_pearson.indices):
                batch = int(index // 22735)
                sample = int(index % 22735)
                batch_preds = torch.load(prep_path + vector + "/cc3m_batches/" + str(batch) + ".pt")
                out[i, j] = batch_preds[sample]
            
        torch.save(out, latent_path + encModel + "/" + vector + "_cc3m_library_preds.pt")
        print("Average Pearson Across Samples: ", (average_pearson / x.shape[0]) ) 
        return out



def predictVector_coco(encModel, vector, x, mask=[], device="cuda:0"):
    mask_path = "masks/"
    masks = {0:torch.full((11838,), False),
            1:torch.load(mask_path + "V1.pt"),
            2:torch.load(mask_path + "V2.pt"),
            3:torch.load(mask_path + "V3.pt"),
            4:torch.load(mask_path + "V4.pt"),
            5:torch.load(mask_path + "V5.pt"),
            6:torch.load(mask_path + "V6.pt"),
            7:torch.load(mask_path + "V7.pt")}
    prep_path = "/export/raid1/home/kneel027/nsd_local/preprocessed_data/"
    latent_path = "latent_vectors/"
    
    if(vector == "c_img_0" or vector == "c_text_0"):
        datasize = 768
    elif(vector == "z_img_mixer"):
        datasize = 16384
    elif(vector == "images"):
        datasize = 541875
    if isinstance(encModel, list):
        x_preds = []
        for model in encModel:
            modelPreds = torch.load(latent_path + model + "/coco_brain_preds.pt", map_location=device)
            # modelPreds = prune_predictions(modelPreds, model)
            x_preds.append(modelPreds)
    else:
        x_preds = torch.load(latent_path + encModel + "/coco_brain_preds.pt", map_location=device)
        # x_preds = prune_predictions(x_preds)
    y_full = torch.load(prep_path + vector + "/vector_73k.pt").reshape(73000, datasize)
    y = prune_predictions(y_full)
    # print(x_preds.shape)
    if(len(mask)>0):
        beta_mask = masks[0]
        if(mask[0] == -1):
            maskList = [1,2,3,4,5,6,7]
            for i in maskList:
                beta_mask = torch.logical_or(beta_mask, masks[i])
            beta_mask = ~beta_mask
        else:   
            for i in mask:
                beta_mask = torch.logical_or(beta_mask, masks[i])
        x = x[:, beta_mask]
        x_preds = x_preds[:, beta_mask]

    out = torch.zeros((x.shape[0], 5, datasize))
    
    PeC = PearsonCorrCoef(num_outputs=21000).to(device)
    average_pearson = 0
    
    for i in tqdm(range(x.shape[0]), desc="scanning library for " + vector):
        # print(torch.sum(torch.count_nonzero(x[i])))
        xDup = x[i]
        xDup = xDup.repeat(21000, 1).moveaxis(0, 1).to(device)
        scores = torch.zeros((63000,))
        for batch in range(3):
            if isinstance(x_preds, list):
                x_preds_t = []
                for modelPreds in x_preds:
                    x_preds_batch = modelPreds[21000*batch:21000*batch+21000]
                    x_preds_t = x_preds_batch.moveaxis(0, 1).to(device)
                    modelScore = PeC(xDup, x_preds_t).cpu().detach()
                    # print(scores.device, modelScore.device)
                    scores[21000*batch:21000*batch+21000] += modelScore.detach()
                scores /= len(x_preds)
            else:
                x_preds_batch = x_preds[21000*batch:21000*batch+21000]
                x_preds_t = x_preds_batch.moveaxis(0, 1).to(device)
            # Pearson correlation
                scores[21000*batch:21000*batch+21000] = PeC(xDup, x_preds_t).detach()
            # print(torch.sum(torch.count_nonzero(xDup)), torch.sum(torch.count_nonzero(x_preds_t)))
            
            # Calculating the Average Pearson Across Samples
        top5_pearson = torch.topk(scores, 5)
        average_pearson += torch.mean(top5_pearson.values.detach()) 
        for j, index in enumerate(top5_pearson.indices):
                out[i, j] = y[index]
        
    # torch.save(out, latent_path + encModel + "/" + vector + "_coco_library_preds.pt")
    print("Average Pearson Across Samples: ", (average_pearson / x.shape[0]) ) 
    return out
       
       
            
# Encode latent z (1x4x64x64) and condition c (1x77x1024) tensors into an image
# Strength parameter controls the weighting between the two tensors
def reconstructNImages(experiment_title, idx, mask=[], ae=False, test=True, average=True):
    
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
    # AE = AutoEncoder(hashNum = "582",
    #                 lr=0.0000001,
    #                 vector="alexnet_encoder_sub1", #c_img_0, c_text_0, z_img_mixer
    #                 encoderHash="579",
    #                 log=False, 
    #                 batch_size=750,
    #                 device="cuda:0"
    #                 )
    AE1 = AutoEncoder(hashNum = "724",
                    lr=0.0000001,
                    vector="c_img_vd", #c_img_0, c_text_0, z_img_mixer
                    encoderHash="722",
                    log=False, 
                    batch_size=750,
                    device="cuda:0"
                    )
    AE2 = AutoEncoder(hashNum = "727",
                    lr=0.0000001,
                    vector="c_text_vd", #c_img_0, c_text_0, z_img_mixer
                    encoderHash="660",
                    log=False, 
                    batch_size=750,
                    device="cuda:0"
                    )
         # Load data and targets
    if test:
        _, _, _, x, _, _, _, targets_c_i, _, trials = load_nsd(vector="c_img_vd", loader=False, average=False, nest=True)
        _, _, _, _, _, _, _, targets_c_t, _, _ = load_nsd(vector="c_text_vd", loader=False, average=False, nest=True)
    else:
        _, _, x, _, _, _, targets_c_i, _, trials, _ = load_nsd(vector="c_img_vd", loader=False, average=False, nest=True)
        _, _, _, _, _, _, targets_c_t, _, _, _ = load_nsd(vector="c_text_vd", loader=False, average=False, nest=True)
    
    x_pruned_ae = torch.zeros((len(idx), 11838))
    x_pruned = torch.zeros((len(idx), 11838))
    for i, index in enumerate(tqdm(idx, desc="Pruning and autoencoding samples")):# and averaging"):
        tqdm.write(str(i) + " " + str(index))
        if(average):
            if(ae):
                x_pruned_ae[i] = torch.mean(AE.predict(x[index]),dim=0)
            x_pruned[i] = torch.mean(x[index], dim=0)
        else:
            x_pruned[i] = x[index, random.randrange(0,3)]
            if(ae):
                x_pruned_ae[i] = AE.predict(x_pruned[i])
    if ae:
        x = x_pruned_ae
    else:
        x = x_pruned
    
    # output_images = predictVector_coco(encModel="alexnet_encoder", vector="images", x=x)
    # output_images = predictVector_cc3m(encModel="alexnet_encoder", vector="images", x=x)
    output_images = predictVector_coco(encModel=["722_model_c_img_vd.pt", "660_model_c_text_vd"], vector="images", x=x, mask=mask)
    # output_images = predictVector_coco(encModel="722_model_c_img_vd.pt", vector="images", x=x)
    
    R = Reconstructor(device="cuda:0")
    for i, val in enumerate(tqdm(idx, desc="Generating reconstructions")):
        
        print(i, val)
        c_i_mean = torch.mean(output_images[i].to(torch.float32), dim=0)
        i_combined = process_image(c_i_mean.to(torch.uint8))
        i_0 = process_image(output_images[i,0])
        i_1 = process_image(output_images[i,1])
        i_2 = process_image(output_images[i,2])
        i_3 = process_image(output_images[i,3])
        i_4 = process_image(output_images[i,4])
        
        
        ci_0 = R.encode_image(i_0)
        ci_1 = R.encode_image(i_1)
        ci_2 = R.encode_image(i_2)
        ci_3 = R.encode_image(i_3)
        ci_4 = R.encode_image(i_4)
        c_i_combined = torch.mean(torch.stack([ci_0, ci_1, ci_2, ci_3, ci_4]), dim=0)
        
        # c_i_combined = torch.mean(outputs_c_i[i], dim=0)
        # ci_0 = outputs_c_i[i,0]
        # ci_1 = outputs_c_i[i,1]
        # ci_2 = outputs_c_i[i,2]
        # ci_3 = outputs_c_i[i,3]
        # ci_4 = outputs_c_i[i,4]
        
        # outputs_c_t = outputs_c_t
        # c_t_combined = torch.mean(outputs_c_t[i], dim=0)
        # ct_0 = outputs_c_t[i,0]
        # ct_1 = outputs_c_t[i,1]
        # ct_2 = outputs_c_t[i,2]
        # ct_3 = outputs_c_t[i,3]
        # ct_4 = outputs_c_t[i,4]
    
        # Make the c reconstrution images. 
        # reconstructed_output_c = R.reconstruct(c=c_combined, strength=strength_c)
        # reconstructed_target_c = R.reconstruct(c=c_combined_target, strength=strength_c)
        target_c_i = R.reconstruct(c_i=targets_c_i[val])
        output_ci = R.reconstruct(c_i=c_i_combined)
        output_ci_0 = R.reconstruct(c_i=ci_0)
        output_ci_1 = R.reconstruct(c_i=ci_1)
        output_ci_2 = R.reconstruct(c_i=ci_2)
        output_ci_3 = R.reconstruct(c_i=ci_3)
        output_ci_4 = R.reconstruct(c_i=ci_4)
        
        # target_c_t = R.reconstruct(c_t=targets_c_t[val])
        # output_ct = R.reconstruct(c_t=c_t_combined)
        # output_ct_0 = R.reconstruct(c_t=ct_0)
        # output_ct_1 = R.reconstruct(c_t=ct_1)
        # output_ct_2 = R.reconstruct(c_t=ct_2)
        # output_ct_3 = R.reconstruct(c_t=ct_3)
        # output_ct_4 = R.reconstruct(c_t=ct_4)
        
        # target_c_c = R.reconstruct(c_i=targets_c_i[val], c_t=targets_c_t[val])
        # output_cc = R.reconstruct(c_i=c_i_combined, c_t=c_t_combined)
        # output_cc_0 = R.reconstruct(c_i=ci_0, c_t=ct_0)
        # output_cc_1 = R.reconstruct(c_i=ci_1, c_t=ct_1)
        # output_cc_2 = R.reconstruct(c_i=ci_2, c_t=ct_2)
        # output_cc_3 = R.reconstruct(c_i=ci_3, c_t=ct_3)
        # output_cc_4 = R.reconstruct(c_i=ci_4, c_t=ct_4)
       
        
        # returns a numpy array 
        nsdId = trials[val]
        ground_truth_np_array = nsda.read_images([nsdId], show=False)
        ground_truth = Image.fromarray(ground_truth_np_array[0])
        ground_truth = ground_truth.resize((512, 512), resample=Image.Resampling.LANCZOS)
        empty = Image.new('RGB', (512, 512), color='white')
        rows = 7
        columns = 2
        images = [ground_truth, target_c_i, 
                  i_combined,   output_ci,
                  i_0,          output_ci_0,
                  i_1,          output_ci_1,
                  i_2,          output_ci_2,
                  i_3,          output_ci_3,
                  i_4,          output_ci_4,]
        captions = ["ground_truth","target_c_i", 
                  "output_i",     "output_ci", 
                  "output_i_0",   "output_ci_0",
                  "output_i_1",   "output_ci_1",
                  "output_i_2",   "output_ci_2", 
                  "output_i_3",   "output_ci_3", 
                  "output_i_4",   "output_ci_4",]
        # images = [ground_truth, target_c_c, target_c_i, target_c_t,
        #           i_combined,   output_cc,  output_ci, output_ct,
        #           i_0,          output_cc_0, output_ci_0, output_ct_0,
        #           i_1,          output_cc_1, output_ci_1, output_ct_1,
        #           i_2,          output_cc_2, output_ci_2, output_ct_2,
        #           i_3,          output_cc_3, output_ci_3, output_ct_3,
        #           i_4,          output_cc_4, output_ci_4, output_ct_4]
        # captions = ["ground_truth","target_c_c", "target_c_i", "target_c_t",
        #           "output_i", "output_cc", "output_ci", "output_ct",
        #           "output_i_0", "output_cc_0", "output_ci_0", "output_ct_0",
        #           "output_i_1", "output_cc_1", "output_ci_1", "output_ct_1",
        #           "output_i_2", "output_cc_2", "output_ci_2", "output_ct_2",
        #           "output_i_3", "output_cc_3", "output_ci_3", "output_ct_3",
        #           "output_i_4", "output_cc_4", "output_ci_4", "output_ct_4"]
        figure = tileImages(experiment_title + ": " + str(val), images, captions, rows, columns)
        
        figure.save('reconstructions/' + experiment_title + '/' + str(val) + '.png')
        
    
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



def reconstruct_test_samples(experiment_title, idx=[], test=False, average=True, ae=True):
    if(len(idx) == 0):
        for file in os.listdir("reconstructions/" + experiment_title + "/"):
            if file.endswith(".png") and file not in ["Search Iterations.png", "Results.png"]:
                idx.append(int(file[:-4]))
        idx = sorted(idx)

    AE = AutoEncoder(hashNum = "582",
                    lr=0.0000001,
                    vector="alexnet_encoder_sub1", 
                    encoderHash="579",
                    log=False, 
                    batch_size=750,
                    device="cuda:0"
                    )
    if test:
        _, _, _, x, _, _, _, targets_c_i, _, trials = load_nsd(vector="c_img_vd", loader=False, average=False, nest=True)
        _, _, _, _, _, _, _, targets_c_t, _, _ = load_nsd(vector="c_text_vd", loader=False, average=False, nest=True)
    else:
        _, _, x, _, _, _, targets_c_i, _, trials, _ = load_nsd(vector="c_img_vd", loader=False, average=False, nest=True)
        _, _, _, _, _, _, targets_c_t, _, _, _ = load_nsd(vector="c_text_vd", loader=False, average=False, nest=True)
    
    x_pruned_ae = torch.zeros((len(idx), 11838))
    x_pruned = torch.zeros((len(idx), 11838))
    for i, index in enumerate(tqdm(idx, desc="Pruning and autoencoding samples")):# and averaging"):
        tqdm.write(str(i) + " " + str(index))
        if(average):
            if(ae):
                x_pruned_ae[i] = torch.mean(AE.predict(x[index]),dim=0)
            x_pruned[i] = torch.mean(x[index], dim=0)
        else:
            x_pruned[i] = x[index, random.randrange(0,3)]
            if(ae):
                x_pruned_ae[i] = AE.predict(x_pruned[i])
    if ae:
        x = x_pruned_ae
    else:
        x = x_pruned

    output_images = predictVector_coco(encModel="alexnet_encoder", vector="images", x=x)
    for i, image in enumerate(output_images):
        top_choice = image[0].reshape(425, 425, 3)
        top_choice = top_choice.detach().cpu().numpy().astype(np.uint8)
        pil_image = Image.fromarray(top_choice).resize((512, 512), resample=Image.Resampling.LANCZOS)
        pil_image.save("reconstructions/" + experiment_title + "/" + str(idx[i]) + "/Library Reconstruction.png")

if __name__ == "__main__":
    main()