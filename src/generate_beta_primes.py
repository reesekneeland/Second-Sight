import os
import numpy as np
from scipy.special import erf
import nibabel as nib
import pandas as pd
import torch
from tqdm import tqdm
import argparse

from clip_encoder       import CLIPEncoder
from gnet8_encoder      import GNet8_Encoder
from stochastic_search  import StochasticSearch

# Create the parser and add arguments
parser = argparse.ArgumentParser()

parser.add_argument('-s',
                    '--subjects', 
                    help="list of subjects to run the algorithm on, if not specified, will run on all subjects",
                    type=list,
                    default=[5, 7])

parser.add_argument('-n',
                    '--num_samples', 
                    help="number of library images generated at every interation of the algorithm.",
                    type=int,
                    default=100)

parser.add_argument('-i',
                    '--iterations', 
                    help="number of interations for the search algorithm.",
                    type=int,
                    default=6)

parser.add_argument('-b',
                    '--branches', 
                    help="number of additional suboptimal paths to explore during the search.",
                    type=int,
                    default=4)

parser.add_argument('-d',
                    '--device', 
                    help="cuda device to run predicts on.",
                    type=str,
                    default="cuda:0")


# Parse and print the results
args = parser.parse_args()


for subject in args.subjects:
    
    print("Initializing CLIP Encoder")
    CLIP = CLIPEncoder(inference=True,
                        subject=subject,
                        device=args.device)

    print("Initializing GNET Encoder")
    GNET = GNet8_Encoder(device=args.device,
                        subject=subject)

    print("Initializing SCS Encoder")
    SCS  = StochasticSearch(modelParams=["gnetEncoder", "clipEncoder"],
                            subject=subject,
                            device=args.device,
                            n_iter=args.iterations,
                            n_samples=args.num_samples,
                            n_branches=args.branches)

        
    with torch.no_grad():
        
        # Load in the correct tensor for that vector. 
        print("torch.load()")
        coco_clip = torch.load("data/preprocessed_data/{}_73k.pt".format("c_i"))
        coco_gnet = torch.load("data/preprocessed_data/{}_73k.pt".format(GNET.vector))
        coco_scs  = torch.load("data/preprocessed_data/{}_73k.pt".format(SCS.vector))
        
        # Create the empty tensor for that vector. 
        coco_preds_clip = torch.zeros((73000, GNET.x_size))
        coco_preds_gnet = torch.zeros((73000, GNET.x_size))
        coco_preds_scs  = torch.zeros((73000, SCS.x_size))
        
        # Calculate predictions for all the images with the encoders predict function. 
        for i in tqdm(range(4), desc="predicting images"):
            
            coco_preds_clip[18250*i:18250*i + 18250] = CLIP.predict(coco_clip[18250*i:18250*i + 18250]).cpu()
            coco_preds_gnet[18250*i:18250*i + 18250] = GNET.predict(coco_gnet[18250*i:18250*i + 18250]).cpu()
            coco_preds_scs[18250*i:18250*i  + 18250] = SCS.predict(coco_scs[18250*i:18250*i + 18250]).cpu()
        
        # Save the tensor of predictions to the correct subject. 
        torch.save(coco_preds_clip, "data/preprocessed_data/subject{}/clip_coco_beta_primes.pt".format(subject))
        torch.save(coco_preds_gnet, "data/preprocessed_data/subject{}/gnet_coco_beta_primes.pt".format(subject))
        torch.save(coco_preds_scs,  "data/preprocessed_data/subject{}/scs_coco_beta_primes.pt".format(subject))
        
        # Load in the training dataframe to select the correct predicted betas for for that subject. 
        stim_descriptions = pd.read_csv('data/nsddata/experiments/nsd/nsd_stim_info_merged.csv', index_col=0)
        subject_data = stim_descriptions[(stim_descriptions['subject{}'.format(subject)] != 0)]
        
        # Create the tensors of predicted brain scans for each subject. 
        subject_sizes = [0, 15724, 14278, 0, 0, 13039, 0, 12682]
        coco_subject_preds_clip = torch.zeros((27000, subject_sizes[subject]))
        coco_subject_preds_gnet = torch.zeros((27000, subject_sizes[subject]))
        coco_subject_preds_scs  = torch.zeros((27000, subject_sizes[subject]))
        
        # Collect the stimuli for that subject. 
        subj = "subject" + str(subject)
        stim_descriptions = pd.read_csv('data/nsddata/experiments/nsd/nsd_stim_info_merged.csv', index_col=0)
        subjx = stim_descriptions[stim_descriptions[subj] != 0]
        
        # Create the tensor of beta primes for that subject. 
        for index in tqdm(range(0, 27000), desc="vector loader subject{}".format(subject)):
            scanId = int(subjx.loc[(subjx[subj + "_rep0"] == i+1) | (subjx[subj + "_rep1"] == i+1) | (subjx[subj + "_rep2"] == i+1)].nsdId)
            coco_subject_preds_clip[index] = coco_preds_clip[scanId]
            coco_subject_preds_gnet[index] = coco_preds_gnet[scanId]
            coco_subject_preds_scs[index]  = coco_preds_scs[scanId]
           
        # Save the specific beta primes for that subject.      
        torch.save(coco_subject_preds_clip,  "data/preprocessed_data/subject{}/clip_subject_specific_beta_primes.pt".format(subject))
        torch.save(coco_subject_preds_gnet,  "data/preprocessed_data/subject{}/gnet_subject_specific_beta_primes.pt".format(subject))
        torch.save(coco_subject_preds_scs,   "data/preprocessed_data/subject{}/scs_subject_specific_beta_primes.pt".format(subject))
        
    # Delete the class instances once 
    # predicts for that subject are done. 
    del CLIP
    del GNET
    del SCS
