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



def generate_beta_primes(subjects, device):
    models = ["hybrid", "clip", "gnet"]
    # Load in the training dataframe to select the correct predicted betas for for that subject. 
    stim_descriptions = pd.read_csv('data/nsddata/experiments/nsd/nsd_stim_info_merged.csv', index_col=0)
    subject_sizes = [0, 15724, 14278, 0, 0, 13039, 0, 12682]
    vectors = {"clip": torch.load("data/preprocessed_data/c_73k.pt").cpu(),
               "gnet": torch.load("data/preprocessed_data/images_73k.pt").cpu(),
               "hybrid": torch.load("data/preprocessed_data/images_73k.pt").cpu()}
    
    with torch.no_grad():
        for subject in subjects:
            print("Processing subject {}".format(subject))
            for modelType in models:
                print("Initializing {} model.".format(modelType))
                if modelType == "clip":
                    model = CLIPEncoder(inference=True,
                            subject=subject,
                            device=device)
                elif modelType == "gnet":
                    model = GNet8_Encoder(device=device,
                                    subject=subject)
                elif modelType == "hybrid":
                    model = StochasticSearch(modelParams=["gnet", "clip"],
                                        subject=subject,
                                        device=device)
            
                # Create the empty tensor for that vector. 
                coco_preds = torch.zeros((73000, subject_sizes[subject]))
                coco_ae_preds = torch.zeros((27750, subject_sizes[subject]))
                subjx = stim_descriptions[(stim_descriptions['subject{}'.format(subject)] != 0)]
                
                # Calculate predictions for all the images with the encoders predict function. 
                for i in tqdm(range(100), desc="predicting image batches"):
                    
                    coco_preds[730*i:730*i + 730] = model.predict(vectors[modelType][730*i:730*i + 730]).cpu()
            
                # Save the tensor of predictions to the correct subject. 
                torch.save(coco_preds, "data/preprocessed_data/subject{}/{}_coco_beta_primes.pt".format(subject, modelType))
                
                # Create the tensor of beta primes for that subject. 
                for i in tqdm(range(0, 27750), desc="vector loader subject{}".format(subject)):
                    scanId = int(subjx.loc[(subjx['subject{}_rep0'.format(subject)] == i+1) | (subjx['subject{}_rep1'.format(subject)] == i+1) | (subjx['subject{}_rep2'.format(subject)] == i+1)].nsdId)
                    coco_ae_preds[i] = coco_preds[scanId]
                
                # Save the specific beta primes for that subject.      
                torch.save(coco_ae_preds,  "data/preprocessed_data/subject{}/{}_ae_beta_primes.pt".format(subject, modelType))
            del model

if __name__ == '__main__':
    # Create the parser and add arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-s',
                        '--subjects', 
                        help="list of subjects to run the algorithm on, if not specified, will run on all subjects",
                        type=str,
                        default="1,2,5,7")

    parser.add_argument('-d',
                        '--device', 
                        help="cuda device to run predicts on.",
                        type=str,
                        default="cuda:0")

    args = parser.parse_args()
    subject_list = [int(sub) for sub in args.subjects.split(",")]
    generate_beta_primes(subject_list, args.device)