import os
import numpy as np
from scipy.special import erf
import nibabel as nib
import torch
from tqdm import tqdm

from clip_encoder       import CLIP_Encoder
from gnet8_encoder      import GNet8_Encoder
from stochastic_search  import StochasticSearch

# Create the parser and add arguments
parser = argparse.ArgumentParser()

parser.add_argument('-s',
                    '--subjects', 
                    help="list of subjects to run the algorithm on, if not specified, will run on all subjects",
                    type=list,
                    default=[1, 2, 5, 7])

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
print(args.log)


for subject in args.subjects:
    
    CLIP = CLIP_Encoder(inference=True,
                        subject=subject,
                        device=args.device)

    GNET = GNet8_Encoder(device=args.device,
                        subject=subject)

    SCS  = StochasticSearch(modelParams=["gnetEncoder", "clipEncoder"],
                            subject=subject,
                            device=args.device,
                            n_iter=args.iterations,
                            num_samples=args.num_samples,
                            n_branches=args.branches)

        
    with torch.no_grad():
        
        # Load in the correct tensor for that vector. 
        coco_clip = torch.load("data/preprocessed_data/{}_73k.pt".format(CLIP.vector))
        coco_gnet = torch.load("data/preprocessed_data/{}_73k.pt".format(GNET.vector))
        coco_scs  = torch.load("data/preprocessed_data/{}_73k.pt".format(SCS.vector))
        
        # Create the empty tensor for that vector. 
        coco_preds_clip = torch.zeros((73000, CLIP.x_size))
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
        
        
        os.makedirs(prep_path + "subject{}/x_encoded/".format(Encoder.subject), exist_ok=True)
        
        _, y = load_nsd(vector = Encoder.vector, subject=Encoder.subject, loader = False, split = False)
        
        
        torch.save(outputs, "data/preprocessed_data/subject{}/clip_subject_specific_beta_primes.pt".format(subject))
        torch.save(outputs, "data/preprocessed_data/subject{}/gnet_subject_specific_beta_primes.pt".format(subject))
        torch.save(outputs, "data/preprocessed_data/subject{}/scs_subject_specific_beta_primes.pt".format(subject))
        
    del CLIP
    del GNET
    del SCS
