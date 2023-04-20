import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
import torch
import numpy as np
from PIL import Image
from nsd_access import NSDAccess
from utils import *
from tqdm import tqdm
import yaml
from decoder_uc import Decoder_UC
from diffusers import StableUnCLIPImg2ImgPipeline
from library_decoder import LibraryDecoder

# First URL: This is the original read-only NSD file path (The actual data)
# Second URL: Local files that we are adding to the dataset and need to access as part of the data
# Object for the NSDAccess package
nsda = NSDAccess('/export/raid1/home/surly/raid4/kendrick-data/nsd', '/export/raid1/home/kneel027/nsd_local')

def main():
    benchmark_library("z_vdvae", subject=1, average=True, config=["gnetEncoder", "clipEncoder"])
    # benchmark_library("c_img_uc", subject=2, average=True, config=["gnetEncoder", "clipEncoder"])
    # benchmark_library("c_img_uc", subject=5, average=True, config=["gnetEncoder", "clipEncoder"])
    # benchmark_library("c_img_uc", subject=7, average=True, config=["gnetEncoder", "clipEncoder"])
    benchmark_library("z_vdvae", subject=1, average=True, config=["gnetEncoder"])
    # reconstructNImages(experiment_title="LD S1 CLIP clipEncoder no AE",
    #                    subject=1,
    #                    idx=[i for i in range(0, 20)],
    #                    ae=False,
    #                    mask=None,
    #                    average=True,
    #                    config=["clipEncoder"])
    # reconstructNImages(experiment_title="LD S1 CLIP gnetEncoder no AE",
    #                    subject=1,
    #                    idx=[i for i in range(0, 20)],
    #                    ae=False,
    #                    mask=None,
    #                    average=True,
    #                    config=["gnetEncoder"])
    # reconstructNImages(experiment_title="LD S1 CLIP dualGuided no AE",
    #                    subject=1,
    #                    idx=[i for i in range(0, 20)],
    #                    ae=False,
    #                    mask=None,
    #                    average=True,
    #                    config=["gnetEncoder", "clipEncoder"])
    # reconstruct_test_samples("SCS UC 747 10:100:4 0.4 Exp3 AE", idx=[], average=True)
    # reconstruct_test_samples("SCS UC 747 10:100:4 0.5 Exp3 AE", idx=[], average=True)
    # reconstruct_test_samples("SCS UC 747 10:100:4 0.6 Exp3 AE", idx=[], average=True)



            
# Encode latent z (1x4x64x64) and condition c (1x77x1024) tensors into an image
# Strength parameter controls the weighting between the two tensors
def reconstructNImages(experiment_title, subject, idx, ae=True, mask=None, average=True, config=["alexnetEncoder"]):
    
    os.makedirs("reconstructions/" + experiment_title + "/", exist_ok=True)
    _, _, x, _, _, targets_c_i, trials = load_nsd(vector="c_img_uc", subject=subject, loader=False, average=False, nest=True)
    _, _, x_averaged, _, _, targets_c_i, trials = load_nsd(vector="c_img_uc", subject=subject, loader=False, average=True, nest=True)
    x = x[idx]
    x_averaged = x_averaged[idx]
    
    
    
    LD_i = LibraryDecoder(vector="images",
                        configList=config,
                        subject=subject,
                        ae=ae,
                        device="cuda")
    output_images, _ = LD_i.rankCoco(x, average=average)
    output_images = output_images[:, 0:5]
    del LD_i
    LD_c = LibraryDecoder(vector="c_img_uc",
                        configList=config,
                        subject=subject,
                        ae=ae,
                        device="cuda")
    output_clips = LD_c.predict(x, average=average)
    output_clips = output_clips.reshape((len(idx), 1, 1024))
    output_clip_top1k, _ = LD_c.rankCoco(x)
    del LD_c
    
    Dc_i = Decoder_UC(config="clipDecoder",
                 inference=True, 
                 subject=subject,
                 device="cuda",
                 )
    outputs_c_d = Dc_i.predict(x=x_averaged).unsqueeze(0).reshape((len(idx), 1, 1024))
    print(outputs_c_d.shape)
    
    targets_c_i = targets_c_i[idx].reshape((len(idx), 1, 1024))
    R = StableUnCLIPImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16, variation="fp16").to("cuda")
    for i, val in enumerate(tqdm(idx, desc="Generating reconstructions")):
        tqdm.write("{}, {}".format(i, val))
        
        i_0 = process_image(output_images[i,0])
        i_1 = process_image(output_images[i,1])
        i_2 = process_image(output_images[i,2])
        i_3 = process_image(output_images[i,3])
        i_4 = process_image(output_images[i,4])
        
        
        ci_0 = torch.mean(output_clip_top1k[i,0:500], dim=0)
        ci_1 = torch.mean(output_clip_top1k[i,0:250], dim=0)
        ci_2 = torch.mean(output_clip_top1k[i,0:100], dim=0)
        ci_3 = torch.mean(output_clip_top1k[i,0:50], dim=0)
        ci_4 = output_clip_top1k[i,0]
        c_i_combined = output_clips[i]
        
        outputs_c_d = Dc_i.predict(x=x_averaged[i]).reshape((1, 1024))
        output_c_d = R.reconstruct(image_embeds=outputs_c_d, strength=1)
        target_c_i = R.reconstruct(image_embeds=targets_c_i[i])
        output_ci = R.reconstruct(image_embeds=c_i_combined)
        output_ci_0 = R.reconstruct(image_embeds=ci_0)
        output_ci_1 = R.reconstruct(image_embeds=ci_1)
        output_ci_2 = R.reconstruct(image_embeds=ci_2)
        output_ci_3 = R.reconstruct(image_embeds=ci_3)
        output_ci_4 = R.reconstruct(image_embeds=ci_4)
    
        
        # returns a numpy array 
        nsdId = trials[val]
        ground_truth_np_array = nsda.read_images([nsdId], show=False)
        ground_truth = Image.fromarray(ground_truth_np_array[0])
        ground_truth = ground_truth.resize((768, 768), resample=Image.Resampling.LANCZOS)
        empty = Image.new('RGB', (768, 768), color='white')
        rows = 7
        columns = 2
        images = [ground_truth, target_c_i, 
                  output_c_d,   output_ci,
                  i_0,          output_ci_0,
                  i_1,          output_ci_1,
                  i_2,          output_ci_2,
                  i_3,          output_ci_3,
                  i_4,          output_ci_4,]
        captions = ["ground_truth","target_c_i", 
                  "Decoded CLIP",     "top N", 
                  "output_i_0",   "top 500",
                  "output_i_1",   "top 250",
                  "output_i_2",   "top 100", 
                  "output_i_3",   "top 50", 
                  "output_i_4",   "top 1",]
        figure = tileImages(experiment_title + ": " + str(val), images, captions, rows, columns)
        
        figure.save('reconstructions/{}/{}.png'.format(experiment_title, val))
        
    
def benchmark_library(vector, subject=1, average=True, config=["gnetEncoder"]):
    device = "cuda"
    LD = LibraryDecoder(vector=vector,
                        configList=config,
                        subject=subject,
                        device=device)
    LD.benchmark(average=average)



def reconstruct_test_samples(experiment_title, idx=[], test=False, average=True):
    if(len(idx) == 0):
        for file in os.listdir("reconstructions/{}/".format(experiment_title)):
            if file.endswith(".png") and file not in ["Search Iterations.png", "Results.png"]:
                idx.append(int(file[:-4]))
        idx = sorted(idx)

    _, _, x, _, _, _, _ = load_nsd(vector="c_img_uc", loader=False, average=False, nest=True)
    
    
    x = x[idx]
    
    device = "cuda"
    LD = LibraryDecoder(vector="images",
                        configList=["alexnetEncoder"],
                        device=device)
    output_images = LD.predict(x, average=average)
    for i, image in enumerate(output_images):
        top_choice = image[0].reshape(425, 425, 3)
        top_choice = top_choice.detach().cpu().numpy().astype(np.uint8)
        pil_image = Image.fromarray(top_choice).resize((768, 768), resample=Image.Resampling.LANCZOS)
        pil_image.save("reconstructions/{}/{}/Library Reconstruction.png".format(experiment_title, idx[i]))

if __name__ == "__main__":
    main()