import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
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
from vdvae import VDVAE

# First URL: This is the original read-only NSD file path (The actual data)
# Second URL: Local files that we are adding to the dataset and need to access as part of the data
# Object for the NSDAccess package
nsda = NSDAccess('/export/raid1/home/surly/raid4/kendrick-data/nsd', '/export/raid1/home/kneel027/nsd_local')

def main():
    # reconstructVDVAE(experiment_title="LD VDVAE S1 gnetEncoder 8 V1",
    #                 subject=1,
    #                 idx=[i for i in range(0, 20)],
    #                 ae=True,
    #                 mask=torch.load("masks/subject1/V1.pt"),
    #                 average=True,
    #                 config=["gnetEncoder"])
    # reconstructVDVAE(experiment_title="LD VDVAE S1 gnetEncoder 9 early_vis 2",
    #                 subject=1,
    #                 idx=[i for i in range(0, 20)],
    #                 ae=True,
    #                 mask=torch.load("masks/subject1/early_vis.pt"),
    #                 average=True,
    #                 config=["gnetEncoder"])
    # reconstructVDVAE(experiment_title="LD VDVAE S1 gnetEncoder 10 nsd_general",
    #                 subject=1,
    #                 idx=[i for i in range(0, 20)],
    #                 ae=True,
    #                 mask=None,
    #                 average=True,
    #                 config=["gnetEncoder"])
    benchmark_library("z_vdvae", subject=1, average=True, config=["gnetEncoder"])
    subjects = [2, 5, 7]
    for subject in subjects:
        benchmark_library(vector="c_img_uc", subject=subject, average=True, config=["gnetEncoder", "clipEncoder"])
        benchmark_library(vector="z_vdvae", subject=subject, average=True, config=["gnetEncoder"])
    # benchmark_library("c_img_uc", subject=2, average=True, config=["gnetEncoder", "clipEncoder"])
    # benchmark_library("c_img_uc", subject=5, average=True, config=["gnetEncoder", "clipEncoder"])
    # benchmark_library("c_img_uc", subject=7, average=True, config=["gnetEncoder", "clipEncoder"])
    # benchmark_library("z_vdvae", subject=1, average=True, config=["gnetEncoder"])
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
    # reconstructNImages(experiment_title="LD S1 CLIP+VDVAE dualGuided",
    #                    subject=1,
    #                    idx=[i for i in range(0, 20)],
    #                    ae=True,
    #                    mask=None,
    #                    config=["gnetEncoder", "clipEncoder"])
    # reconstruct_test_samples("SCS UC 747 10:100:4 0.4 Exp3 AE", idx=[], average=True)
    # reconstruct_test_samples("SCS UC 747 10:100:4 0.5 Exp3 AE", idx=[], average=True)
    # reconstruct_test_samples("SCS UC 747 10:100:4 0.6 Exp3 AE", idx=[], average=True)



def reconstructNImages(experiment_title, subject, idx, ae=True, mask=None, config=["gnetEncoder"]):
    
    os.makedirs("reconstructions/subject{}/{}/".format(subject, experiment_title), exist_ok=True)
    _, _, x, _, _, targets_clips, trials = load_nsd(vector="c_img_uc", subject=subject, loader=False, average=False, nest=True)
    _, _, _, _, _, targets_vdvae, _ = load_nsd(vector="z_vdvae", subject=subject, loader=False, average=True, nest=True)
    x = x[idx]
    
    targets_vdvae = normalize_vdvae(targets_vdvae[idx]).reshape((len(idx), 1, 91168))
    targets_clips = targets_clips[idx].reshape((len(idx), 1, 1024))
    
    LD = LibraryDecoder(configList=config,
                        subject=subject,
                        ae=ae,
                        device="cuda")
    output_images  = LD.predict(x, vector="images")
    
    output_clips = LD.predict(x, vector="c_img_uc").reshape((len(idx), 1, 1024))
    del LD

    LD_v = LibraryDecoder(configList=["gnetEncoder"],
                        subject=subject,
                        ae=ae,
                        mask=torch.load("masks/subject{}/early_vis_big.pt".format(subject)),
                        device="cuda")
    output_vdvae = LD_v.predict(x, vector="z_vdvae", topn=25)
    output_vdvae = normalize_vdvae(output_vdvae).reshape((len(idx), 1, 91168))
    del LD_v
    V = VDVAE()
    R = StableUnCLIPImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16, variation="fp16").to("cuda")
    for i, val in enumerate(tqdm(idx, desc="Generating reconstructions")):
        tqdm.write("{}, {}".format(i, val))
        print(output_vdvae[i].shape, output_vdvae[i].device)
        rec_target_vdvae = V.reconstruct(latents=targets_vdvae[i])
        rec_vdvae = V.reconstruct(latents=output_vdvae[i])
        
        best_coco_image = process_image(output_images[i])
        
        rec_clip = R.reconstruct(image_embeds=output_clips[i], strength=1)
        rec_target_clip = R.reconstruct(image_embeds=targets_clips[i], strength=1)
        
        rec_combined = R.reconstruct(image=rec_vdvae, image_embeds=output_clips[i], strength=0.9)
        rec_target_combined = R.reconstruct(image=rec_target_vdvae, image_embeds=targets_clips[i], strength=0.9)
        
        # returns a numpy array 
        nsdId = trials[val]
        ground_truth_np_array = nsda.read_images([nsdId], show=False)
        ground_truth = Image.fromarray(ground_truth_np_array[0])
        ground_truth = ground_truth.resize((768, 768), resample=Image.Resampling.LANCZOS)
        empty = Image.new('RGB', (768, 768), color='white')
        rows = 4
        columns = 2
        images = [ground_truth,         best_coco_image, 
                  rec_target_vdvae,     rec_vdvae,
                  rec_target_clip,      rec_clip,
                  rec_target_combined,  rec_combined]
        captions = ["Ground Truth",          "Best COCO Image", 
                  "Ground Truth VDVAE",      "Decoded VDVAE", 
                  "Ground Truth CLIP",       "Decoded CLIP",
                  "Ground Truth CLIP+VDVAE", "Decoded CLIP+VDVAE"]
        figure = tileImages(experiment_title + ": " + str(val), images, captions, rows, columns)
        
        figure.save('reconstructions/subject{}/{}/{}.png'.format(subject, experiment_title, val))
        
def reconstructVDVAE(experiment_title, subject, idx, ae=True, mask=None, average=True, config=["gnetEncoder", "clipEncoder"]):
    
    os.makedirs("reconstructions/subject{}/{}/".format(subject, experiment_title), exist_ok=True)
    _, _, x, _, _, targets_vdvae, trials = load_nsd(vector="z_vdvae", subject=subject, loader=False, average=False, nest=True)
    _, _, x_avg, _, _, _, _ = load_nsd(vector="z_vdvae", subject=subject, loader=False, average=True)
    x = x[idx]
    x_avg = x_avg[idx]
    LD_i = LibraryDecoder(vector="images",
                        configList=config,
                        subject=subject,
                        ae=ae,
                        device="cuda")
    output_images = LD_i.predict(x, topn=5)
    del LD_i

    Dv = Decoder_UC(config="vdvaeDecoder",
                 inference=True, 
                 subject=subject,
                 device="cuda",
                 )
    outputs_decoded_vdvae = Dv.predict(x=x_avg)
    outputs_decoded_vdvae = normalize_vdvae(outputs_decoded_vdvae.to("cpu"))
    del Dv

    LD_v = LibraryDecoder(vector="z_vdvae",
                        configList=["gnetEncoder"],
                        subject=subject,
                        ae=ae,
                        mask=torch.load("masks/subject{}/early_vis.pt".format(subject)),
                        device="cuda")
 
    v_0 = normalize_vdvae(LD_v.predict(x, topn=100).reshape((len(idx), 1, 91168)))
    v_1 = normalize_vdvae(LD_v.predict(x, topn=50).reshape((len(idx), 1, 91168)))
    v_2 = normalize_vdvae(LD_v.predict(x, topn=25).reshape((len(idx), 1, 91168)))
    v_3 = normalize_vdvae(LD_v.predict(x, topn=10).reshape((len(idx), 1, 91168)))
    v_4 = normalize_vdvae(LD_v.predict(x, topn=1).reshape((len(idx), 1, 91168)))
    del LD_v
    targets_vdvae = normalize_vdvae(targets_vdvae).to("cuda")
    
    V = VDVAE()
    for i, val in enumerate(tqdm(idx, desc="Generating reconstructions")):
        tqdm.write("{}, {}".format(i, val))
        
        i_0 = process_image(output_images[i,0])
        i_1 = process_image(output_images[i,1])
        i_2 = process_image(output_images[i,2])
        i_3 = process_image(output_images[i,3])
        i_4 = process_image(output_images[i,4])

        rec_v_0_norm = V.reconstruct(v_0[i])
        rec_v_1_norm = V.reconstruct(v_1[i])
        rec_v_2_norm = V.reconstruct(v_2[i])
        rec_v_3_norm = V.reconstruct(v_3[i])
        rec_v_4_norm = V.reconstruct(v_4[i])
        
        rec_dec_v = V.reconstruct(outputs_decoded_vdvae[i])
        
        # returns a numpy array 
        nsdId = trials[val]
        ground_truth_np_array = nsda.read_images([nsdId], show=False)
        ground_truth = Image.fromarray(ground_truth_np_array[0])
        ground_truth = ground_truth.resize((768, 768), resample=Image.Resampling.LANCZOS)
        empty = Image.new('RGB', (768, 768), color='white')
        rows = 6
        columns = 2
        images = [ground_truth,  rec_dec_v,
                  i_0,    rec_v_0_norm,
                  i_1,    rec_v_1_norm,
                  i_2,    rec_v_2_norm,
                  i_3,    rec_v_3_norm,
                  i_4,    rec_v_4_norm,]
        captions = ["ground_truth","Decoded",
                  "output_i_0",   "top 100",
                  "output_i_1",   "top 50",
                  "output_i_2",  "top 25", 
                  "output_i_3",  "top 10", 
                  "output_i_4", "top 1",]
        
        figure = tileImages(experiment_title + ": " + str(val), images, captions, rows, columns)
        
        figure.save('reconstructions/subject{}/{}/{}.png'.format(subject, experiment_title, val))
        

def benchmark_library(vector, subject=1, average=True, config=["gnetEncoder"]):
    device = "cuda"
    LD = LibraryDecoder(configList=config,
                        subject=subject,
                        device=device)
    LD.benchmark(average=average, vector=vector)



def reconstruct_test_samples(experiment_title, subject, idx=[], test=False, average=True):
    if(len(idx) == 0):
        for file in os.listdir("reconstructions/subject{}/{}/".format(subject, experiment_title)):
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
        pil_image.save("reconstructions/subject{}/{}/{}/Library Reconstruction.png".format(subject, experiment_title, idx[i]))

if __name__ == "__main__":
    main()