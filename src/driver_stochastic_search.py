import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
from utils import *
from tqdm import tqdm
from library_decoder import LibraryDecoder
from vdvae import VDVAE
from nsd_access import NSDAccess
from stochastic_search import StochasticSearch
from diffusers import StableUnCLIPImg2ImgPipeline
from torchmetrics import PearsonCorrCoef
from decoder_uc import Decoder_UC
import wandb
import math

def main():
    generateTestSamples(experiment_title="SCS Plan B Samples", 
                        idx=[0, 15], 
                        modelParams=["gnetEncoder"],
                        subject=1,
                        n_samples=100,
                        n_iter=10,
                        n_branches=4,
                        ae=True,  
                        refine_clip = False,
                        refine_z = True,
                        custom_weighting=False,
                        library=False)
    SCS = StochasticSearch(modelParams=["gnetEncoder", "clipEncoder"],
                          device="cuda:0",
                          n_iter=6,
                          n_samples=100,
                          n_branches=4,
                          ae=True)
    for index in [46, 0, 15, 52]:
        for iteration in range(-1, 10):
            SCS.generate_image(experiment_title="SCS Plan B Samples", sample=index, iteration=iteration, n=10)
    # SCS.generate_image_distribution(experiment_title="SCS UC LD 6:100:4 Dual Guided clip_iter 39", sample=0, iteration=-1, n=10)
    # SCS.generate_image_distribution(experiment_title="SCS UC LD 6:100:4 Dual Guided clip_iter 39", sample=0, iteration=0, n=10)
    # SCS.generate_image_distribution(experiment_title="SCS UC LD 6:100:4 Dual Guided clip_iter 39", sample=0, iteration=1, n=10)
    # SCS.generate_image_distribution(experiment_title="SCS UC LD 6:100:4 Dual Guided clip_iter 39", sample=0, iteration=2, n=10)
    # SCS.generate_image_distribution(experiment_title="SCS UC LD 6:100:4 Dual Guided clip_iter 39", sample=0, iteration=3, n=10)
    # SCS.generate_image_distribution(experiment_title="SCS UC LD 6:100:4 Dual Guided clip_iter 39", sample=0, iteration=4, n=10)
    # SCS.generate_image_distribution(experiment_title="SCS UC LD 6:100:4 Dual Guided clip_iter 39", sample=0, iteration=5, n=10)
    # # SCS.generate_accuracy_weights()
    # process_x_encoded(SCS)
    # SCS.benchmark_config()
    
    
    # generateTestSamples(experiment_title="SCS UC LD 6:100:4 Dual Guided clip_iter 41", 
    #                     idx=[i for i in range(0, 20)], 
    #                     modelParams=["gnetEncoder", "clipEncoder"],
    #                     subject=1,
    #                     n_samples=100,
    #                     n_iter=6,
    #                     n_branches=4,
    #                     ae=True,  
    #                     refine_clip = True,
    #                     refine_z = True,
    #                     custom_weighting=False,
    #                     library=True)
    
def generateTestSamples(experiment_title, 
                        idx, 
                        modelParams=["gnetEncoder, clipEncoder"], 
                        subject=1,
                        n_samples=100,
                        n_iter=6,
                        n_branches=4, 
                        ae=True, 
                        refine_clip = True,
                        refine_z = True,
                        custom_weighting=False,
                        library=True):    
    with torch.no_grad():
        os.makedirs("reconstructions/subject{}/{}/".format(subject, experiment_title), exist_ok=True)
        os.makedirs("logs/subject{}/{}/".format(subject, experiment_title), exist_ok=True)
        nsda = NSDAccess('/export/raid1/home/surly/raid4/kendrick-data/nsd', '/export/raid1/home/kneel027/nsd_local')
            # Load data and targets
        _, _, x_test, _, _, targets_clips, trials = load_nsd(vector="c_img_uc", subject=subject, loader=False, average=False, nest=True)
        _, _, x_test_avg, _, _, targets_vdvae, _ = load_nsd(vector="z_vdvae", subject=subject, loader=False, average=True, nest=False)
        x_test = x_test[idx]
        x_test_avg = x_test_avg[idx]
        
        targets_vdvae = normalize_vdvae(targets_vdvae[idx]).reshape((len(idx), 1, 91168))
        targets_clips = targets_clips[idx].reshape((len(idx), 1, 1024))
        
        LD = LibraryDecoder(configList=modelParams,
                            subject=subject,
                            ae=ae,
                            device="cuda")
        output_images  = LD.predict(x_test, vector="images", topn=1)
        if library:
            output_clips = LD.predict(x_test, vector="c_img_uc", topn=200).reshape((len(idx), 1, 1024))
            del LD
            LD_v = LibraryDecoder(configList=["gnetEncoder"],
                                subject=subject,
                                ae=ae,
                                mask=torch.load("masks/subject{}/early_vis_big.pt".format(subject)),
                                device="cuda")
            output_vdvae = LD_v.predict(x_test, vector="z_vdvae", topn=25)
            del LD_v
        else:
            del LD
            D_c = Decoder_UC(inference=True,
                            config="clipDecoder",
                            device="cuda")
            output_clips = D_c.predict(x_test_avg).reshape((len(idx), 1, 1024))
            del D_c
            
            D_v = Decoder_UC(inference=True,
                            config="vdvaeDecoder",
                            device="cuda")
            output_vdvae = D_v.predict(x_test_avg)
            del D_v
        output_vdvae = normalize_vdvae(output_vdvae).reshape((len(idx), 1, 91168))
        
        # Initialize Models
        V = VDVAE()
        SCS = StochasticSearch(modelParams=modelParams,
                            device="cuda:0",
                            n_iter=n_iter,
                            n_samples=n_samples,
                            n_branches=n_branches,
                            ae=ae)
                
        PeC = PearsonCorrCoef(num_outputs=len(idx)).to("cpu")
        PeC1 = PearsonCorrCoef(num_outputs=1).to("cpu")
        
        #Log the CLIP scores
        clip_scores = np.array(PeC(output_clips[:,0,:].moveaxis(0,1).to("cpu"), targets_clips[:,0,:].moveaxis(0,1).to("cpu")).detach())
        np.save("logs/subject{}/{}/decoded_c_img_PeC.npy".format(subject, experiment_title), clip_scores)
        for i, val in enumerate(tqdm(idx, desc="Reconstructing samples")):
            os.makedirs("reconstructions/subject{}/{}/{}/".format(subject, experiment_title, val), exist_ok=True)
            os.makedirs("logs/subject{}/{}/{}/".format(subject, experiment_title, val), exist_ok=True)
            torch.save(output_clips[i], "reconstructions/subject{}/{}/{}/decoded_clip.pt".format(subject, experiment_title, val))
            # Generate reconstructions
            output_v = V.reconstruct(output_vdvae[i])
            target_v = V.reconstruct(targets_vdvae[i])
            output_c = SCS.R.reconstruct(image_embeds=output_clips[i], negative_prompt="text, caption", strength=1.0)
            target_c = SCS.R.reconstruct(image_embeds=targets_clips[i], negative_prompt="text, caption", strength=1.0)
            output_cv = SCS.R.reconstruct(image=output_v, image_embeds=output_clips[i], negative_prompt="text, caption", strength=0.9)
            target_cv = SCS.R.reconstruct(image=target_v, image_embeds=targets_clips[i], negative_prompt="text, caption", strength=0.9)
            scs_reconstruction, image_list, clip_list, score_list, var_list = SCS.search(beta=x_test[i], c_i=output_clips[i], init_img=None, refine_z=refine_z, refine_clip=refine_clip, n=n_samples, max_iter=n_iter, n_branches=n_branches, custom_weighting=custom_weighting)
            
            # Log the data to a file
            np.save("logs/subject{}/{}/{}/score_list.npy".format(subject, experiment_title, val), np.array(score_list))
            np.save("logs/subject{}/{}/{}/var_list.npy".format(subject, experiment_title, val), np.array(var_list))
            
            # Test if CLIP is improving
            decoded_c_i = SCS.R.encode_image_raw(output_c).reshape((1024,))
            scs_c_i = SCS.R.encode_image_raw(scs_reconstruction).reshape((1024,))
            new_clip_score = float(PeC1(scs_c_i.to("cpu"), targets_clips[i].reshape((1024,)).to("cpu").detach()))
            old_clip_score = float(PeC1(decoded_c_i.to("cpu"), targets_clips[i].reshape((1024,)).to("cpu").detach()))
            tqdm.write("CLIP IMPROVEMENT: {} -> {}".format(old_clip_score, new_clip_score))
            
            # Format output diagram
            nsdId = trials[val]
            ground_truth_np_array = nsda.read_images([nsdId], show=True)
            ground_truth = Image.fromarray(ground_truth_np_array[0])
            ground_truth = ground_truth.resize((768, 768), resample=Image.Resampling.LANCZOS)
            rows = int(math.ceil(len(image_list)/2 + 4))
            columns = 2
            images = [ground_truth, scs_reconstruction, target_v, output_v, target_c, output_c, target_cv, output_cv]
            captions = ["Ground Truth", "Search Reconstruction", "Ground Truth VDVAE", "Decoded VDVAE", "Ground Truth CLIP", "Decoded CLIP Only", "Ground Truth CLIP+VDVAE", "Decoded CLIP+VDVAE"]
            intermediate_clip_scores = []
            for j in range(len(image_list)):
                images.append(image_list[j])
                intermediate_clip = SCS.R.encode_image_raw(image_list[j]).reshape((1024,)).to("cpu")
                intermediate_clip_scores.append(float(PeC1(intermediate_clip.to("cpu"), targets_clips[i].reshape((1024,)).to("cpu").detach())))
                captions.append("BC: {} CLIP: {}".format(round(score_list[j], 3), round(intermediate_clip_scores[j], 3)))
            # Log the data to a file
            np.save("logs/subject{}/{}/{}/clip_list.npy".format(subject, experiment_title, val), np.array(intermediate_clip_scores))
            figure = tileImages("{}:{}".format(experiment_title, val), images, captions, rows, columns)
            
            figure.save('reconstructions/subject{}/{}/{}.png'.format(subject, experiment_title, val))
            count = 0
            images.append(process_image(output_images[i]))
            captions.append("Library Reconstruction")
            for j in range(len(images)):
                if("BC" in captions[j]):
                    images[j].save("reconstructions/subject{}/{}/{}/iter_{}.png".format(subject, experiment_title, val, count))
                    torch.save(clip_list[count], "reconstructions/subject{}/{}/{}/iter_clip_{}.pt".format(subject, experiment_title, val, count))
                    count +=1
                else:
                    images[j].save("reconstructions/subject{}/{}/{}/{}.png".format(subject, experiment_title, val, captions[j]))
if __name__ == "__main__":
    main()