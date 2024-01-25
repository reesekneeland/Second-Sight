import os

# Move Experiment Infos
# os.makedirs("data/nsddata/experiments/nsd/", exist_ok=True)
# os.system('rsync --info=progress2 /export/raid1/home/surly/raid4/kendrick-data/nsd/nsddata/experiments/nsd/nsd_stim_info_merged.csv data/nsddata/experiments/nsd/nsd_stim_info_merged.csv')

# # Move Stimuli
# os.makedirs("data/nsddata_stimuli/stimuli/nsd/", exist_ok=True)
# os.system('rsync --info=progress2 /export/raid1/home/surly/raid4/kendrick-data/nsd/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5 data/nsddata_stimuli/stimuli/nsd/')

# Move Betas
# for sub in [1,2,5,7]:
#     os.makedirs("data/nsddata_betas/ppdata/subj{:02d}/func1pt8mm/betas_fithrf_GLMdenoise_RR/".format(sub), exist_ok=True)
#     for sess in range(1,41):
#         #GLM Denoise
#         try:
#             os.system('rsync --info=progress2 /export/raid1/home/surly/raid4/kendrick-data/nsd/nsddata_betas/ppdata/subj{:02d}/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session{:02d}.nii.gz data/nsddata_betas/ppdata/subj{:02d}/func1pt8mm/betas_fithrf_GLMdenoise_RR/'.format(sub,sess,sub))
#         except:
#             print('subj{:02d} sess{:02d} failed'.format(sub,sess))
        #fithrf
        # os.system('rsync --info=progress2 /export/raid1/home/surly/raid4/kendrick-data/nsd/nsddata_betas/ppdata/subj{:02d}/func1pt8mm/betas_fithrf/betas_session{:02d}.nii.gz data/nsddata_betas/ppdata/subj{:02d}/func1pt8mm/betas_fithrf/'.format(sub,sess,sub))
        
# Move betas to other repo
for sub in [1,2,5,7]:
    os.makedirs("data/nsddata_betas/ppdata/subj{:02d}/func1pt8mm/betas_fithrf_GLMdenoise_RR/".format(sub), exist_ok=True)
    for sess in range(1,41):
        #GLM Denoise
        try:
            os.system('rsync --info=progress2 /export/raid1/home/surly/raid4/kendrick-data/nsd/nsddata_betas/ppdata/subj{:02d}/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session{:02d}.nii.gz /home/naxos2-raid25/kneel027/home/kneel027/StableDiffusionReconstruction/nsd/nsddata_betas/ppdata/subj{:02d}/func1pt8mm/betas_fithrf_GLMdenoise_RR/'.format(sub,sess,sub))
        except:
            print('subj{:02d} sess{:02d} failed'.format(sub,sess))
        
# # Move ROIs
# for sub in [1,2,3,4,5,6,7,8]:
#     os.makedirs('data/nsddata/ppdata/subj{:02d}/func1pt8mm/roi/'.format(sub), exist_ok=True)
#     os.system('rsync --info=progress2 /export/raid1/home/surly/raid4/kendrick-data/nsd/nsddata/ppdata/subj{:02d}/func1pt8mm/roi/* data/nsddata/ppdata/subj{:02d}/func1pt8mm/roi/'.format(sub,sub))

# # Move Imagery Betas
# for sub in [1,2,3,4,5,6,7,8]:
#     os.makedirs("data/nsddata_betas/ppdata/subj{:02d}/func1pt8mm/nsdimagerybetas_fithrf/".format(sub), exist_ok=True)
#     os.system('rsync --info=progress2 /export/raid1/home/surly/raid4/kendrick-data/nsd/nsddata_betas/ppdata/subj{:02d}/func1pt8mm/nsdimagerybetas_fithrf/betas_nsdimagery.nii.gz data/nsddata_betas/ppdata/subj{:02d}/func1pt8mm/nsdimagerybetas_fithrf/'.format(sub,sub))
