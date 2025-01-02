import os
# os.chdir("..")
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
# for sub in [1,2,5,7]:
#     os.makedirs("data/nsddata_betas/ppdata/subj{:02d}/func1pt8mm/betas_fithrf_GLMdenoise_RR/".format(sub), exist_ok=True)
#     for sess in range(1,41):
#         #GLM Denoise
#         try:
#             os.system('rsync --info=progress2 /export/raid1/home/surly/raid4/kendrick-data/nsd/nsddata_betas/ppdata/subj{:02d}/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session{:02d}.nii.gz /home/naxos2-raid25/kneel027/home/kneel027/StableDiffusionReconstruction/nsd/nsddata_betas/ppdata/subj{:02d}/func1pt8mm/betas_fithrf_GLMdenoise_RR/'.format(sub,sess,sub))
#         except:
#             print('subj{:02d} sess{:02d} failed'.format(sub,sess))
        
# # Move ROIs
for sub in [1,2,3,4,5,6,7,8]:
    os.makedirs('nsddata/freesurfer/subj{:02d}/label/'.format(sub), exist_ok=True)
    os.system('rsync --info=progress2 /export/raid1/home/surly/kendrick-data/nsd/nsddata/freesurfer/subj{:02d}/label/* nsddata/freesurfer/subj{:02d}/label/'.format(sub,sub))
    os.makedirs('nsddata/ppdata/subj{:02d}/anat/roi/other/'.format(sub), exist_ok=True)
    os.system('rsync --info=progress2 /export/raid1/home/surly/kendrick-data/nsd/nsddata/ppdata/subj{:02d}/anat/roi/other/* nsddata/ppdata/subj{:02d}/anat/roi/other/'.format(sub,sub))
# # Move Imagery Betas
# for sub in [1,2,3,4,5,6,7,8]:
#     os.makedirs("data/nsddata_betas/ppdata/subj{:02d}/func1pt8mm/nsdimagerybetas_fithrf/".format(sub), exist_ok=True)
#     os.makedirs("data/nsddata_betas/ppdata/subj{:02d}/func1pt8mm/nsdimagerybetas_fithrf_GLMdenoise_RR/".format(sub), exist_ok=True)
#     #copy b2 betas
#     # os.system('rsync --info=progress2 /export/raid1/home/surly/raid4/kendrick-data/nsd/nsddata_betas/ppdata/subj{:02d}/func1pt8mm/nsdimagerybetas_fithrf/betas_nsdimagery.nii.gz data/nsddata_betas/ppdata/subj{:02d}/func1pt8mm/nsdimagerybetas_fithrf/'.format(sub,sub))
#     #copy b3 betas
#     os.system('rsync --info=progress2 /export/raid1/home/surly/raid4/kendrick-data/nsd/nsddata_betas/ppdata/subj{:02d}/func1pt8mm/nsdimagerybetas_fithrf_GLMdenoise_RR/betas_nsdimagery.nii.gz data/nsddata_betas/ppdata/subj{:02d}/func1pt8mm/nsdimagerybetas_fithrf_GLMdenoise_RR/'.format(sub,sub))
    
# for sub in [1,2,5,7]:
#     os.system('rsync --info=progress2 /export/raid1/home/surly/raid4/kendrick-data/nsd/nsddata/ppdata/subj{:02d}/behav/responses.tsv data/nsddata/ppdata/subj{:02d}/behav/'.format(sub,sub))


# MOVE IMAGERY FILES TO DEDICATED DIRECTORY FOR SHARING

# Move ROIs
# for sub in [1,2,3,4,5,6,7,8]:
#         os.makedirs('data_sharing/nsddata/ppdata/subj{:02d}/func1pt8mm/roi/'.format(sub), exist_ok=True)
#         os.system('rsync --info=progress2 /export/raid1/home/surly/raid4/kendrick-data/nsd/nsddata/ppdata/subj{:02d}/func1pt8mm/roi/* data_sharing/nsddata/ppdata/subj{:02d}/func1pt8mm/roi/'.format(sub,sub))

# # Move Imagery Betas
# for sub in [1,2,3,4,5,6,7,8]:
#         os.makedirs("data_sharing/nsddata_betas/ppdata/subj{:02d}/func1pt8mm/nsdimagerybetas_fithrf/".format(sub), exist_ok=True)
#         os.makedirs("data_sharing/nsddata_betas/ppdata/subj{:02d}/func1pt8mm/nsdimagerybetas_fithrf_GLMdenoise_RR/".format(sub), exist_ok=True)
#         #copy b2 betas
#         os.system('rsync --info=progress2 /export/raid1/home/surly/raid4/kendrick-data/nsd/nsddata_betas/ppdata/subj{:02d}/func1pt8mm/nsdimagerybetas_fithrf/betas_nsdimagery.nii.gz data_sharing/nsddata_betas/ppdata/subj{:02d}/func1pt8mm/nsdimagerybetas_fithrf/'.format(sub,sub))
#         #copy b3 betas
#         os.system('rsync --info=progress2 /export/raid1/home/surly/raid4/kendrick-data/nsd/nsddata_betas/ppdata/subj{:02d}/func1pt8mm/nsdimagerybetas_fithrf_GLMdenoise_RR/betas_nsdimagery.nii.gz data_sharing/nsddata_betas/ppdata/subj{:02d}/func1pt8mm/nsdimagerybetas_fithrf_GLMdenoise_RR/'.format(sub,sub))
    
# Move Jesse's Current Biology dataset
# Ghislain's averaged betas
# for sub in ["TN_new", "CO_new", "LF_new"]:
#         os.makedirs(f"Jesse-CB-Imagery/{sub}/")
#         os.system(f'rsync --info=progress2 /export/raid1/home/styvesg/data/imagery_rf/{sub}/Stimuli_metadata.pkl Jesse-CB-Imagery/{sub}/Stimuli_metadata.pkl')
#         os.system(f'rsync --info=progress2 /export/raid1/home/styvesg/data/imagery_rf/{sub}/Stimuli.h5py Jesse-CB-Imagery/{sub}/Stimuli.h5py')
#         os.system(f'rsync --info=progress2 /export/raid1/home/styvesg/data/imagery_rf/{sub}/Voxels.h5py Jesse-CB-Imagery/{sub}/Voxels.h5py')
#         os.system(f'rsync --info=progress2 /export/raid1/home/styvesg/data/imagery_rf/{sub}/Voxels_metadata.pkl Jesse-CB-Imagery/{sub}/Voxels_metadata.pkl')

#Tiasha's single trial beta preparations
# for sub in ["TN", "CO", "LF"]:
        # os.system(f'rsync --info=progress2 -r /home/naxos2-raid25/kneel027/home/kneel027/home/tsaharoy/imageryRF/glmsingle_data/GLMsingle_results_{sub} Jesse-CB-Imagery/single_trial_betas/{sub}/GLMsingle_results')
        # os.system(f'rsync --info=progress2 -r /home/naxos2-raid25/kneel027/home/kneel027/home/breedlov/all_processed_subjects/{sub}/ROIs/ Jesse-CB-Imagery/single_trial_betas/{sub}/ROIs/')
        # os.system(f'rsync --info=progress2 -r /home/naxos2-raid25/kneel027/home/kneel027/home/breedlov/all_processed_subjects/{sub}_old/ROIs/{sub}_prob_ROIs/ Jesse-CB-Imagery/single_trial_betas/{sub}/ROIs/')
# for i in range(1,9):
#         os.system(f'rsync --info=progress2 -r /home/naxos2-raid25/kneel027/home/kneel027/home/breedlov/imagery.rf.7T.July.2014/imagery_00{i}/frame_files/design_matrix.mat Jesse-CB-Imagery/stimuli/imagery_00{i}/frame_files/design_matrix.mat')

# os.system(f'rsync --info=progress2 -r /home/naxos2-raid25/kneel027/home/kneel027/home/breedlov/all_processed_subjects/{sub}/GLMd/DB_for_running_allGLMdenoise_w_newPreproc_{sub}_loopedGLM_noFWHM.csv Jesse-CB-Imagery/single_trial_betas/CO/GLMd/DB_for_running_allGLMdenoise_w_newPreproc_CO_loopedGLM_noFWHM.csv')
# os.system(f'rsync --info=progress2 -r /home/naxos2-raid25/kneel027/home/kneel027/home/breedlov/all_processed_subjects/{sub}/GLMd/DB_for_running_allGLMdenoise_w_newPreproc_LF_loopedGLM_noFWHM.csv Jesse-CB-Imagery/single_trial_betas/LF/GLMd/DB_for_running_allGLMdenoise_w_newPreproc_LF_loopedGLM_noFWHM.csv')
# os.system(f'rsync --info=progress2 -r /home/naxos2-raid25/kneel027/home/kneel027/home/breedlov/all_processed_subjects/TN/scripts/DB_for_running_allGLMdenoise_w_newPreproc_TN_loopedGLM_noFWHM.csv Jesse-CB-Imagery/single_trial_betas/TN/GLMd/DB_for_running_allGLMdenoise_w_newPreproc_TN_loopedGLM_noFWHM.csv')
# os.system(f'rsync --info=progress2 -r /home/naxos2-raid25/kneel027/home/kneel027/home/tsaharoy/imageryRF/stimuli/ Jesse-CB-Imagery/stimuli')
# os.system(f'zip -r ../Jesse-CB-Imagery.zip Jesse-CB-Imagery/')

# Move NSD Synthetic betas:
# os.makedirs("NSD-Synthetic/nsddata_stimuli/stimuli/nsdsynthetic/", exist_ok=True)
# os.makedirs("nsddata/experiments/nsdsynthetic/", exist_ok=True)
# os.system('rsync -r --info=progress2 /export/raid1/home/surly/raid4/kendrick-data/nsd/nsddata_stimuli/stimuli/nsdsynthetic/ nsddata_stimuli/stimuli/nsdsynthetic/')
# os.system('rsync -r --info=progress2 /export/raid1/home/surly/raid4/kendrick-data/nsd/nsddata/experiments/nsdsynthetic/nsdsynthetic_expdesign.mat nsddata/experiments/nsdsynthetic/nsdsynthetic_expdesign.mat')
# for sub in [1,2,3,4,5,6,7,8]:
#         os.makedirs("nsddata_betas/ppdata/subj{:02d}/func1pt8mm/nsdsyntheticbetas_fithrf/".format(sub), exist_ok=True)
#         os.makedirs("nsddata_betas/ppdata/subj{:02d}/func1pt8mm/nsdsyntheticbetas_fithrf_GLMdenoise_RR/".format(sub), exist_ok=True)
#         os.makedirs("nsddata/stimuli/nsdsynthetic/".format(sub), exist_ok=True)
#         #copy b2 betas
#         os.system('rsync -r --info=progress2 /export/raid1/home/surly/raid4/kendrick-data/nsd/nsddata_betas/ppdata/subj{:02d}/func1pt8mm/nsdsyntheticbetas_fithrf/ nsddata_betas/ppdata/subj{:02d}/func1pt8mm/nsdsyntheticbetas_fithrf_GLMdenoise_RR/'.format(sub,sub))
#         #copy b3 betas
#         os.system('rsync -r --info=progress2 /export/raid1/home/surly/raid4/kendrick-data/nsd/nsddata_betas/ppdata/subj{:02d}/func1pt8mm/nsdsyntheticbetas_fithrf_GLMdenoise_RR/ nsddata_betas/ppdata/subj{:02d}/func1pt8mm/nsdsyntheticbetas_fithrf_GLMdenoise_RR/'.format(sub,sub))
#         # copy stimuli
#         os.system('rsync -r --info=progress2 /export/raid1/home/surly/raid4/kendrick-data/nsd/nsddata/stimuli/nsdsynthetic/ nsddata/stimuli/nsdsynthetic/')
# os.system(f'zip -r ../NSD-Synthetic.zip NSD-Synthetic/')