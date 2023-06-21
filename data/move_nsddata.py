import os
#os.system('ls -l')

# Download Experiment Infos
# os.system('aws s3 cp s3://natural-scenes-dataset/nsddata/experiments/nsd/nsd_expdesign.mat nsddata/experiments/nsd/')
# os.system('aws s3 cp s3://natural-scenes-dataset/nsddata/experiments/nsd/nsd_stim_info_merged.pkl nsddata/experiments/nsd/')

# Download Stimuli
os.makedirs("nsddata_stimuli/stimuli/nsd/", exist_ok=True)
os.system('cp /home/naxos2-raid25/kneel027/home/kneel027/home/surly/raid4/kendrick-data/nsd/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5 nsddata_stimuli/stimuli/nsd/')

# Download Betas
for sub in [2,5,7]:
    os.makedirs("nsddata_betas/ppdata/subj{:02d}/func1pt8mm/betas_fithrf_GLMdenoise_RR/".format(sub))
    for sess in range(1,38):
        os.system('cp /home/naxos2-raid25/kneel027/home/kneel027/home/surly/raid4/kendrick-data/nsd/nsddata_betas/ppdata/subj{:02d}/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session{:02d}.nii.gz nsddata_betas/ppdata/subj{:02d}/func1pt8mm/betas_fithrf_GLMdenoise_RR/'.format(sub,sess,sub))

# Download ROIs
for sub in [2,5,7]:
    os.makedirs('nsddata/ppdata/subj{:02d}/func1pt8mm/roi/'.format(sub))
    os.system('cp /home/naxos2-raid25/kneel027/home/kneel027/home/surly/raid4/kendrick-data/nsd/nsddata/ppdata/subj{:02d}/func1pt8mm/roi/* nsddata/ppdata/subj{:02d}/func1pt8mm/roi/'.format(sub,sub))
