import os
import argparse
#os.system('ls -l')

def download_nsd(subjects):
    # Download Experiment Infos
    os.system('aws s3 cp s3://natural-scenes-dataset/nsddata/experiments/nsd/nsd_expdesign.mat data/nsddata/experiments/nsd/')
    os.system('aws s3 cp s3://natural-scenes-dataset/nsddata/experiments/nsd/nsd_stim_info_merged.csv data/nsddata/experiments/nsd/')

    # Download Stimuli
    os.system('aws s3 cp s3://natural-scenes-dataset/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5 data/nsddata_stimuli/stimuli/nsd/')

    # Download Betas
    for sub in subjects:
        for sess in range(1,38):
            os.system('aws s3 cp s3://natural-scenes-dataset/nsddata_betas/ppdata/subj{:02d}/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session{:02d}.nii.gz data/nsddata_betas/ppdata/subj{:02d}/func1pt8mm/betas_fithrf_GLMdenoise_RR/'.format(sub,sess,sub))

    # Download ROIs
    for sub in subjects:
        os.system('aws s3 cp s3://natural-scenes-dataset/nsddata/ppdata/subj{:02d}/func1pt8mm/roi/* data/nsddata/ppdata/subj{:02d}/func1pt8mm/roi/'.format(sub,sub))

if __name__ == '__main__':
    # Create the parser and add arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-s',
                        '--subjects', 
                        help="list of subjects to download models for, if not specified, will run on all subjects",
                        type=str,
                        default="1,2,5,7")
    args = parser.parse_args()
    subject_list = [int(sub) for sub in args.subjects.strip().split(",")]
    download_nsd(subject_list)