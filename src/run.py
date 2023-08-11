import sys
import argparse
sys.path.append('data')
from download_nsddata import download_nsd
from download_weights import download_weights
from setup_data import process_images, process_trial_data

sys.path.append('src')
from autoencoder import AutoEncoder
from clip_encoder import CLIPEncoder
from generate_beta_primes import generate_beta_primes

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("-d",
                        "--downloadnsd",
                        help="flag to download the NSD data via AWS, requires aws command to be configued. If not set, assumes that the required NSD files are present under the data/ directory",
                        action='store_true')

    parser.add_argument("-t",
                        "--train",
                        help="flag to train models from scratch, will only download the gnet model, and train all others from scratch. If flag is not set it will download all pretrained models.",
                        action='store_true')
    
    parser.add_argument('-s',
                        '--subjects', 
                        help="list of subjects to run the algorithm on, if not specified, will run on all subjects",
                        type=str,
                        default="1,2,5,7")

    parser.add_argument('--device', 
                        help="cuda device to run predicts on.",
                        type=str,
                        default="cuda:0")

    args = parser.parse_args()
    subject_list = [int(sub) for sub in args.subjects.split(",")]
    if args.downloadnsd:
        download_nsd(subjects = subject_list)
    
    download_weights(subjects = subject_list, only_gnet=args.train)

    process_images(device = args.device)

    process_trial_data(subjects = subject_list)

    if args.train:
        for sub in subject_list:
            encModel = CLIPEncoder(subject=sub, device=args.device, log=False)
            encModel.train()
            encModel.benchmark(average=False)
            encModel.benchmark(average=True)

    generate_beta_primes(subjects=subject_list, device=args.device)
    
    if args.train:
        for sub in subject_list:
            for config in ["hybrid", "clip", "gnet"]:
                autoEncModel = AutoEncoder(config=config, subject=sub, device=args.device, log=False)
                autoEncModel.train()
                autoEncModel.benchmark(encodedPass=False, average=False)
                autoEncModel.benchmark(encodedPass=False, average=True)


