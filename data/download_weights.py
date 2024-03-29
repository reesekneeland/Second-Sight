import wget
import os
import argparse

def download_weights(subjects, only_gnet=False):
    os.makedirs("models", exist_ok=True)
    
    file_ids = []
    destinations = []
    
    file_ids.append("https://huggingface.co/reesekneeland/Second-Sight/resolve/main/gnet_multisubject")
    destinations.append("models/gnet_multisubject")
    
    #VDVAE Models
    file_ids.append("https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets-2/imagenet64-iter-1600000-log.jsonl")
    destinations.append("vdvae/model/imagenet64-iter-1600000-log.jsonl")
    
    file_ids.append("https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets-2/imagenet64-iter-1600000-model.th")
    destinations.append("vdvae/model/imagenet64-iter-1600000-model.th")
    
    file_ids.append("https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets-2/imagenet64-iter-1600000-model-ema.th")
    destinations.append("vdvae/model/imagenet64-iter-1600000-model-ema.th")
    
    file_ids.append("https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets-2/imagenet64-iter-1600000-opt.th")
    destinations.append("vdvae/model/imagenet64-iter-1600000-opt.th")
    
    
    
    if not only_gnet:
        if(1 in subjects):
            file_ids.append("https://huggingface.co/reesekneeland/Second-Sight/resolve/main/sub01_dual_autoencoder.pt")
            destinations.append("models/sub01_hybrid_autoencoder.pt")
            
            file_ids.append("https://huggingface.co/reesekneeland/Second-Sight/resolve/main/sub01_clip_encoder.pt")
            destinations.append("models/sub01_clip_encoder.pt")
            
            file_ids.append("https://huggingface.co/reesekneeland/Second-Sight/resolve/main/sub01_clip_autoencoder.pt")
            destinations.append("models/sub01_clip_autoencoder.pt")
            
            file_ids.append("https://huggingface.co/reesekneeland/Second-Sight/resolve/main/sub01_gnet_autoencoder.pt")
            destinations.append("models/sub01_gnet_autoencoder.pt")
            
        if(2 in subjects):
            file_ids.append("https://huggingface.co/reesekneeland/Second-Sight/resolve/main/sub02_dual_autoencoder.pt")
            destinations.append("models/sub02_hybrid_autoencoder.pt")
            
            file_ids.append("https://huggingface.co/reesekneeland/Second-Sight/resolve/main/sub02_clip_encoder.pt")
            destinations.append("models/sub02_clip_encoder.pt")
            
            file_ids.append("https://huggingface.co/reesekneeland/Second-Sight/resolve/main/sub02_clip_autoencoder.pt")
            destinations.append("models/sub02_clip_autoencoder.pt")
            
            file_ids.append("https://huggingface.co/reesekneeland/Second-Sight/resolve/main/sub02_gnet_autoencoder.pt")
            destinations.append("models/sub02_gnet_autoencoder.pt")
            
        if(5 in subjects):
            file_ids.append("https://huggingface.co/reesekneeland/Second-Sight/resolve/main/sub05_dual_autoencoder.pt")
            destinations.append("models/sub05_hybrid_autoencoder.pt")
            
            file_ids.append("https://huggingface.co/reesekneeland/Second-Sight/resolve/main/sub05_clip_encoder.pt")
            destinations.append("models/sub05_clip_encoder.pt")
            
            file_ids.append("https://huggingface.co/reesekneeland/Second-Sight/resolve/main/sub05_clip_autoencoder.pt")
            destinations.append("models/sub05_clip_autoencoder.pt")
            
            file_ids.append("https://huggingface.co/reesekneeland/Second-Sight/resolve/main/sub05_gnet_autoencoder.pt")
            destinations.append("models/sub05_gnet_autoencoder.pt")
                
        if(7 in subjects):
            file_ids.append("https://huggingface.co/reesekneeland/Second-Sight/resolve/main/sub07_dual_autoencoder.pt")
            destinations.append("models/sub07_hybrid_autoencoder.pt")
            
            file_ids.append("https://huggingface.co/reesekneeland/Second-Sight/resolve/main/sub07_clip_encoder.pt")
            destinations.append("models/sub07_clip_encoder.pt")
            
            file_ids.append("https://huggingface.co/reesekneeland/Second-Sight/resolve/main/sub07_clip_autoencoder.pt")
            destinations.append("models/sub07_clip_autoencoder.pt")
            
            file_ids.append("https://huggingface.co/reesekneeland/Second-Sight/resolve/main/sub07_gnet_autoencoder.pt")
            destinations.append("models/sub07_gnet_autoencoder.pt")
            
    for file_id, dest in zip(file_ids, destinations):
        print("\nDownloading {}".format(file_id))
        wget.download(file_id, out=dest)
    print("\nSuccessfully downloaded model weights!")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument(
                        '--gnet', 
                        help="flag to download only the GNet models and the VDVAE models, but you will be required to train the other models, otherwise all models will be downloaded.",
                        action='store_true')
    
    parser.add_argument('-s',
                        '--subjects', 
                        help="list of subjects to download models for, if not specified, will run on all subjects",
                        type=str,
                        default="1,2,5,7")
    args = parser.parse_args()
    subject_list = [int(sub) for sub in args.subjects.strip().split(",")]
    download_weights(subject_list, args.gnet)