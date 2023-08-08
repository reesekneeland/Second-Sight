import wget
import os
import argparse

def download_weights(subjects, only_gnet=False):
    os.makedirs("models", exist_ok=True)
    
    file_ids = []
    destinations = []
    
    file_ids.append("https://huggingface.co/reesekneeland/Second-Sight/resolve/main/gnet_multisubject")
    destinations.append("models_old/gnet_multisubject")
    
    if not only_gnet:
        if(1 in subjects):
            file_ids.append("https://huggingface.co/reesekneeland/Second-Sight/resolve/main/sub01_dual_autoencoder.pt")
            destinations.append("models_old/sub01_dual_autoencoder.pt")
            
            file_ids.append("https://huggingface.co/reesekneeland/Second-Sight/resolve/main/sub01_clip_encoder.pt")
            destinations.append("models_old/sub01_clip_encoder.pt")
            
            file_ids.append("https://huggingface.co/reesekneeland/Second-Sight/resolve/main/sub01_clip_autoencoder.pt")
            destinations.append("models_old/sub01_clip_autoencoder.pt")
            
            file_ids.append("https://huggingface.co/reesekneeland/Second-Sight/resolve/main/sub01_gnet_autoencoder.pt")
            destinations.append("models_old/sub01_gnet_autoencoder.pt")
            
        if(2 in subjects):
            file_ids.append("https://huggingface.co/reesekneeland/Second-Sight/resolve/main/sub02_dual_autoencoder.pt")
            destinations.append("models_old/sub02_dual_autoencoder.pt")
            
            file_ids.append("https://huggingface.co/reesekneeland/Second-Sight/resolve/main/sub02_clip_encoder.pt")
            destinations.append("models_old/sub02_clip_encoder.pt")
            
            file_ids.append("https://huggingface.co/reesekneeland/Second-Sight/resolve/main/sub02_clip_autoencoder.pt")
            destinations.append("models_old/sub02_clip_autoencoder.pt")
            
            file_ids.append("https://huggingface.co/reesekneeland/Second-Sight/resolve/main/sub02_gnet_autoencoder.pt")
            destinations.append("models_old/sub02_gnet_autoencoder.pt")
            
        if(5 in subjects):
            file_ids.append("https://huggingface.co/reesekneeland/Second-Sight/resolve/main/sub05_dual_autoencoder.pt")
            destinations.append("models_old/sub05_dual_autoencoder.pt")
            
            file_ids.append("https://huggingface.co/reesekneeland/Second-Sight/resolve/main/sub05_clip_encoder.pt")
            destinations.append("models_old/sub05_clip_encoder.pt")
            
            file_ids.append("https://huggingface.co/reesekneeland/Second-Sight/resolve/main/sub05_clip_autoencoder.pt")
            destinations.append("models_old/sub05_clip_autoencoder.pt")
            
            file_ids.append("https://huggingface.co/reesekneeland/Second-Sight/resolve/main/sub05_gnet_autoencoder.pt")
            destinations.append("models_old/sub05_gnet_autoencoder.pt")
                
        if(7 in subjects):
            file_ids.append("https://huggingface.co/reesekneeland/Second-Sight/resolve/main/sub07_dual_autoencoder.pt")
            destinations.append("models_old/sub07_dual_autoencoder.pt")
            
            file_ids.append("https://huggingface.co/reesekneeland/Second-Sight/resolve/main/sub07_clip_encoder.pt")
            destinations.append("models_old/sub07_clip_encoder.pt")
            
            file_ids.append("https://huggingface.co/reesekneeland/Second-Sight/resolve/main/sub07_clip_autoencoder.pt")
            destinations.append("models_old/sub07_clip_autoencoder.pt")
            
            file_ids.append("https://huggingface.co/reesekneeland/Second-Sight/resolve/main/sub07_gnet_autoencoder.pt")
            destinations.append("models_old/sub07_gnet_autoencoder.pt")
            
    for file_id, dest in zip(file_ids, destinations):
        print("\nDownloading {}".format(file_id))
        wget.download(file_id, out=dest)
    print("\nSuccessfully downloaded model weights!")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
                        '--gnet', 
                        help="flag to download only the GNet model (1.18GB) but you will be required to train the other models, otherwise all models will be downloaded (14GB)",
                        action='store_true')
    
    parser.add_argument('-s',
                        '--subjects', 
                        help="list of subjects to download models for, if not specified, will run on all subjects",
                        type=list,
                        default=[1, 2, 5, 7])
    args = parser.parse_args()
    download_weights(args.subjects, args.gnet)