import requests
import zipfile
import os
import argparse
from tqdm import tqdm
import gdown

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download&confirm=1"

    session = requests.Session()

    response = session.get(URL, headers = {'User-agent': 'Weight Downloader'}, params={"id": id}, stream=True)
    # print(response.status_code)
    # print(response.headers["Retry-After"])
    token = get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    total_size_in_bytes= int(response.headers.get('content-length', 0))
    filename = destination.split("/")[1]
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc="Downloading {}".format(filename))
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                progress_bar.update(len(chunk))


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--gnet', 
        help="flag to download only the GNet model (1.18GB) but you will be required to train the other models, otherwise all models will be downloaded (14GB)",
        action='store_true')
    args = parser.parse_args()
    
    parser.add_argument('-s',
                        '--subjects', 
                        help="list of subjects to download models for, if not specified, will run on all subjects",
                        type=list,
                        default=[1, 2, 5, 7])
    args = parser.parse_args()
    os.makedirs("models", exist_ok=True)
    
    print(f"Downloading model weights...")
    file_ids = []
    destinations = []
    
    file_ids.append("1pmlLRCMkKdowZp9w14JtT5gK8FAvfSIa")
    destinations.append("models/gnet_multisubject")
    
    if not args.gnet:
        if(1 in args.subjects):
            file_ids.append("101NjeMMFurhvwanowzaWb8VPtbGkpqO1")
            destinations.append("models/sub01_dual_autoencoder.pt")
            
            file_ids.append("1AZSEQF88mJixENK4i3bWu33Y6RjvrObe")
            destinations.append("models/sub01_clip_encoder.pt")
            
            file_ids.append("15g80G3uMpUiXlYI8YiocvK4-s9ADmULi")
            destinations.append("models/sub01_clip_autoencoder.pt")
            
            file_ids.append("1M8BD4jQsYQPr5fldChzgyxSjOWA6DDi9")
            destinations.append("models/sub01_gnet_autoencoder.pt")
            
        if(2 in args.subjects):
            file_ids.append("1pbcpZprmnPVZl6nhsx8HD7p79_sLqjwG")
            destinations.append("models/sub02_clip_autoencoder.pt")
            
            file_ids.append("1wjnpLMTj4LPqEOL-HIA8Y2hyCUA3chO6")
            destinations.append("models/sub02_clip_encoder.pt")
            
            file_ids.append("1DZXCm3ShnydDQHlyl82VDYm22LW6aJq9")
            destinations.append("models/sub02_dual_autoencoder.pt")
            
            file_ids.append("1XGecZhEzuYUeS-Y3M20R2tiphxWJGPgv")
            destinations.append("models/sub02_gnet_autoencoder.pt")
            
        if(5 in args.subjects):
            file_ids.append("1kxBJaEbYx6zcMJpKGDmFotYEAjw5E5VH")
            destinations.append("models/sub05_clip_autoencoder.pt")
            
            file_ids.append("1OBs-y58NJziS07MsZ-vtCeVWJtScUAJ8")
            destinations.append("models/sub05_clip_encoder.pt")
            
            file_ids.append("1Ghst4D0U_ShHc7TuGx6Zr-bWQQWRmf-Y")
            destinations.append("models/sub05_dual_autoencoder.pt")
            
            file_ids.append("1ZzY49uMg07DsozbAOPlS0i3gzwaesQUE")
            destinations.append("models/sub05_gnet_autoencoder.pt")
                
        if(7 in args.subjects):
            file_ids.append("182jQ_-rfYozemc0eqkwjK4ZfJYEbG9OE")
            destinations.append("models/sub05_clip_autoencoder.pt")
            
            file_ids.append("1Lh43aaI9vgObKrfk1os2U5OGXL4PskMU")
            destinations.append("models/sub05_clip_encoder.pt")
            
            file_ids.append("1cLFS1GYOPfBXhXTvREcSSNVLWH-LoOzV")
            destinations.append("models/sub05_dual_autoencoder.pt")
            
            file_ids.append("11HDCebrHr9EFyeVneCZP_v2pmGoCqvI5")
            destinations.append("models/sub05_gnet_autoencoder.pt")
            
    for file_id, dest in zip(file_ids, destinations):
        # download_file_from_google_drive(file_id, dest)
        gdown.download(id=file_id, output=dest)
    print(f"Successfully downloaded model weights!")
