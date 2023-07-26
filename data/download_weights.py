import sys
import requests
import zipfile
import os
import argparse
from tqdm import tqdm

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download&confirm=1"

    session = requests.Session()

    response = session.get(URL, params={"id": id}, stream=True)
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
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                progress_bar.update(len(chunk))


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--gnet', 
        help="flag to download only the GNet model (1.18GB) but you will be required to train the other models, otherwise all models will be downloaded (13.4GB)",
        action='store_true')
    args = parser.parse_args()
    
    os.makedirs("models", exist_ok=True)
    if not args.gnet:
        file_id = "TAKE_ID_FROM_SHAREABLE_LINK"
        destination = "models/models.zip"
    else:
        file_id = "1pmlLRCMkKdowZp9w14JtT5gK8FAvfSIa"
        destination = "models/gnet_multisubject"
    print(f"Downloading model weights...")
    download_file_from_google_drive(file_id, destination)
    print(f"Successfully downloaded model weights!")
    if not args.gnet:
        print("Unzipping model weights...")
        with zipfile.ZipFile(destination, 'r') as zip_ref:
            zip_ref.extractall(destination)

