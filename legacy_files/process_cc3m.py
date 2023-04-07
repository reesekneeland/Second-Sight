import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
import torch
import numpy as np
from PIL import Image
sys.path.append('../src')
# from reconstructor import Reconstructor

from PIL import Image
import random
from nsd_access import NSDAccess
from tqdm import tqdm
import time

rootdir = "/home/naxos2-raid25/kneel027/home/kneel027/nsd_local/cc3m/cc3m/"
folder_list = []
# R = Reconstructor()
start = time.time()
for it in os.scandir(rootdir):
    if it.is_dir():
        folder_list.append(it.name)
print(folder_list)
count = 0
batch = 0
images = torch.zeros((22735,541875))
# Encoding vectors for 2819140 images
for folder in tqdm(folder_list):
    if(folder == "_tmp"):
        pass
    print(folder)
    for file in tqdm(sorted(os.scandir(rootdir + folder), key=lambda e: e.name)):
        if file.name.endswith(".jpg"):
            print(count, file.name)
            image = Image.open(rootdir + folder + "/" + file.name).convert('RGB')
            image.save("/home/naxos2-raid25/kneel027/home/kneel027/nsd_local/cc3m/cc3m/images/" + str(count) + ".jpg")
            im_array = torch.from_numpy(np.array(image)).reshape((1,541875))
            images[count % 22735] = im_array
            count +=1
            if(count % 22735 == 0):
                torch.save(images, "/export/raid1/home/kneel027/nsd_local/preprocessed_data/images/cc3m_batches/" + str(batch) + ".pt")
                images = torch.zeros((22735,541875))
                batch+=1
            
        else:
            pass
end = time.time()



print(count, start-end)