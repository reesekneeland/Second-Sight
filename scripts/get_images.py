import os
import numpy as np
from PIL import Image
from nsd_access import NSDAccess
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm

nsda = NSDAccess('/export/raid1/home/surly/raid4/kendrick-data/nsd', '/export/raid1/home/kneel027/nsd_local')
subj1_test = nsda.stim_descriptions[(nsda.stim_descriptions['subject1'] != 0) & (nsda.stim_descriptions['shared1000'] == True)]
for i in tqdm(range(1000)):
    nsdId = subj1_test.iloc[i]['nsdId']
    im_arr = nsda.read_images([nsdId], show=False)
    im = Image.fromarray(im_arr[0])
    im.save("logs/shared1000_images/" + str(i) + ".png")


