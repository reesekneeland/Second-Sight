import os, sys
# os.environ["PATH"] = os.environ["PATH"]+":/usr/local/cuda/bin/"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import torch
import numpy as np
from PIL import Image
sys.path.append('src')
from utils import *
from pearson import PearsonCorrCoef
from PIL import Image
import random
from nsd_access import NSDAccess
from tqdm import tqdm
import time
from sklearn.decomposition import PCA, IncrementalPCA
import seaborn as sns
import matplotlib.pyplot as plt
import pickle as pk

# import pycuda.autoinit
# import pycuda.gpuarray as gpuarray
# import numpy as np
# import skcuda.linalg as linalg
# from skcuda.linalg import PCA as cuPCA

# dataloader = load_nsd(vector = "c_img_vd", loader = True, split = False, batch_size=13875)
x, y = load_nsd(vector = "c_img_vd", loader = False, split = False)
x_pca, y_pca = load_nsd(vector = "c_img_vd", loader = False, split = False, pca=True)

pca = PCA(n_components=15000)
# pca.fit(y)
# pk.dump(pca, open("masks/pca_c_img_vd_15k.pkl","wb"))
pca = pk.load(open("masks/pca_c_img_vd_15k.pkl",'rb'))
c = torch.from_numpy(pca.components_)
m = torch.from_numpy(pca.mean_)
# for i, (x,y) in enumerate(tqdm(dataloader)):
    # pca.partial_fit(y)

# Computed mean per feature
# mean = pca.mean_
# and stddev
# stddev = np.sqrt(pca.var_)

# Ytransformed = None
# for i, (x,y) in enumerate(tqdm(dataloader)):
#     Ychunk = sklearn_pca.transform(y)
#     if Ytransformed == None:
#         Ytransformed = Ychunk
#     else:
#         Ytransformed = np.vstack((Ytransformed, Ychunk))
PeC = PearsonCorrCoef(num_outputs=1)
testy = y[0:100]
print(c.shape, testy.shape)
start = time.time()
# reducedY = y_pca[0:100]
reducedY = pca.transform(testy)
print(reducedY.shape)
mid = time.time()
print(mid - start)
unscaledY = (reducedY @ c) + m
# unscaledY = pca.inverse_transform(reducedY)
print(unscaledY.shape)
end = time.time()
print(end - start)
testy = testy.T
unscaledY = unscaledY.T.to(torch.float32)
print(testy.dtype, unscaledY.dtype)
print(testy == unscaledY)



l = pca.explained_variance_ratio_
print(c.shape, np.sum(l))

# pk.dump(pca, open("masks/pca_c_text_vd_10k.pkl","wb"))


# g = sns.lineplot(data=l)
# g.set(xlabel='Principle Components', ylabel='Ratio of Variance Explained', title='PCA Analaysis of c_img_vd 27k')
# plt.savefig("charts/c_img_vd_PCA_analysis_ratio27k.png")