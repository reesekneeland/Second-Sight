# Only GPU's in use
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
import numpy as np
import matplotlib.pyplot as plt
import rcca
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score
from torchmetrics.functional import pearson_corrcoef
sys.path.append('../src')
from utils import *
from tqdm import tqdm


# Loads the preprocessed data
hashNum = update_hash()
vector = "c_combined"
prep_path = "/export/raid1/home/kneel027/nsd_local/preprocessed_data/"
x = torch.load(prep_path + "x/whole_region_11838_old_norm.pt").requires_grad_(False)
y  = torch.load(prep_path + vector + "/vector.pt").requires_grad_(False)
print(x.shape, y.shape)
x_train = x[:25500]
x_test = x[25500:27750]
y_train = y[:25500]
y_test = y[25500:27750]
        
n_alphas = 20
regs = np.linspace(1/n_alphas, 1 + 1/n_alphas, n_alphas)
# model = rcca.CCA(kernelcca = False, reg = 1., numCC = 4)
model = rcca.CCACrossValidate(regs = regs, numCCs = np.arange(3,11))
print("fitting")
model.train([y_train, x_train])
print("predicting")
corrs = model.validate([y_test, x_test])
preds = model.preds
ws = model.ws
print("ws shape", len(ws))
for i in range(len(ws)):
    print(ws[i].shape)
print("preds shape", len(preds))
preds = np.array(preds[1])
print("preds shape 2", preds.shape)
np.save("CCA_pred_brain.npy", preds)
print("scoring")
frr_r2 = r2_score(x_test, preds)
# out = torch.from_numpy(np.load("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/CCA_pred_clip.npy"))
out = torch.from_numpy(preds)
target = x_test
r=[]
print(out.shape, target.shape)
for p in range(out.shape[1]):
    r.append(pearson_corrcoef(out[:,p], target[:,p]))
r = np.array(r)
print(np.mean(r))
threshold = round((min(r) * -1), 6)
print(threshold)
mask = np.array(len(r) * [True])
threshmask = np.where(np.array(r) > threshold, mask, False)
print(threshmask.sum())
np.save("/export/raid1/home/kneel027/Second-Sight/masks/" + hashNum + "_" + vector + "2voxels_pearson_thresh" + str(threshold), threshmask)
    
#print(r)
#r = np.log(r)
plt.hist(r, bins=40, log=True)
#plt.yscale('log')
plt.savefig("/export/raid1/home/kneel027/Second-Sight/charts/" + hashNum + "_" + vector + "2voxels_pearson_histogram_log_applied_CCA.png")