import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
sys.path.append('/home/naxos2-raid25/ojeda040/local/styvesg/code/nsd_gnet8x/')

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:95% !important; }</style>"))

import src.numpy_utility as pnu
from src.file_utility import save_stuff, flatten_dict, embed_dict, zip_dict
from src.load_nsd import ordering_split
from config import *
import torch
import matplotlib.pyplot as plt

use_device = "cuda:1"
print ('#device:', torch.cuda.device_count())
print ('device#:', torch.cuda.current_device())
print ('device name:', torch.cuda.get_device_name(torch.cuda.current_device()))

torch.manual_seed(time.time())
device = torch.device(use_device) #cuda
torch.backends.cudnn.enabled=True

print ('\ntorch:', torch.__version__)
print ('cuda: ', torch.version.cuda)
print ('cudnn:', torch.backends.cudnn.version())
print ('dtype:', torch.get_default_dtype())
#torch.set_default_dtype(torch.float64)


# File Locations

input_dir = root_dir + "output/multisubject/"
model_dir = input_dir + 'anet_fwrf_nsdgeneral_Feb-02-2022_1002/' 

stim_dir = root_dir+'../../data/nsd/stims/'
voxel_dir = root_dir+'../../data/nsd/voxels/'

exp_design_file = root_dir+"../../data/nsd/nsd_expdesign.mat"
#exp_design_file = root_dir+"../../data/nsd/nsdsynthetic_expdesign.mat"

output_dir = model_dir

# Reload Model Files
checkpoint = torch.load(model_dir + 'model_params')
print (checkpoint.keys())
model_params = checkpoint['best_params']
subjects = list(checkpoint['subjects'])
print ([p.shape for p in checkpoint['best_params'][3]])


# Load Stimuli
exp_design = loadmat(exp_design_file)
ordering = exp_design['masterordering'].flatten() - 1 # zero-indexed ordering of indices (matlab-like to python-like)

from src.load_nsd import image_feature_fn

image_data = {}
for s in subjects: 
    image_data_set = h5py.File(stim_dir + "S%d_stimuli_227.h5py"%s, 'r')
    image_data[s] = image_feature_fn(np.copy(image_data_set['stimuli']))
    image_data_set.close()
    print ('--------  subject %d  -------' % s)
    print ('block size:', image_data[s].shape, ', dtype:', image_data[s].dtype, ', value range:',\
           np.min(image_data[s]), np.max(image_data[s]))
    
    
n = 1000
plt.figure(figsize=(6,2*len(subjects)))
for k,s in enumerate(subjects): 
    for i in range(3):
        plt.subplot(len(subjects), 3, 3*k+i+1)
        plt.imshow(image_data[s][n+i].transpose((1,2,0)), cmap='gray', interpolation='None')
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)
        
        
        
# Voxel Mask
from src.file_utility import load_mask_from_nii, view_data
from src.roi import roi_map, iterate_roi

group_names = ['V1', 'V2', 'V3', 'hV4', 'V3ab', 'LO', 'IPS', 'VO', 'PHC', 'MT', 'MST', 'other']
group = [[1,2],[3,4],[5,6], [7], [16, 17], [14, 15], [18,19,20,21,22,23], [8, 9], [10,11], [13], [12], [24,25,0]]

brain_nii_shape = checkpoint['brain_nii_shape']
nsdcore_val_cc  = checkpoint['val_cc']
voxel_mask      = checkpoint['voxel_mask']
voxel_idx       = checkpoint['voxel_index']
voxel_roi       = checkpoint['voxel_roi']


# Load NSD voxel data

#voxel_data_set = h5py.File(root_dir+'voxel_synth_data_V1-4_part1.h5py', 'r')
voxel_data_set = h5py.File(voxel_dir+'voxel_data_nsdgeneral_part1.h5py', 'r')
#voxel_data_set = h5py.File(voxel_dir+'voxel_data_V1_4_part1.h5py', 'r')
voxel_data_dict = embed_dict({k: np.copy(d) for k,d in voxel_data_set.items()})
voxel_data_set.close()
voxel_data = voxel_data_dict['voxel_data']

#voxel_data_set = h5py.File(root_dir+'voxel_synth_data_V1-4_part2.h5py', 'r')
voxel_data_set = h5py.File(voxel_dir+'voxel_data_nsdgeneral_part2.h5py', 'r')
#voxel_data_set = h5py.File(voxel_dir+'voxel_data_V1_4_part2.h5py', 'r')
voxel_data_dict = embed_dict({k: np.copy(d) for k,d in voxel_data_set.items()})
voxel_data_set.close()
voxel_data.update(voxel_data_dict['voxel_data'])

voxel_data = {int(s): voxel_data[s] for s in voxel_data.keys()}
print (voxel_data.keys())

# Ordering
from src.load_nsd import ordering_split
trn_stim_ordering, val_stim_ordering, val_voxel_data = {},{},{}

data_size, nnv = {}, {}
for k,s in enumerate(voxel_data.keys()):
    print ('--------  subject %d  -------' % s)
    data_size[s], nnv[s] = voxel_data[s].shape 

    trn_stim_ordering[s], _, val_stim_ordering[s], val_voxel_data[s] = \
        ordering_split(voxel_data[s], ordering, combine_trial=False)
        

# Rebuild Model
from imp import reload
import src.torch_fwrf as aaa
reload(aaa)
from src.torch_fwrf import get_predictions

from models.alexnet import Alexnet_fmaps
from src.torch_feature_space import Torch_filter_fmaps
from src.torch_fwrf import Torch_fwRF_voxel_block


voxel_batch_size = 500 # 200
#_log_act_func = lambda _x: torch.log(1 + torch.abs(_x))*torch.tanh(torch.abs(_x))
_log_act_func = lambda _x: torch.log(1 + torch.abs(_x))

_fmaps_fn = Alexnet_fmaps().to(device)
_fmaps_fn = Torch_filter_fmaps(_fmaps_fn, checkpoint['lmask'], checkpoint['fmask'])
_fwrf_fn  = Torch_fwRF_voxel_block(_fmaps_fn, [p[:voxel_batch_size] if p is not None else None for p in model_params[subjects[0]]], \
                                   _nonlinearity=_log_act_func, input_shape=image_data[subjects[0]].shape, aperture=1.0)


sample_batch_size = 1000

subject_image_pred = {}
for s,bp in model_params.items():
    subject_image_pred[s] = get_predictions(image_data[s], _fmaps_fn, _fwrf_fn, bp, sample_batch_size=sample_batch_size)
    
    
# Validation Accuracy
subject_val_cc = {s: np.zeros(v.shape[1]) for s,v in val_voxel_data.items()}
for s,p,o,v in zip_dict(subject_image_pred, val_stim_ordering, val_voxel_data):
    for i in range(v.shape[1]):
        subject_val_cc[s][i] = np.corrcoef(p[o,i], v[:,i])[0,1]
        
        
plt.figure(figsize=(6,6))
plt.plot(nsdcore_val_cc[1], subject_val_cc[1], marker='.', linestyle='None')
plt.xlabel('old')
plt.ylabel('new')
plt.savefig("alexnet.png")


# Save Predictions
# pred_data = {'pred': subject_image_pred,
#             'val_cc': subject_val_cc,
#             'voxel_roi': voxel_roi,
#             'voxel_mask': voxel_mask,
#             'voxel_index': voxel_idx,
#             'brain_nii_shape': brain_nii_shape ,
#            }

# save_stuff( output_dir + 'nsd_prediction_all', flatten_dict(pred_data))


