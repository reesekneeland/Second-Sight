# Set our directory to work with Ghislains files. 
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
sys.path.append('/export/raid1/home/styvesg/code/nsd_gnet8x/')
import torch
import matplotlib.pyplot as plt
from nsd_access import NSDAccess
from PIL import Image
from src.file_utility import zip_dict
from models.alexnet import Alexnet_fmaps
from src.torch_feature_space import Torch_filter_fmaps
from src.numpy_utility import iterate_range, make_gaussian_mass_stack
from utils import *


# First URL: This is the original read-only NSD file path (The actual data)
# Second URL: Local files that we are adding to the dataset and need to access as part of the data
# Object for the NSDAccess package
nsda = NSDAccess('/export/raid1/home/surly/raid4/kendrick-data/nsd', '/export/raid1/home/kneel027/nsd_local')


def get_value(_x):
    return np.copy(_x.data.cpu().numpy())

def set_value(_x, x):
    if list(x.shape)!=list(_x.size()):
        _x.resize_(x.shape)
    _x.data.copy_(torch.from_numpy(x))
    
def _to_torch(x, device=None):
    return torch.from_numpy(x).float().to(device)        

class Torch_fwRF_voxel_block(nn.Module):
    '''
    This is a variant of the fwRF model as a module for a voxel block (we can't have it all at once)
    '''

    def __init__(self, _fmaps_fn, params, _nonlinearity=None, input_shape=(1,3,227,227), aperture=1.0):
        super(Torch_fwRF_voxel_block, self).__init__()
        
        self.aperture = aperture
        models, weights, bias, mstmt, mstst = params
        device = next(_fmaps_fn.parameters()).device
        _x =torch.empty((1,)+input_shape[1:], device=device).uniform_(0, 1)
        _fmaps = _fmaps_fn(_x)
        self.fmaps_rez = []
        for k,_fm in enumerate(_fmaps):
            assert _fm.size()[2]==_fm.size()[3], 'All feature maps need to be square'
            self.fmaps_rez += [_fm.size()[2],]
        
        self.pfs = []
        for k,n_pix in enumerate(self.fmaps_rez):
            pf = make_gaussian_mass_stack(models[:,0], models[:,1], models[:,2], n_pix, size=aperture, dtype=np.float32)[2]
            self.pfs += [nn.Parameter(torch.from_numpy(pf).to(device), requires_grad=False),]
            self.register_parameter('pf%d'%k, self.pfs[-1])
            
        self.weights = nn.Parameter(torch.from_numpy(weights).to(device), requires_grad=False)
        self.bias = None
        if bias is not None:
            self.bias = nn.Parameter(torch.from_numpy(bias).to(device), requires_grad=False)
            
        self.mstm = None
        self.msts = None
        if mstmt is not None:
            self.mstm = nn.Parameter(torch.from_numpy(mstmt.T).to(device), requires_grad=False)
        if mstst is not None:
            self.msts = nn.Parameter(torch.from_numpy(mstst.T).to(device), requires_grad=False)
        self._nl = _nonlinearity
              
    def load_voxel_block(self, *params):
        models = params[0]
        for _pf,n_pix in zip(self.pfs, self.fmaps_rez):
            pf = make_gaussian_mass_stack(models[:,0], models[:,1], models[:,2], n_pix, size=self.aperture, dtype=np.float32)[2]
            if len(pf)<_pf.size()[0]:
                pp = np.zeros(shape=_pf.size(), dtype=pf.dtype)
                pp[:len(pf)] = pf
                set_value(_pf, pp)
            else:
                set_value(_pf, pf)
        for _p,p in zip([self.weights, self.bias], params[1:3]):
            if _p is not None:
                if len(p)<_p.size()[0]:
                    pp = np.zeros(shape=_p.size(), dtype=p.dtype)
                    pp[:len(p)] = p
                    set_value(_p, pp)
                else:
                    set_value(_p, p)
        for _p,p in zip([self.mstm, self.msts], params[3:]):
            if _p is not None:
                if len(p)<_p.size()[1]:
                    pp = np.zeros(shape=(_p.size()[1], _p.size()[0]), dtype=p.dtype)
                    pp[:len(p)] = p
                    set_value(_p, pp.T)
                else:
                    set_value(_p, p.T)
 
    def forward(self, _fmaps):
        _mst = torch.cat([torch.tensordot(_fm, _pf, dims=[[2,3], [1,2]]) for _fm,_pf in zip(_fmaps, self.pfs)], dim=1) # [#samples, #features, #voxels] 
        if self._nl is not None:
            _mst = self._nl(_mst)
        if self.mstm is not None:              
            _mst -= self.mstm[None]
        if self.msts is not None:
            _mst /= self.msts[None]
        _mst = torch.transpose(torch.transpose(_mst, 0, 2), 1, 2) # [#voxels, #samples, features]
        _r = torch.squeeze(torch.bmm(_mst, torch.unsqueeze(self.weights, 2))).t() # [#samples, #voxels]
        if self.bias is not None:
            _r += torch.unsqueeze(self.bias, 0)
        return _r

    
def get_predictions(data, _fmaps_fn, _fwrf_fn, params, sample_batch_size=100):
    """
    The predictive fwRF model for arbitrary input image.

    Parameters
    ----------
    data : ndarray, shape (#samples, #channels, x, y)
        Input image block.
    _fmaps_fn: Torch module
        Torch module that returns a list of torch tensors.
    _fwrf_fn: Torch module
    Torch module that compute the fwrf model for one batch of voxels
    params: list including all of the following:
    [
        models : ndarray, shape (#voxels, 3)
            The RF model (x, y, sigma) associated with each voxel.
        weights : ndarray, shape (#voxels, #features)
            Tuning weights
        bias: Can contain a bias parameter of shape (#voxels) if add_bias is True.
           Tuning biases: None if there are no bias
        mst_mean (optional): ndarray, shape (#voxels, #feature)
            None if zscore is False. Otherwise returns zscoring average per feature.
        mst_std (optional): ndarray, shape (#voxels, #feature)
            None if zscore is False. Otherwise returns zscoring std.dev. per feature.
    ]
    sample_batch_size (default: 100)
        The sample batch size (used where appropriate)

    Returns
    -------
    pred : ndarray, shape (#samples, #voxels)
        The prediction of voxel activities for each voxels associated with the input data.
    """
    dtype = data.dtype.type
    device = next(_fmaps_fn.parameters()).device
    _params = [_p for _p in _fwrf_fn.parameters()]
    #print("This is the _params: ", _params)
    voxel_batch_size = _params[0].size()[0]    
    # nt is the len of the images 
    # nv is the len of the brain scan
    nt, nv = len(data), len(params[0])
    #print ('val_size = %d' % nt)
    pred = np.full(fill_value=0, shape=(nt, nv), dtype=dtype)
    start_time = time.time()
    with torch.no_grad():
        for rv, lv in iterate_range(0, nv, voxel_batch_size):
            _fwrf_fn.load_voxel_block(*[p[rv] if p is not None else None for p in params])
            pred_block = np.full(fill_value=0, shape=(nt, voxel_batch_size), dtype=dtype)
            for rt, lt in iterate_range(0, nt, sample_batch_size):
                sys.stdout.write('\rsamples [%5d:%-5d] of %d, voxels [%6d:%-6d] of %d' % (rt[0], rt[-1], nt, rv[0], rv[-1], nv))
                pred_block[rt] = get_value(_fwrf_fn(_fmaps_fn(_to_torch(data[rt], device)))) 
            pred[:,rv] = pred_block[:,:lv]
    total_time = time.time() - start_time
    print ('\n---------------------------------------')
    print ('total time = %fs' % total_time)
    print ('sample throughput = %fs/sample' % (total_time / nt))
    print ('voxel throughput = %fs/voxel' % (total_time / nv))
    sys.stdout.flush()
    return pred 


class Alexnet():
    
    def __init__(self,
                predict_normal=False,
                predict_73k=False):
        
        # Input Variables
        self.normal_predict = predict_normal
        self.predict_73K = predict_73k
        
        # Setting up Cuda
        torch.manual_seed(time.time())
        self.device = torch.device("cuda") #cuda
        torch.backends.cudnn.enabled=True
        
        # Config Information
        self.root_dir   = '/export/raid1/home/styvesg/code/nsd_gnet8x/'
        
        # File locations
        self.input_dir = self.root_dir + "output/multisubject/"
        self.model_dir = self.input_dir + 'anet_fwrf_nsdgeneral_Feb-02-2022_1002/' 

        self.stim_dir = self.root_dir+'../../data/nsd/stims/'
        self.voxel_dir = self.root_dir+'../../data/nsd/voxels/'

        self.exp_design_file = self.root_dir+"../../data/nsd/nsd_expdesign.mat"
        self.output_dir = self.model_dir
        
        
        # Reload model files
        self.checkpoint = torch.load(self.model_dir + 'model_params')
        self.model_params = self.checkpoint['best_params']
        self.subjects = list(self.checkpoint['subjects'])
        
        # Voxel Info
        self.brain_nii_shape = self.checkpoint['brain_nii_shape']
        self.nsdcore_val_cc  = self.checkpoint['val_cc']
        self.voxel_mask      = self.checkpoint['voxel_mask']
        self.voxel_idx       = self.checkpoint['voxel_index']
        self.voxel_roi       = self.checkpoint['voxel_roi']   
        
        # Masking information
        mask_path = "masks/"
        self.masks = {0:torch.full((11838,), False),
                      1:torch.load(mask_path + "V1.pt"),
                      2:torch.load(mask_path + "V2.pt"),
                      3:torch.load(mask_path + "V3.pt"),
                      4:torch.load(mask_path + "V4.pt"),
                      5:torch.load(mask_path + "V5.pt"),
                      6:torch.load(mask_path + "V6.pt"),
                      7:torch.load(mask_path + "V7.pt")}
        
        
    def load_data(self):
        
        if(self.predict_normal):
            
            # ONE SUBJECTS 
            # ----------- Load data ------------
            # x_train, x_val, _, _, _, _, _, _, _, _, val_trails, _ = load_nsd(vector = "c_img_0", loader = False, average = True, return_trial = True)
            _, _, _, x_test, _, _, _, _, alexnet_stimuli_order, _, _ = load_nsd(vector = "c_img_0", loader = False, return_trial = True)
    
            print(alexnet_stimuli_order)

            self.val_voxel_data    = {}
            self.val_stim_ordering = {}
            self.val_voxel_data[1] = x_test.numpy()
            self.val_stim_ordering[1] = alexnet_stimuli_order
            print(max(alexnet_stimuli_order))
            print(len(alexnet_stimuli_order))
            print(x_test.shape)


            # ----------- Load Stimuli Subject 1------------
            subj1_train = nsda.stim_descriptions[(nsda.stim_descriptions['subject1'] != 0)]
            print(subj1_train)
            self.image_data = {}
            data = []
            w, h = 227, 227  # resize to integer multiple of 64
            for i in tqdm(range(10000), desc="loading in images"):
                
                nsdId = subj1_train.iloc[i]['nsdId']
                ground_truth_np_array = nsda.read_images([nsdId], show=False)
                ground_truth = Image.fromarray(ground_truth_np_array[0])
                
                imagePil = ground_truth.resize((w, h), resample=Image.Resampling.LANCZOS)
                image = np.array(imagePil).astype(np.float32) / 255.0
                
                data.append(image)
                
            self.image_data[1] = np.moveaxis(np.array(data), 3, 1)
            print(self.image_data.keys())
            print ('block size:', self.image_data[1].shape, ', dtype:', self.image_data[1].dtype, ', value range:',\
                    np.min(self.image_data[1]), np.max(self.image_data[1]))
                
        elif(self.predict_73K):
            
            # ----------- Load Stimuli Whole COCO ------------
            self.image_data = {}
            data = []
            w, h = 227, 227  # resize to integer multiple of 64
            for i in tqdm(range(73000), desc="loading in images"):
                
                ground_truth_np_array = nsda.read_images([i], show=False)
                ground_truth = Image.fromarray(ground_truth_np_array[0])
                
                imagePil = ground_truth.resize((w, h), resample=Image.Resampling.LANCZOS)
                image = np.array(imagePil).astype(np.float32) / 255.0
            
                data.append(image)
                
                
            imgs = []
            subj1 = nsda.stim_descriptions[nsda.stim_descriptions['subject1'] != 0]
            nsdIds = set(subj1['nsdId'].tolist())
            
            # load 73k images in for loop        
            imgs_full = data 
            count = 0
            for pred in tqdm(range(73000), desc="loading in images"):
                 if pred not in nsdIds:
                     imgs.append(imgs_full[pred])
                     count += 1
            
            self.image_data[1] = np.moveaxis(np.array(imgs), 3, 1)
            print(self.image_data.keys())
            print ('block size:', self.image_data[1].shape, ', dtype:', self.image_data[1].dtype, ', value range:',\
                    np.min(self.image_data[1]), np.max(self.image_data[1]))
            
    
    def load_image(self, image_path):
        
        image = Image.open(image_path).convert('RGB')
        
        w, h = 227, 227  # resize to integer multiple of 64
        imagePil = image.resize((w, h), resample=Image.Resampling.LANCZOS)
        image = np.array(imagePil).astype(np.float32) / 255.0
        
        return image
            
        
        
    def predict_normal(self):
        
        voxel_batch_size = 500 # 200
        _log_act_func = lambda _x: torch.log(1 + torch.abs(_x))

        _fmaps_fn = Alexnet_fmaps().to(self.device)
        _fmaps_fn = Torch_filter_fmaps(_fmaps_fn, self.checkpoint['lmask'], self.checkpoint['fmask'])
        _fwrf_fn  = Torch_fwRF_voxel_block(_fmaps_fn, [p[:voxel_batch_size] if p is not None else None for p in self.model_params[self.subjects[0]]], \
                                        _nonlinearity=_log_act_func, input_shape = self.image_data[self.subjects[0]].shape, aperture=1.0)
        
        sample_batch_size = 1000

        subject_image_pred = {}
        for s,bp in self.model_params.items():
            subject_image_pred[1] = get_predictions(self.image_data[1], _fmaps_fn, _fwrf_fn, bp, sample_batch_size=sample_batch_size)
            break
        
        torch.save(torch.from_numpy(subject_image_pred[1]), "latent_vectors/alexnet_encoder/alexnet_pred_subject1_10k.pt")
        
        subject_val_cc = {s: np.zeros(v.shape[1]) for s,v in self.val_voxel_data.items()}
        for s,p,o,v in zip_dict(subject_image_pred, self.val_stim_ordering, self.val_voxel_data):
            for i in range(v.shape[1]):
                subject_val_cc[s][i] = np.corrcoef(p[o, i], v[:,i])[0,1]
                
        plt.figure(figsize=(6,6))
        plt.plot(self.nsdcore_val_cc[1], subject_val_cc[1], marker='.', linestyle='None')
        plt.xlabel('old')
        plt.ylabel('new')
        plt.savefig("charts/alexnet_line.png")
        
        plt.hist(subject_val_cc[1], bins=80, log=True)
        print(np.mean(subject_val_cc[1]))
        plt.savefig("/export/raid1/home/kneel027/Second-Sight/charts/alexnet_hist.png")
        
    def predict_73k_coco(self):
        
        voxel_batch_size = 500 # 200
        _log_act_func = lambda _x: torch.log(1 + torch.abs(_x))

        _fmaps_fn = Alexnet_fmaps().to(self.device)
        _fmaps_fn = Torch_filter_fmaps(_fmaps_fn, self.checkpoint['lmask'], self.checkpoint['fmask'])
        _fwrf_fn  = Torch_fwRF_voxel_block(_fmaps_fn, [p[:voxel_batch_size] if p is not None else None for p in self.model_params[self.subjects[0]]], \
                                        _nonlinearity=_log_act_func, input_shape=self.image_data[self.subjects[0]].shape, aperture=1.0)
        
        sample_batch_size = 1000

        subject_image_pred = {}
        for s,bp in self.model_params.items():
            subject_image_pred[1] = get_predictions(self.image_data[1], _fmaps_fn, _fwrf_fn, bp, sample_batch_size=sample_batch_size)
            break
        
        print(subject_image_pred[1].shape)
        
        torch.save(torch.from_numpy(subject_image_pred[1]), "/export/raid1/home/kneel027/Second-Sight/latent_vectors/alexnet_encoder/alexnet_pred_73k.pt")
        
    #def calulate_predict(self):
        
        
    def predict_cc3m(self):
        
        rootdir = "/export/raid1/home/kneel027/nsd_local/cc3m/cc3m/"
        folder_list = []
        
        for it in os.scandir(rootdir):
            if it.is_dir():
                folder_list.append(it.name)
            
        data = []
        image_data = {}
        count = 0
        batch_count = 0
        while count < 2819140: 
            for folder in folder_list:
                if(folder == "_tmp"):
                    pass
                for file in tqdm(sorted(os.scandir(rootdir + folder), key=lambda e: e.name)):
                    try:
                        if(file.name.endswith(".jpg")):
                            image = self.load_image(rootdir + folder + "/" + file.name)
                            data.append(image)  
                            count += 1
                            if((count != 0) and (count % 22735 == 0)):
                                print(count)
                                print(np.array(data).shape)
                                image_data[1] = np.moveaxis(np.array(data), 3, 1)
                                print ('block size:', image_data[1].shape, ', dtype:', image_data[1].dtype, ', value range:',\
                                        np.min(image_data[1]), np.max(image_data[1])) 
                                    
                                voxel_batch_size = 500
                                _log_act_func = lambda _x: torch.log(1 + torch.abs(_x))

                                _fmaps_fn = Alexnet_fmaps().to(self.device)
                                _fmaps_fn = Torch_filter_fmaps(_fmaps_fn, self.checkpoint['lmask'], self.checkpoint['fmask'])
                                _fwrf_fn  = Torch_fwRF_voxel_block(_fmaps_fn, [p[:voxel_batch_size] if p is not None else None for p in self.model_params[self.subjects[0]]], \
                                                                _nonlinearity=_log_act_func, input_shape=image_data[self.subjects[0]].shape, aperture=1.0)
                                
                                sample_batch_size = 1000

                                subject_image_pred = {}
                                for s,bp in self.model_params.items():
                                    subject_image_pred[1] = get_predictions(image_data[1], _fmaps_fn, _fwrf_fn, bp, sample_batch_size=sample_batch_size)
                                    break
                                
                                torch.save(torch.from_numpy(subject_image_pred[1]), "latent_vectors/alexnet_encoder/alexnet_pred_cc3m_batches/" + str(batch_count) + ".pt")
                                batch_count += 1
                                data = []
                                image_data = {}
                        else:
                            pass
                    except:
                        print(file.name)
                        
                        
                        
    def predict(self, images, mask = []):
        
        self.image_data = {}
        data = []
        w, h = 227, 227  # resize to integer multiple of 64
        for i in range(len(images)):
            
            imagePil = images[i].resize((w, h), resample=Image.Resampling.LANCZOS)
            image = np.array(imagePil).astype(np.float32) / 255.0
            
            data.append(image)
            
        self.image_data[1] = np.moveaxis(np.array(data), 3, 1)
        
        beta_mask = self.masks[0]
        for i in mask:
            beta_mask = torch.logical_or(beta_mask, self.masks[i])        
        
        voxel_batch_size = 200 # 200
        _log_act_func = lambda _x: torch.log(1 + torch.abs(_x)) 

        _fmaps_fn = Alexnet_fmaps().to(self.device)
        _fmaps_fn = Torch_filter_fmaps(_fmaps_fn, self.checkpoint['lmask'], self.checkpoint['fmask'])
        _fwrf_fn  = Torch_fwRF_voxel_block(_fmaps_fn, [p[:voxel_batch_size] if p is not None else None for p in self.model_params[self.subjects[0]]], \
                                        _nonlinearity=_log_act_func, input_shape=self.image_data[self.subjects[0]].shape, aperture=1.0)
        
        sample_batch_size = 1000

        subject_image_pred = {}
        for s,bp in self.model_params.items():
            if(len(mask) == 0):
                subject_image_pred[1] = get_predictions(self.image_data[1], _fmaps_fn, _fwrf_fn, bp, sample_batch_size=sample_batch_size)
                break
            
            else:
                masked_params = []
                for params in bp:
                    masked_params.append(params[beta_mask])
                
                subject_image_pred[1] = get_predictions(self.image_data[1], _fmaps_fn, _fwrf_fn, masked_params, sample_batch_size=sample_batch_size)
                break
        
        print(subject_image_pred[1].shape)
        return torch.from_numpy(subject_image_pred[1])
      

def main():
    
    AN = Alexnet(predict_normal = False, predict_73k = False)
    
    subj1_train = nsda.stim_descriptions[(nsda.stim_descriptions['subject1'] != 0)]
    data = []
    for i in tqdm(range(100), desc="loading in images"):
        
        nsdId = subj1_train.iloc[i]['nsdId']
        ground_truth_np_array = nsda.read_images([nsdId], show=False)
        ground_truth = Image.fromarray(ground_truth_np_array[0])
        data.append(ground_truth)
    
    #AN.predict_cc3m()
    
    #AN.load_data()
    #AN.predict_normal()
    
    AN.predict(data, [1])
           
        
if __name__ == "__main__":
    main()
        
        