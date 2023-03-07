
# Set our directory to work with Ghislains files. 
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
from nsd_access import NSDAccess
from PIL import Image
from src.file_utility import load_mask_from_nii, view_data
from src.roi import roi_map, iterate_roi
from imp import reload
import torch_fwrf as aaa
reload(aaa)
from torch_fwrf import get_predictions
from models.alexnet import Alexnet_fmaps
from src.torch_feature_space import Torch_filter_fmaps
from src.torch_fwrf import Torch_fwRF_voxel_block
from utils import *


# First URL: This is the original read-only NSD file path (The actual data)
# Second URL: Local files that we are adding to the dataset and need to access as part of the data
# Object for the NSDAccess package
nsda = NSDAccess('/home/naxos2-raid25/kneel027/home/surly/raid4/kendrick-data/nsd', '/home/naxos2-raid25/kneel027/home/kneel027/nsd_local')


class Alexnet():
    
    def __init__(self,
                predict_normal=False,
                predict_73k=False):
        
        # Input Variables
        self.normal_predict = predict_normal
        self.predict_73K = predict_73k
        
        # Setting up Cuda
        torch.manual_seed(time.time())
        self.device = torch.device("cuda:1") #cuda
        torch.backends.cudnn.enabled=True
        
        # File locations
        self.input_dir = root_dir + "output/multisubject/"
        self.model_dir = self.input_dir + 'anet_fwrf_nsdgeneral_Feb-02-2022_1002/' 

        self.stim_dir = root_dir+'../../data/nsd/stims/'
        self.voxel_dir = root_dir+'../../data/nsd/voxels/'

        self.exp_design_file = root_dir+"../../data/nsd/nsd_expdesign.mat"
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
        mask_path = "/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/masks/"
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
        
        torch.save(torch.from_numpy(subject_image_pred[1]), "/export/raid1/home/kneel027/Second-Sight/latent_vectors/alexnet_encoder/alexnet_pred_subject1_10k.pt")
        
        subject_val_cc = {s: np.zeros(v.shape[1]) for s,v in self.val_voxel_data.items()}
        for s,p,o,v in zip_dict(subject_image_pred, self.val_stim_ordering, self.val_voxel_data):
            for i in range(v.shape[1]):
                subject_val_cc[s][i] = np.corrcoef(p[o, i], v[:,i])[0,1]
                
        plt.figure(figsize=(6,6))
        plt.plot(self.nsdcore_val_cc[1], subject_val_cc[1], marker='.', linestyle='None')
        plt.xlabel('old')
        plt.ylabel('new')
        plt.savefig("/export/raid1/home/kneel027/Second-Sight/charts/alexnet_line.png")
        
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
        
        rootdir = "/home/naxos2-raid25/kneel027/home/kneel027/nsd_local/cc3m/cc3m/"
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
                                
                                torch.save(torch.from_numpy(subject_image_pred[1]), "/export/raid1/home/kneel027/Second-Sight/latent_vectors/alexnet_encoder/alexnet_pred_cc3m_batches/" + str(batch_count) + ".pt")
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
    
    AN = Alexnet(predict_normal = True, predict_73k = False)
    
    # subj1_train = nsda.stim_descriptions[(nsda.stim_descriptions['subject1'] != 0)]
    # data = []
    # for i in tqdm(range(100), desc="loading in images"):
        
    #     nsdId = subj1_train.iloc[i]['nsdId']
    #     ground_truth_np_array = nsda.read_images([nsdId], show=False)
    #     ground_truth = Image.fromarray(ground_truth_np_array[0])
    #     data.append(ground_truth)
    
    #AN.predict_cc3m()
    
    AN.load_data()
    AN.predict_normal()
    
    # AN.predict(data, [1])
           
        
if __name__ == "__main__":
    main()
        
        