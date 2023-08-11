import torch
import torch as T
import matplotlib.pyplot as plt
from torch import nn
from PIL import Image
from utils import *
from autoencoder import AutoEncoder
import time
from torchmetrics import PearsonCorrCoef
from model_zoo import Encoder, Torch_LayerwiseFWRF


def subject_pred_pass(_pred_fn, _ext, _con, x, batch_size):
    pred = _pred_fn(_ext, _con, x[:batch_size]) # this is just to get the shape
    pred = np.zeros(shape=(len(x), pred.shape[1]), dtype=np.float32) # allocate
    for rb,_ in iterate_range(0, len(x), batch_size):
        pred[rb] = get_value(_pred_fn(_ext, _con, x[rb]))
    return pred



def gnet8j_predictions(image_data, _pred_fn, trunk_width, pass_through, checkpoint, mask, batch_size, device=torch.device("cuda:0")):
    
    subjects = list(image_data.keys())

    if(mask is None):
        subject_nv = {s: len(v) for s,v in checkpoint['val_cc'].items()} 
    else:
        subject_nv = {s: len(v) for s,v in checkpoint['val_cc'].items()}    
        subject_nv[subjects[0]] = int(torch.sum(mask == True)) 

    # allocate
    subject_image_pred = {s: np.zeros(shape=(len(image_data[s]), subject_nv[s]), dtype=np.float32) for s in subjects}
    # print(subject_image_pred)
    _log_act_fn = lambda _x: T.log(1 + T.abs(_x))*T.tanh(_x)
     
    best_params = checkpoint['best_params']
    # print(best_params)
    shared_model = Encoder(np.array(checkpoint['input_mean']).astype(np.float32), trunk_width=trunk_width, pass_through=pass_through).to(device)
    shared_model.load_state_dict(best_params['enc'])
    shared_model.eval() 

    # example fmaps
    rec, fmaps, h = shared_model(T.from_numpy(image_data[list(image_data.keys())[0]][:20]).to(device))                                     
    for s in subjects:
        sd = Torch_LayerwiseFWRF(fmaps, nv=subject_nv[s], pre_nl=_log_act_fn, post_nl=_log_act_fn, dtype=np.float32).to(device) 
        params = best_params['fwrfs'][s]
        
        if(mask is None):
            sd.load_state_dict(params)
        
        else:
            masked_params = {}
            for key, value in params.items():
                masked_params[key] = value[mask]
                
            sd.load_state_dict(masked_params)
            
        # print(params['w'].shape)
        # print(params['b'].shape)
        # sd.load_state_dict(best_params['fwrfs'][s])
        sd.eval() 
        # print(sd)
        
        subject_image_pred[s] = subject_pred_pass(_pred_fn, shared_model, sd, image_data[s], batch_size)

    return subject_image_pred




#######################################


class GNet8_Encoder():
    
    def __init__(self, subject = 1, device = "cuda"):
        
        # Setting up Cuda
        torch.manual_seed(time.time())
        self.device = torch.device(device)
        torch.backends.cudnn.enabled=True
        # Subject number
        self.subject = subject
        
        # Vector type
        self.vector = "images"
        
        # x size
        subject_sizes = [0, 15724, 14278, 0, 0, 13039, 0, 12682]
        self.x_size = subject_sizes[self.subject]
        
        # Reload joined GNet model files
        self.joined_checkpoint = torch.load('models/gnet_multisubject', map_location=self.device)
        
        self.subjects = list(self.joined_checkpoint['voxel_mask'].keys())
        self.gnet8j_voxel_mask = self.joined_checkpoint['voxel_mask']
        self.gnet8j_voxel_roi  = self.joined_checkpoint['voxel_roi']
        self.gnet8j_voxel_index= self.joined_checkpoint['voxel_index']
        self.gnet8j_brain_nii_shape= self.joined_checkpoint['brain_nii_shape']
        self.gnet8j_val_cc = self.joined_checkpoint['val_cc']
        
            
    
    def load_image(self, image_path):
        
        image = Image.open(image_path).convert('RGB')
        
        w, h = 227, 227  # resize to integer multiple of 64
        imagePil = image.resize((w, h), resample=Image.Resampling.LANCZOS)
        image = np.array(imagePil).astype(np.float32) / 255.0
        
        return image  
    
    # Rebuild Model
    def _model_fn(self, _ext, _con, _x):
        '''model consists of an extractor (_ext) and a connection model (_con)'''
        _y, _fm, _h = _ext(_x)
        return _con(_fm)

    def _pred_fn(self, _ext, _con, xb):
        return self._model_fn(_ext, _con, torch.from_numpy(xb).to(self.device))  
                    

    def predict(self, images, mask = None):
        self.stim_data = {}
        data = []
        w, h = 227, 227  # resize to integer multiple of 64
        
        if(isinstance(images, list)):
            for i in range(len(images)):
                
                imagePil = images[i].resize((w, h), resample=Image.Resampling.LANCZOS)
                image = np.array(imagePil).astype(np.float32) / 255.0
                data.append(image)
            
        elif(isinstance(images, torch.Tensor)):
            for i in range(images.shape[0]):
                
                imagePil = process_image(images[i], w, h)
                image = np.array(imagePil).astype(np.float32) / 255.0
                data.append(image)
            
        
        self.stim_data[self.subject] = np.moveaxis(np.array(data), 3, 1)

        gnet8j_image_pred = gnet8j_predictions(self.stim_data, self._pred_fn, 64, 192, self.joined_checkpoint, mask, batch_size=100, device=self.device)

        return torch.from_numpy(gnet8j_image_pred[self.subject])
      
    def benchmark(self, average=True, ae=False):
        _, _, y_test, _, _, test_images, _ = load_nsd(vector="images",
                                                      subject=self.subject,  
                                                      loader=False,
                                                      average=average,
                                                      big=True)
        if(ae):
            AE = AutoEncoder(config="gnet",
                             inference=True,
                             subject=self.subject,
                             device=self.device)
            y_test = AE.predict(y_test)
            
        # Load our best model into the class to be used for predictions
        images = []
        for im in test_images:
            images.append(process_image(im))
        criterion = nn.MSELoss()
        PeC = PearsonCorrCoef(num_outputs=y_test.shape[0]).to(self.device)
        
        y_test = y_test.to(self.device)
        
        pred_y = self.predict(images).to(self.device)
        pearson = torch.mean(PeC(pred_y.moveaxis(0,1), y_test.moveaxis(0,1)))
        loss = criterion(pred_y, y_test)
        
        pred_y = pred_y.detach()
        y_test = y_test.detach()
        PeC = PearsonCorrCoef().to(self.device)
        r = []
        for voxel in range(pred_y.shape[1]):
            
            # Correlation across voxels for a sample (Taking a column)
            r.append(PeC(pred_y[:,voxel], y_test[:,voxel]).cpu())
        r = np.array(r)
        
        print("Model ID: {}, Subject: {}, Averaged: {}, AutoEncoded: {}".format(self.hashNum, self.subject, average, ae))
        print("Vector Correlation: ", float(pearson))
        print("Mean Pearson: ", np.mean(r))
        print("Loss: ", float(loss))
        
        plt.hist(r, bins=50, log=True)
        plt.savefig("charts/gnet_encoder_voxel_PeC.png")
        
    def score_voxels(self, average=True):
            
        _, _, y_test, _, _, test_images, _ = load_nsd(vector="images",
                                                      subject=self.subject,  
                                                      loader=False,
                                                      average=average,
                                                      big=True)
        
        print(len(y_test), len(y_test)*0.2)
        test_images = test_images[0:int((len(y_test)*0.2))].to(self.device)
        y_test = y_test[0:int((len(y_test)*0.2))]
        # Load our best model into the class to be used for predictions
        images = []
        for im in test_images:
            images.append(process_image(im))
        
        PeC = PearsonCorrCoef(num_outputs=y_test.shape[0]).to(self.device)
        
        y_test = y_test.to(self.device)
        
        pred_y = self.predict(images).to(self.device)

        PeC = PearsonCorrCoef(num_outputs=y_test.shape[0]).to(self.device)

        pred_y = pred_y.cpu().detach()
        y_test = y_test.cpu().detach()
        PeC = PearsonCorrCoef()
        r = []
        for voxel in range(pred_y.shape[1]):
            # Correlation across voxels for a sample (Taking a column)
            r.append(PeC(pred_y[:,voxel], y_test[:,voxel]))
        r = torch.stack(r)
        print(r.shape, r)
        torch.save(r, "masks/subject{}/{}_{}_encoder_voxel_PeC.pt".format(self.subject, self.hashNum, self.vector))
        
        modelId = "{hash}_model_{vec}.pt".format(hash=self.hashNum, vec=self.vector)
        print("Scoring Voxels, Model ID: {}, Subject: {}, Averaged: {}".format(modelId, self.subject, average))
        print("Mean Pearson: {}".format(torch.mean(r)))
        
def main():
    
    GN = GNet8_Encoder(subject=1, device="cuda:3")
    
    GN.benchmark(average=False, ae=False)
    GN.benchmark(average=True, ae=False)

        
if __name__ == "__main__":
    main()