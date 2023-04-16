import sys
from utils import *
sys.path.append('vdvae')
import torch
from torch.optim import Adam
import torchvision.transforms as T
import torch.nn as nn
from torchmetrics import PearsonCorrCoef
import numpy as np
from image_utils import *
from model_utils import *
from PIL import Image
import wandb
from tqdm import tqdm

prep_path = "/export/raid1/home/kneel027/nsd_local/preprocessed_data/"
latent_path = "/export/raid1/home/kneel027/Second-Sight/latent_vectors/"
H = {'image_size': 64, 'image_channels': 3,'seed': 0, 'port': 29500, 'save_dir': './saved_models/test', 'data_root': './', 'desc': 'test', 'hparam_sets': 'imagenet64', 'restore_path': 'imagenet64-iter-1600000-model.th', 'restore_ema_path': 'vdvae/model/imagenet64-iter-1600000-model-ema.th', 'restore_log_path': 'imagenet64-iter-1600000-log.jsonl', 'restore_optimizer_path': 'imagenet64-iter-1600000-opt.th', 'dataset': 'imagenet64', 'ema_rate': 0.999, 'enc_blocks': '64x11,64d2,32x20,32d2,16x9,16d2,8x8,8d2,4x7,4d4,1x5', 'dec_blocks': '1x2,4m1,4x3,8m4,8x7,16m8,16x15,32m16,32x31,64m32,64x12', 'zdim': 16, 'width': 512, 'custom_width_str': '', 'bottleneck_multiple': 0.25, 'no_bias_above': 64, 'scale_encblock': False, 'test_eval': True, 'warmup_iters': 100, 'num_mixtures': 10, 'grad_clip': 220.0, 'skip_threshold': 380.0, 'lr': 0.00015, 'lr_prior': 0.00015, 'wd': 0.01, 'wd_prior': 0.0, 'num_epochs': 10000, 'n_batch': 4, 'adam_beta1': 0.9, 'adam_beta2': 0.9, 'temperature': 1.0, 'iters_per_ckpt': 25000, 'iters_per_print': 1000, 'iters_per_save': 10000, 'iters_per_images': 10000, 'epochs_per_eval': 1, 'epochs_per_probe': None, 'epochs_per_eval_save': 1, 'num_images_visualize': 8, 'num_variables_visualize': 6, 'num_temperatures_visualize': 3, 'mpi_size': 1, 'local_rank': 0, 'rank': 0, 'logdir': './saved_models/test/log'}
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
H = dotdict(H)

H, preprocess_fn = set_up_data(H)

print('Models is Loading')
ema_vae = load_vaes(H)
  
class batch_generator_external_images(Dataset):

    def __init__(self, data_path):
        self.data_path = data_path
        self.im = np.load(data_path).astype(np.uint8)


    def __getitem__(self,idx):
        img = Image.fromarray(self.im[idx])
        img = T.functional.resize(img,(64,64))
        img = torch.tensor(np.array(img)).float()
        #img = img/255
        #img = img*2 - 1
        return img

    def __len__(self):
        return  len(self.im)






# Pytorch model class for Linear regression layer Neural Network
class MLP(torch.nn.Module):
    def __init__(self, vector):
        super(MLP, self,).__init__()
        self.vector = vector
        if(self.vector == "c_img_uc"):
            # self.linear = nn.Linear(11838, 1024)
            self.linear = nn.Linear(11838, 15000)
            self.linear2 = nn.Linear(15000, 15000)
            self.linear3 = nn.Linear(15000, 15000)
            self.linear4 = nn.Linear(15000, 15000)
            self.outlayer = nn.Linear(15000, 1024)
        if(self.vector == "c_text_uc"):
            self.linear = nn.Linear(11838, 78848)
            # self.linear = nn.Linear(11838, 15000)
            # self.linear2 = nn.Linear(15000, 15000)
            # self.outlayer = nn.Linear(15000, 78848)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        if(self.vector == "c_img_uc"):
            # y_pred = self.linear(x)
            y_pred = self.relu(self.linear(x))
            y_pred = self.relu(self.linear2(y_pred))
            y_pred = self.relu(self.linear3(y_pred))
            y_pred = self.relu(self.linear4(y_pred))
            y_pred = self.outlayer(y_pred)
        if(self.vector=="c_text_uc"):
            y_pred = self.linear(x)
            # y_pred = self.relu(self.linear(x))
            # y_pred = self.relu(self.linear2(y_pred))
            # y_pred = self.outlayer(y_pred)
        return y_pred

    
# Main Class    
class Decoder_VDVAE():
    def __init__(self, 
                 hashNum,
                 vector, 
                 log, 
                 lr=0.00001,
                 batch_size=750,
                 device="cuda",
                 num_workers=4,
                 epochs=200
                 ):

        # Set the parameters for pytorch model training
        self.hashNum = hashNum
        self.vector = vector
        self.device = torch.device(device)
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = epochs
        self.num_workers = num_workers
        self.log = log
        # Initialize the Pytorch model class
        self.model = MLP(self.vector)

        # Send model to Pytorch Device 
        self.model.to(self.device)
        
        # Initialize the data loaders
        self.trainLoader, self.valLoader, self.testLoader = None, None, None
        
        # Initializes Weights and Biases to keep track of experiments and training runs
        if(self.log):
            wandb.init(
                # set the wandb project where this run will be logged
                project="decoder_uc",
                # track hyperparameters and run metadata
                config={
                "hash": self.hashNum,
                "architecture": "MLP unCLIP",
                "vector": self.vector,
                "dataset": "Z scored",
                "epochs": self.num_epochs,
                "learning_rate": self.lr,
                "batch_size:": self.batch_size,
                "num_workers": self.num_workers
                }
            )
    

    def train(self):
        self.trainLoader, self.valLoader, _ = load_nsd(vector=self.vector, 
                                                        batch_size=self.batch_size, 
                                                        num_workers=self.num_workers, 
                                                        loader=True)
        # Set best loss to negative value so it always gets overwritten
        best_loss = -1.0
        loss_counter = 0
        
        # Configure the pytorch objects, loss function (criterion)
        criterion = nn.MSELoss(reduction='sum')
        # criterion = nn.CrossEntropyLoss()
        # Set the optimizer to Adam
        optimizer = Adam(self.model.parameters(), lr = self.lr)
        # optimizer = bnb.optim.Adam8bit(self.model.parameters(), lr = self.lr)
        # Begin training, iterates through epochs, and through the whole dataset for every epoch
        for epoch in tqdm(range(self.num_epochs), desc="epochs"):
            # For each epoch, do a training and a validation stage
            # Entering training stage
            self.model.train()
            # Keep track of running loss for this training epoch
            running_loss = 0.0
            for i, data in enumerate(self.trainLoader):
                # x_data = Brain Data
                # y_data = Clip/Z vector Data
                x_data, y_data = data
                # Moving the tensors to the GPU
                x_data = x_data.to(self.device)
                y_data = y_data.to(self.device)
                # Forward pass: Compute predicted y by passing x to the model
                # Train the x data in the model to get the predicted y value. 
                pred_y = self.model(x_data).to(self.device)
                # scaled_pred_y /= torch.norm(y_data)
                
                # Compute the loss between the predicted y and the y data. 
                loss = criterion(pred_y, y_data)
                loss.backward()
                optimizer.step()
                # Zero gradients in the optimizer
                optimizer.zero_grad()
                # Add up the loss for this training round
                running_loss += loss.item()
            tqdm.write('[%d] train loss: %.8f' %
                (epoch + 1, running_loss /len(self.trainLoader.dataset)))
                #     # wandb.log({'epoch': epoch+1, 'loss': running_loss/(50 * self.batch_size)})
            
                
            # Entering validation stage
            # Set model to evaluation mode
            self.model.eval()
            running_test_loss = 0.0
            for i, data in enumerate(self.valLoader):
                
                # Loading in the test data
                x_data, y_data = data
                x_data = x_data.to(self.device)
                y_data = y_data.to(self.device)
                # Forward pass: Compute predicted y by passing x to the model
                # Train the x data in the model to get the predicted y value. 
                pred_y = self.model(x_data).to(self.device)
                # scaled_pred_y /= torch.norm(y_data)
                
                # Compute the loss between the predicted y and the y data. 
                loss = criterion(pred_y, y_data)
                running_test_loss += loss.item()
            test_loss = running_test_loss / len(self.valLoader.dataset)
                
            # Printing and logging loss so we can keep track of progress
            tqdm.write('[%d] test loss: %.8f' %
                        (epoch + 1, test_loss))
            if(self.log):
                wandb.log({'epoch': epoch+1, 'test_loss': test_loss})
                    
            # Check if we need to save the model
            # Early stopping
            if(best_loss == -1.0 or test_loss < best_loss):
                best_loss = test_loss
                torch.save(self.model.state_dict(), "models/{hash}_model_{vec}.pt".format(hash=self.hashNum, vec=self.vector))
                loss_counter = 0
            else:
                loss_counter += 1
                tqdm.write("loss counter: " + str(loss_counter))
                if(loss_counter >= 5):
                    break
                
        

    def predict(self, x):
        self.model.load_state_dict(torch.load("models/{hash}_model_{vec}.pt".format(hash=self.hashNum, vec=self.vector), map_location=self.device))
        self.model.eval()
        self.model.to(self.device)
        out = self.model(x.to(self.device)).to(torch.float16)
        return out
    
    def benchmark(self, average=True):
        _, _, x_test, _, _, y_test, _ = load_nsd(vector=self.vector, 
                                                loader=False,
                                                average=average)
        # Load our best model into the class to be used for predictions
        self.model.load_state_dict(torch.load("models/{hash}_model_{vec}.pt".format(hash=self.hashNum, vec=self.vector)))
        self.model.eval()

        criterion = nn.MSELoss()
        PeC = PearsonCorrCoef(num_outputs=y_test.shape[0]).to(self.device)
        
        x_test = x_test.to(self.device)
        y_test = y_test.to(self.device)
        
        pred_y = self.model(x_test)
        
        pearson = torch.mean(PeC(pred_y.moveaxis(0,1), y_test.moveaxis(0,1)))
        loss = criterion(pred_y, y_test)

        
        
        print("Vector Correlation: ", float(pearson))
        print("Loss: ", float(loss))






    
image_path = prep_path + "subject{:02d}/nsd_train_stim_sub{}.npy'.format(sub,sub)
train_images = batch_generator_external_images(data_path = image_path)

image_path = 'data/processed_data/subj{:02d}/nsd_test_stim_sub{}.npy'.format(sub,sub)
test_images = batch_generator_external_images(data_path = image_path)

trainloader = DataLoader(train_images,batch_size,shuffle=False)
testloader = DataLoader(test_images,batch_size,shuffle=False)


num_latents = 31
test_latents = []
for i,x in enumerate(testloader):
  data_input, target = preprocess_fn(x)
  with torch.no_grad():
        print(i*batch_size)
        activations = ema_vae.encoder.forward(data_input)
        px_z, stats = ema_vae.decoder.forward(activations, get_latents=True)
        #recons = ema_vae.decoder.out_net.sample(px_z)
        batch_latent = []
        for i in range(num_latents):
            #test_latents[i].append(stats[i]['z'].cpu().numpy())
            batch_latent.append(stats[i]['z'].cpu().numpy().reshape(len(data_input),-1))
        test_latents.append(np.hstack(batch_latent))

test_latents = np.concatenate(test_latents)  

train_latents = []
for i,x in enumerate(trainloader):
  data_input, target = preprocess_fn(x)
  with torch.no_grad():
        print(i*batch_size)
        activations = ema_vae.encoder.forward(data_input)
        px_z, stats = ema_vae.decoder.forward(activations, get_latents=True)
        #recons = ema_vae.decoder.out_net.sample(px_z)
        batch_latent = []
        for i in range(num_latents):
            batch_latent.append(stats[i]['z'].cpu().numpy().reshape(len(data_input),-1))
        train_latents.append(np.hstack(batch_latent))
train_latents = np.concatenate(train_latents)      

np.savez("data/extracted_features/subj{:02d}/nsd_vdvae_features_31l.npz".format(sub),train_latents=train_latents,test_latents=test_latents)

