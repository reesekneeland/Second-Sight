# Only GPU's in use
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
import numpy as np
import matplotlib.pyplot as plt
import rcca
from torch.nn import MSELoss
from sklearn import datasets
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score
from torchmetrics.functional import pearson_corrcoef
from utils import *
import wandb
from tqdm import tqdm

class CCADecoder():
    def __init__(self, 
                 hashNum,
                 vector, 
                 log, 
                 threshold,
                 device="cuda",
                 n_alphas=20
                 ):

        # Set the parameters for pytorch model training
        self.hashNum = hashNum
        self.vector = vector
        self.threshold = threshold
        self.device = torch.device(device)
        self.log = log
        self.model = rcca.CCA(kernelcca = False, reg = 0., numCC = np.arange(3, 11))
        self.n_alphas = n_alphas
        # Initialize the data
        self.x_train, self.x_test, self.y_train, self.y_test = get_data(vector=self.vector, threshold=self.threshold, loader=False)

        # Initializes Weights and Biases to keep track of experiments and training runs
        if(self.log):
            wandb.init(
                # set the wandb project where this run will be logged
                project="CCA Decoder",
                # track hyperparameters and run metadata
                config={
                "hash": self.hashNum,
                "threshold": self.threshold,
                "architecture": "Canonical Correlation Analysis",
                # "architecture": "2 Convolutional Layers",
                "vector": self.vector,
                "dataset": "custom masked positive pearson correlation on vector data"
                }
            )
    def train(self):
        regs = regs = np.array(np.logspace(-4, 2, self.n_alphas))
        print("fitting")
        self.model.train([self.x_train, self.y_train])
        print("predicting")
        pred_frr, _ = self.model.predict(self.x_test, self.model.ws)
        np.save("CCA_pred_clip.npy", pred_frr)
        # pred_frr = np.load("CCA_pred_brain.npy")
        print("scoring")
        # frr_r2 = r2_score(x_test, pred_frr)
        y_pred = torch.from_numpy(pred_frr)
        criterion = MSELoss(size_average = False)
        loss = criterion(y_pred, self.y_test)
        if(self.log):
            wandb.log({'test_loss': loss})
        os.makedirs("latent_vectors/" + self.hashNum + "_CCA_" + self.vector, exist_ok=True)
        torch.save(y_pred, "/export/raid1/home/kneel027/Second-Sight/latent_vectors/" + self.hashNum + "_CCA_" + self.vector + "/" + "y_test_preds.pt")
        return self.hashNum, y_pred, self.y_test
    
    def predict(self, hashNum, indices=[i for i in range(2250)]):
        if(not os.path.isdir("latent_vectors/" + hashNum + "_CCA_" + self.vector)):
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), hashNum + "_CCA_" + self.vector)
        preds = torch.load("/export/raid1/home/kneel027/Second-Sight/latent_vectors/" + self.hashNum + "_CCA_" + self.vector + "/" + "y_test_preds.pt").to(torch.float32)
        outputs, targets = [], []
        for i in indices:
            # Loading in the test data
            x_data = self.x_test[i]
            y_data = self.y_test[i]
            pred_y = preds[i]
            outputs.append(pred_y)
            targets.append(y_data)
            torch.save(pred_y, "/export/raid1/home/kneel027/Second-Sight/latent_vectors/" + self.hashNum + "_CCA_" + self.vector +  "/" + "output_" + str(i) + "_" + self.vector + ".pt")
            torch.save(y_data, "/export/raid1/home/kneel027/Second-Sight/latent_vectors/" + self.hashNum + "_CCA_" + self.vector +  "/" + "target_" + str(i) + "_" + self.vector + ".pt")
        return outputs, targets