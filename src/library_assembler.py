import torch
import numpy as np
import torch.nn as nn
from utils import *
import yaml
from tqdm import tqdm
from autoencoder  import AutoEncoder
from torchmetrics import PearsonCorrCoef

class LibraryAssembler():
    def __init__(self, 
                 configList=["gnet"],
                 subject=1,
                 ae=True,
                 device="cuda",
                 mask=None,
                 ):
        
        with torch.no_grad():
            self.subject=subject
            subject_sizes = [0, 15724, 14278, 0, 0, 13039, 0, 12682]
            self.x_size = subject_sizes[self.subject]
            self.AEModels = []
            self.configList = configList
            self.device = device
            if mask is not None:
                self.mask = mask
            else:
                self.mask = torch.full((self.x_size,), True)
            self.ae = ae
            self.datasize = {"c": 1024, "images": 541875, "z_vdvae": 91168}

            self.y_indices = get_pruned_indices(subject=self.subject)

            for param in self.configList:
                if param == "gnet":
                    if self.ae:
                        self.AEModels.append(AutoEncoder(config="gnet",
                                                        inference=True,
                                                        subject=self.subject,
                                                        device=self.device))
                elif param == "clip":
                    if self.ae:
                        self.AEModels.append(AutoEncoder(config="clip",
                                                        inference=True,
                                                        subject=self.subject,
                                                        device=self.device))
            self.x_preds = []
            for model in self.configList:
                modelPreds = torch.load("data/preprocessed_data/subject{}/{}_coco_beta_primes.pt".format(self.subject, model), map_location=self.device)
                prunedPreds = modelPreds[self.y_indices]
                self.x_preds.append(prunedPreds)
                
    def rankCoco(self, x, average=True, topn=100):
        with torch.no_grad():
            if not average:
                x = x[0]
            
            PeC = PearsonCorrCoef(num_outputs=21000).to(self.device)
            
            scores = torch.zeros((self.y_indices.shape[0],))
            div = 0
            for AEmodel, preds in zip(self.AEModels, self.x_preds):
                for rep in range(x.shape[0]):
                    x_rep = x[rep]
                    if(torch.count_nonzero(x_rep) > 0):
                        if self.ae:
                            x_rep = AEmodel.predict(x_rep)
                        xDup = x_rep[self.mask].repeat(21000, 1).moveaxis(0, 1)
                        for coco_batch in range(3):
                            x_preds_t = []
                            x_preds_batch = preds[21000*coco_batch:21000*coco_batch+21000, self.mask]
                            x_preds_t = x_preds_batch.moveaxis(0, 1).to(self.device)
                            modelScore = PeC(xDup, x_preds_t).cpu()
                            
                            scores[21000*coco_batch:21000*coco_batch+21000] += modelScore
                
                    div +=1
            scores /= div
            topn_pearson = torch.topk(scores, topn)
        return topn_pearson.indices

    def predict(self, x, vector="images", topn=1):
        with torch.no_grad():
            library = torch.load("data/preprocessed_data/{}_73k.pt".format(vector)).reshape((73000, self.datasize[vector]))[self.y_indices]
            if vector == "images" and topn > 1:
                x_preds = torch.zeros((x.shape[0], topn, self.datasize[vector])).cpu()
            else:
                x_preds = torch.zeros((x.shape[0], self.datasize[vector])).cpu()
            # Get best match for each sample
            for sample in tqdm(range(x.shape[0]), desc="predicting samples for {}".format(vector)):
                best_im_indices = self.rankCoco(x[sample], topn=topn)
                if vector == "images":
                    x_preds[sample] = library[best_im_indices.tolist()].reshape((topn, self.datasize[vector]))
                else:
                    vectors = library[best_im_indices.tolist()].reshape((topn, self.datasize[vector]))
                    x_preds[sample] = torch.mean(vectors, dim=0)
        return x_preds

    
    def benchmark(self, vector="c"):

        # Load data and targets
        _, _, x_test, _, _, y_test, _ = load_nsd(vector=vector, subject=self.subject, loader=False, average=False, nest=True)
        
        if vector == "c":
            topn = 100
        elif vector == "z_vdvae":
            topn=25
        elif vector == "images":
            topn=1
        
        
        
        criterion = nn.MSELoss()
        
        PeC = PearsonCorrCoef(num_outputs=y_test.shape[0]).to(self.device)
        
        x_test = x_test.to(self.device)
        y_test = y_test.to(self.device)
        
        pred_y = self.predict(x=x_test, vector=vector, topn=topn).to(self.device)
        pearson = torch.mean(PeC(pred_y.moveaxis(0,1), y_test.moveaxis(0,1)))
        loss = criterion(pred_y, y_test)

        
        print("Library Assembler Benchmark, Subject: {}".format(self.subject))
        print("Vector Correlation: ", float(pearson))
        print("Loss: ", float(loss))

if __name__ == "__main__":
    device = "cuda:2"
    subject=1
    
    LD = LibraryAssembler(configList=["gnet", "clip"],
                        subject=subject,
                        ae=True,
                        device=device)
    LD.benchmark(vector="c")
    # LD.benchmark(vector="images")
    LD_v = LibraryAssembler(configList=["gnet"],
                        subject=subject,
                        ae=True,
                        mask=torch.load("data/subject{}/masks/early_vis.pt".format(subject)),
                        device=device)
    LD_v.benchmark(vector="z_vdvae")