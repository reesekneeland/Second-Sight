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
                 configList=["gnetEncoder"],
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
            self.EncModels = []
            self.configList = configList
            self.device = device
            if mask is not None:
                self.mask = mask
            else:
                self.mask = torch.full((self.x_size,), True)
            self.ae = ae
            self.datasize = {"c_i": 1024, "images": 541875, "z_vdvae": 91168}

            self.y_indices = get_pruned_indices(subject=self.subject)

            for param in self.configList:
                if param == "gnetEncoder":
                    self.EncModels.append("gnet_encoder")
                    if self.ae:
                        self.AEModels.append(AutoEncoder(config="gnet",
                                                        inference=True,
                                                        subject=self.subject,
                                                        device=self.device,
                                                        big=self.big))
                elif param == "clipEncoder":
                    self.EncModels.append("clip_encoder")
                    if self.ae:
                        self.AEModels.append(AutoEncoder(config="clip",
                                                        inference=True,
                                                        subject=self.subject,
                                                        device=self.device,
                                                        big=self.big))
            self.x_preds = []
            for model in self.EncModels:
                modelPreds = torch.load("data/preprocessed_data/subject{}/{}_beta_primes.pt".format(self.subject, model), map_location=self.device)
                self.x_preds.append(modelPreds)
                
    def rankCoco(self, x, average=True, topn=1000):
        with torch.no_grad():
            if not average:
                x = x[0]
            out = torch.zeros((topn, ))
            ret_scores = torch.zeros((x.shape[0], topn))
            
            PeC = PearsonCorrCoef(num_outputs=21000).to(self.device)
            average_pearson = 0
            
            scores = torch.zeros((self.y_indices.shape[0],))
            div = 0
            for mId, model in enumerate(self.EncModels):
                for rep in range(x.shape[0]):
                    x_rep = x[rep]
                    if(torch.count_nonzero(x_rep) > 0):
                        if self.ae:
                            x_rep = self.AEModels[mId].predict(x_rep)
                        xDup = x_rep[self.mask].repeat(21000, 1).moveaxis(0, 1)
                        for coco_batch in range(3):
                            x_preds_t = []
                            x_preds_batch = self.x_preds[mId][21000*coco_batch:21000*coco_batch+21000, self.mask]
                            x_preds_t = x_preds_batch.moveaxis(0, 1).to(self.device)
                            modelScore = PeC(xDup, x_preds_t).cpu()
                            
                            scores[21000*coco_batch:21000*coco_batch+21000] += modelScore
                
                    div +=1
            scores /= div
                    
            # Calculating the Average Pearson Across Samples
            topn_pearson = torch.topk(scores, topn)
            average_pearson += torch.mean(topn_pearson.values) 
            for rank, index in enumerate(topn_pearson.indices):
                # out[rank] = int(self.y_indices[index])
                out[rank] = int(index)
                ret_scores = topn_pearson.values
            
        # torch.save(out, latent_path + encModel + "/" + vector + "_coco_library_preds.pt")
        # print("Average Pearson Across Samples: {}".format(average_pearson)) 
        return out, ret_scores

    def predict(self, x, vector="images", topn=None):
        library = torch.load("/export/raid1/home/kneel027/nsd_local/preprocessed_data/{}_73k.pt".format(vector), map_location="cpu").reshape((73000, self.datasize[vector]))
        library = prune_vector(library)
        with torch.no_grad():
            if vector == "images" and topn is not None:
                x_preds = torch.zeros((x.shape[0], topn, self.datasize[vector])).cpu()
            else:
                x_preds = torch.zeros((x.shape[0], self.datasize[vector])).cpu()
            #pull topN parameter from config
            if topn is None:
                # but only for clip vectors
                if vector == "c_i":
                    if len(self.configList) > 1:
                        topn = self.config["LibraryAssembler"]["dualGuided"]
                    else:
                        topn = self.config["LibraryAssembler"][self.configList[0]]
                else:
                    topn = 1
            # Get best match for each sample
            for sample in tqdm(range(x.shape[0]), desc="predicting samples for {}".format(vector)):
                best_im_indices, _ = self.rankCoco(x[sample], topn=topn)
                if vector == "images":
                    if topn == 1:
                        x_preds[sample] = library[int(best_im_indices)].reshape((1, self.datasize[vector]))
                    else:
                        for idx, index in enumerate(best_im_indices):
                            x_preds[sample, idx] = library[int(index)].reshape((1, self.datasize[vector]))
                else:
                    vectors = torch.zeros((topn, self.datasize[vector]))
                    for idx, index in enumerate(best_im_indices):
                        vectors[idx] = library[int(index)].reshape((1, self.datasize[vector]))
                    x_preds[sample] = torch.mean(vectors, dim=0)
        return x_preds

    
    def benchmark(self, vector="c_i"):

        # Load data and targets
        _, _, x_test, _, _, target, _ = load_nsd(vector=vector, subject=self.subject, loader=False, average=False, nest=True, big=self.big)
        library = torch.load("/export/raid1/home/kneel027/nsd_local/preprocessed_data/{}_73k.pt".format(vector), map_location="cpu").reshape((73000, self.datasize[vector]))
        library = prune_vector(library)
        out = torch.zeros((x_test.shape[0], 1000, self.datasize[vector]))
        topn = 1000
        for sample in tqdm(range(x_test.shape[0]), desc="predicting samples for {}".format(vector)):
            best_im_indices, _ = self.rankCoco(x_test[sample], topn=topn)
            for idx, index in enumerate(best_im_indices):
                out[sample, idx] = library[int(index)].reshape((1, self.datasize[vector]))
        
        criterion = nn.MSELoss()
        
        PeC = PearsonCorrCoef(num_outputs=x_test.shape[0])
        target = target
        out = out
        vc = []
        l2 = []
        for i in tqdm(range(1, 1000), desc="Scoring progressive topn"):
            loss = criterion(torch.mean(out[:,0:i], dim=1), target)
            pearson_loss = torch.mean(PeC(torch.mean(out[:,0:i], dim=1).moveaxis(0,1), target.moveaxis(0,1).detach()))
            vc.append(float(pearson_loss))
            l2.append(float(loss))
        vc = np.array(vc)
        l2 = np.array(l2)
        print("Library Decoder: Subject: {}, Config: {}".format(self.subject, ", ".join(self.configList)))
        print("Vector Correlation Top {}: {}".format(float(np.argmax(vc)), float(np.max(vc))))
        print("Loss Top {}: {}".format(float(np.argmin(l2)), float(np.min(l2))))
        print("Vector Correlation Top 1: ", float(vc[0]))
        print("Loss Top 1:", float(l2[0]))
        print("Vector Correlation Top 10: ", float(vc[9]))
        print("Loss Top 10:", float(l2[9]))
        print("Vector Correlation Top 100: ", float(vc[99]))
        print("Loss Top 100:", float(l2[99]))
        print("Vector Correlation Top 500: ", float(vc[499]))
        print("Loss Top 500:", float(l2[499]))
        print("Vector Correlation Top 1000: ", float(vc[-1]))
        print("Loss Top 1000:", float(l2[-1]))
        # os.mkdir("logs/subject{sub}/library_assembler_scores/".format(sub=self.subject), exist_ok=True)
        # np.save("logs/subject{sub}/library_assembler_scores/S{sub}_{vec}_{config}_PeC.npy".format(sub=self.subject, vec=vector, config="_".join(self.configList)), vc)
        # np.save("logs/subject{sub}/library_assembler_scores/S{sub}_{vec}_{config}_L2.npy".format(sub=self.subject, vec=vector, config="_".join(self.configList)), l2)