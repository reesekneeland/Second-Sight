import torch
import numpy as np
import torch.nn as nn
from utils import *
import yaml
from tqdm import tqdm
from autoencoder  import AutoEncoder
from torchmetrics import PearsonCorrCoef


# Config options:
#   - AlexNet
#   - c_img_vd
#   - c_text_vd
class LibraryDecoder():
    def __init__(self, 
                 vector="images",
                 subject=1,
                 configList=["gnetEncoder"],
                 ae=True,
                 device="cuda",
                 mask=torch.full((11838,), True)
                 ):
        
        self.subject=subject
        with open("config.yml", "r") as yamlfile:
            self.config = yaml.load(yamlfile, Loader=yaml.FullLoader)[self.subject]
        self.AEModels = []
        self.EncModels = []
        self.configList = configList
        self.vector = vector
        self.device = device
        self.mask = mask
        self.ae = ae
        if(vector == "c_img_0" or vector == "c_text_0"):
            self.datasize = 768
        elif(vector == "z_img_mixer"):
            self.datasize = 16384
        elif(vector == "c_img_uc"):
            self.datasize = 1024
        elif(vector == "c_text_uc"):
            self.datasize = 78848
        elif(vector == "images"):
            self.datasize = 541875
        elif(vector == "z_vdvae"):
            self.datasize = 91168
            
        self.prep_path = "/export/raid1/home/kneel027/nsd_local/preprocessed_data/"
        self.latent_path = "latent_vectors/"

        for param in self.configList:
            self.EncModels.append(self.config[param]["modelId"])
            if param == "alexnetEncoder":
                if self.ae:
                    self.AEModels.append(AutoEncoder(config="alexnetAutoEncoder",
                                                    inference=True,
                                                    subject=self.subject,
                                                    device=self.device))
            elif param == "gnetEncoder":
                if self.ae:
                    self.AEModels.append(AutoEncoder(config="gnetAutoEncoder",
                                                    inference=True,
                                                    subject=self.subject,
                                                    device=self.device))
            elif param == "clipEncoder":
                if self.ae:
                    self.AEModels.append(AutoEncoder(config="clipAutoEncoder",
                                                    inference=True,
                                                    subject=self.subject,
                                                    device=self.device))
                
    def rankCoco(self, x, average=True, topn=1000):
        x_preds = []
        for model in self.EncModels:
            modelPreds = torch.load("{}/subject{}/{}/coco_brain_preds.pt".format(self.latent_path, self.subject, model), map_location=self.device)
            x_preds.append(modelPreds)
       
        y_full = torch.load("{}/{}_73k.pt".format(self.prep_path, self.vector)).reshape(73000, self.datasize)
        y = prune_vector(y_full)
        # print(x_preds.shape)
        if(average):
            # x = x[:, :, self.mask]
            x = x
        else:
            # x = x[:, 0, self.mask]
            x = x[:, 0]

        out = torch.zeros((x.shape[0], topn, self.datasize))
        ret_scores = torch.zeros((x.shape[0], topn))
        
        PeC = PearsonCorrCoef(num_outputs=21000).to(self.device)
        average_pearson = 0
        
        for sample in tqdm(range(x.shape[0]), desc="scanning library for {}".format(self.vector)):
            scores = torch.zeros((y.shape[0],))
            div = 0
            for mId, model in enumerate(self.EncModels):
                # print("Model: ", model)
                for rep in range(x.shape[1]):
                    x_rep = x[sample, rep]
                    if(torch.count_nonzero(x_rep) > 0):
                        if self.ae:
                            x_rep = self.AEModels[mId].predict(x_rep)
                        xDup = x_rep.repeat(21000, 1).moveaxis(0, 1).to(self.device)
                        for coco_batch in range(3):
                            x_preds_t = []
                            x_preds_batch = x_preds[mId][21000*coco_batch:21000*coco_batch+21000]
                            x_preds_t = x_preds_batch.moveaxis(0, 1).to(self.device)
                            modelScore = PeC(xDup, x_preds_t).cpu().detach()
                            scores[21000*coco_batch:21000*coco_batch+21000] += modelScore.detach()
                
                    div +=1
                # print("Best Score: ", torch.max(scores/div))
            scores /= div
                    
                # Calculating the Average Pearson Across Samples
            topn_pearson = torch.topk(scores, topn)
            average_pearson += torch.mean(topn_pearson.values.detach()) 
            for rank, index in enumerate(topn_pearson.indices):
                out[sample, rank] = y[index]
                ret_scores[sample] = topn_pearson.values.detach()
            
        # torch.save(out, latent_path + encModel + "/" + vector + "_coco_library_preds.pt")
        print("Average Pearson Across Samples: {}".format(average_pearson / x.shape[0])) 
        return out, ret_scores

    def predict(self, x, average=True, topn=None):
        #pull topN parameter from config
        if topn is None:
            # but only for clip vectors
            if self.vector == "c_img_uc":
                if len(self.configList) > 1:
                    topn = self.config["libraryDecoder"]["dualGuided"]
                else:
                    topn = self.config["libraryDecoder"][self.configList[0]]
            else:
                topn = 1
        x_preds, _ = self.rankCoco(x, average=average, topn=topn)
        topn_x_preds = torch.mean(x_preds[:,0:topn], dim=1)
        return topn_x_preds

    
    def benchmark(self, average=True):

        # Load data and targets
        _, _, x_test, _, _, target, _ = load_nsd(vector=self.vector, subject=self.subject, loader=False, average=False, nest=True)

        
        out, _ = self.rankCoco(x_test, average=average, topn=500)
        
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
        print("Vector Correlation Top 1: ", float(vc[0]))
        print("Loss Top 1:", float(l2[0]))
        print("Vector Correlation Top 10: ", float(vc[9]))
        print("Loss Top 10:", float(l2[9]))
        print("Vector Correlation Top 100: ", float(vc[99]))
        print("Loss Top 100:", float(l2[99]))
        print("Vector Correlation Top 500: ", float(vc[499]))
        print("Loss Top 1000:", float(l2[-1]))
        print("Vector Correlation Top {}: {}".format(float(np.argmax(vc)), float(np.max(vc))))
        print("Loss Top Top {}: {}".format(float(np.argmin(l2)), float(np.min(l2))))
        np.save("logs/subject{sub}/library_decoder_scores/S{sub}_{vec}_{config}_PeC.npy".format(sub=self.subject, vec=self.vector, config="_".join(self.configList)), vc)
        np.save("logs/subject{sub}/library_decoder_scores/S{sub}_{vec}_{config}_L2.npy".format(sub=self.subject, vec=self.vector, config="_".join(self.configList)), l2)