import torch.nn as nn
from utils import *
import torch
import torch as T
import torch.nn.init as I
import torch.nn.functional as F

#CLIP Encoder class
class CLIPEncoderModel(torch.nn.Module):
    def __init__(self, x_size):
        super(CLIPEncoderModel, self).__init__()
        self.linear = nn.Linear(1024, 15000)
        self.linear2 = nn.Linear(15000, 15000)
        self.outlayer = nn.Linear(15000, x_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        y_pred = self.relu(self.linear(x))
        y_pred = self.relu(self.linear2(y_pred))
        y_pred = self.outlayer(y_pred)
        return y_pred

#AutoEncoder model class
class AutoEncoderModel(torch.nn.Module):
    def __init__(self, x_size):
        super(AutoEncoderModel, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(x_size, 5000),
            torch.nn.ReLU(),
            torch.nn.Linear(5000, 1000),
            torch.nn.ReLU(),
            torch.nn.Linear(1000, 500)
        )
         
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(500, 1000),
            torch.nn.ReLU(),
            torch.nn.Linear(1000, 5000),
            torch.nn.ReLU(),
            torch.nn.Linear(5000, x_size)
        )
 
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

#Subclass for GNET
class TrunkBlock(nn.Module):
    def __init__(self, feat_in, feat_out):
        super(TrunkBlock, self).__init__()
        self.conv1 = nn.Conv2d(feat_in, int(feat_out*1.), kernel_size=3, stride=1, padding=1, dilation=1)
        self.drop1 = nn.Dropout2d(p=0.5, inplace=False)
        self.bn1 = nn.BatchNorm2d(feat_in, eps=1e-05, momentum=0.25, affine=True, track_running_stats=True)

        I.xavier_normal_(self.conv1.weight, gain=I.calculate_gain('relu'))
        I.constant_(self.conv1.bias, 0.0) # current
        
    def forward(self, x):
        return F.relu(self.conv1(self.drop1(self.bn1(x))))

#Subclass for GNET
class PreFilter(nn.Module):
    def __init__(self):
        super(PreFilter, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )        
        
    def forward(self, x):
        c1 = self.conv1(x)
        y = self.conv2(c1)
        return y 

#Subclass for GNET
class EncStage(nn.Module):
    def __init__(self, trunk_width=64, pass_through=64):
        super(EncStage, self).__init__()
        self.conv3  = nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=0)
        self.drop1  = nn.Dropout2d(p=0.5, inplace=False) ##
        self.bn1    = nn.BatchNorm2d(192, eps=1e-05, momentum=0.25, affine=True, track_running_stats=True) ##
        self.pool1  = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ##
        self.tw = int(trunk_width)
        self.pt = int(pass_through)
        ss = (self.tw + self.pt)
        self.conv4a  = TrunkBlock(128, ss)
        self.conv5a  = TrunkBlock(ss, ss)
        self.conv6a  = TrunkBlock(ss, ss)
        self.conv4b  = TrunkBlock(ss, ss)
        self.conv5b  = TrunkBlock(ss, ss)
        self.conv6b  = TrunkBlock(ss, self.tw)
        ##
        I.xavier_normal_(self.conv3.weight, gain=I.calculate_gain('relu'))        
        I.constant_(self.conv3.bias, 0.0)
        
    def forward(self, x):
        c3 = (F.relu(self.conv3(self.drop1(self.bn1(x))), inplace=False))
        c4a = self.conv4a(c3)
        c4b = self.conv4b(c4a)
        c5a = self.conv5a(self.pool1(c4b))
        c5b = self.conv5b(c5a)
        c6a = self.conv6a(c5b)
        c6b = self.conv6b(c6a)

        return [T.cat([c3, c4a[:,:self.tw], c4b[:,:self.tw]], dim=1), 
                T.cat([c5a[:,:self.tw], c5b[:,:self.tw], c6a[:,:self.tw], c6b], dim=1)], c6b

#Subclass for GNET
class Encoder(nn.Module):
    def __init__(self, mu, trunk_width, pass_through=64 ):
        super(Encoder, self).__init__()
        self.mu = nn.Parameter(T.from_numpy(mu), requires_grad=False) #.to(device)
        self.pre = PreFilter()
        self.enc = EncStage(trunk_width, pass_through) 

    def forward(self, x):
        fmaps, h = self.enc(self.pre(x - self.mu))
        return x, fmaps, h

#Main GNET model class
class Torch_LayerwiseFWRF(nn.Module):
    def __init__(self, fmaps, nv=1, pre_nl=None, post_nl=None, dtype=np.float32):
        super(Torch_LayerwiseFWRF, self).__init__()
        self.fmaps_shapes = [list(f.size()) for f in fmaps]
        self.nf = np.sum([s[1] for s in self.fmaps_shapes])
        self.pre_nl  = pre_nl
        self.post_nl = post_nl
        self.nv = nv
        ##
        self.rfs = []
        self.sm = nn.Softmax(dim=1)
        for k,fm_rez in enumerate(self.fmaps_shapes):
            rf = nn.Parameter(T.tensor(np.ones(shape=(self.nv, fm_rez[2], fm_rez[2]), dtype=dtype), requires_grad=True))
            self.register_parameter('rf%d'%k, rf)
            self.rfs += [rf,]
        #self.w  = nn.Parameter(T.tensor(np.random.normal(0, 0.001, size=(self.nv, self.nf)).astype(dtype=dtype), requires_grad=True))
        #self.b  = nn.Parameter(T.tensor(np.full(fill_value=0.0, shape=(self.nv,), dtype=dtype), requires_grad=True))
        self.w  = nn.Parameter(T.tensor(np.random.normal(0, 0.01, size=(self.nv, self.nf)).astype(dtype=dtype), requires_grad=True))
        self.b  = nn.Parameter(T.tensor(np.random.normal(0, 0.01, size=(self.nv,)).astype(dtype=dtype), requires_grad=True))
        
    def forward(self, fmaps):
        phi = []
        for fm,rf in zip(fmaps, self.rfs): #, self.scales):
            g = self.sm(T.flatten(rf, start_dim=1))
            f = T.flatten(fm, start_dim=2)  # *s
            if self.pre_nl is not None:          
                f = self.pre_nl(f)
            # fmaps : [batch, features, space]
            # v     : [nv, space]
            phi += [T.tensordot(g, f, dims=[[1],[2]]),] # apply pooling field and add to list.
            # phi : [nv, batch, features] 
        Phi = T.cat(phi, dim=2)
        if self.post_nl is not None:
            Phi = self.post_nl(Phi)
        vr = T.squeeze(T.bmm(Phi, T.unsqueeze(self.w,2))).t() + T.unsqueeze(self.b,0)
        return vr